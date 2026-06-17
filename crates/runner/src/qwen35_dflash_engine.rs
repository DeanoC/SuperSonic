//! Dense Qwen DFlash speculative-decoding engine (M3.3).
//!
//! Drives a dense Qwen low-bit target and the DFlash draft together through
//! the speculative loop described in `docs/dflash.md` §5–§6:
//!
//! 1. Prefill target with prompt (via `prefill_with_taps`); keep the last
//!    prompt token's taps as the first draft context.
//! 2. Per round:
//!    a. Draft `forward()` on `noise_embedding = embed([bonus_seed, MASK,…])`
//!       with `target_hidden = taps` → B draft candidates (via target's
//!       `lm_head`).
//!    b. Snapshot linear state of the target.
//!    c. Verify: one `persistent_decode_4b` megakernel launch at positions
//!       `[L, L+B)` (see `DecodeEngine::verify_block_fused_decode`). The
//!       launch shares the live sequence's KV/linear buffers across all B
//!       batch slots with `seqlen_offset[b] = L + b`, so each position
//!       reads the K/V written by prior positions within the same launch
//!       (M4.3).
//!    d. Compute `accepted` = longest prefix match vs target's greedy
//!       per-position picks (§6).
//!    e. Restore linear state, then re-decode each committed position via
//!       `decode_step_with_taps_kernel`, stacking per-position tap rows so
//!       the next round's draft receives `ctx_len = accepted + 1` taps.
//!    f. Rewind target's full-attention `kv_filled` to `L + accepted + 1`.
//!    g. Crop the draft's KV cache to the new committed length.

use std::env;
use std::sync::{Arc, Once};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::state::LinearStateSnapshot;
use qwen35::weights::Qwen35Weights;
use qwen35_dflash as dflash;

use crate::bakes::{effective_quant_profile, load_qwen35_weights};
use crate::decode_engine::DecodeEngine;
use crate::dflash_ddtree::{
    accepted_tokens_for_path, build_ddtree, build_verify_plan, extract_draft_topk_bf16,
    follow_verified_tree, DDTree,
};
use crate::prefill_engine::{PrefillAppendVerifyResult, PrefillTreeVerifyResult};
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};
use crate::Cli;

const DDTREE_DEFAULT_BUDGET: usize = 22;
const DDTREE_DEFAULT_TOP_K: usize = 8;

#[derive(Debug, Clone)]
struct DFlashDDTreeProbeConfig {
    budget: usize,
    top_k: usize,
    temperature: f32,
    chain_seed: bool,
}

#[derive(Debug, Clone)]
struct DFlashDDTreeProbeRound {
    depth_limit: usize,
    top_k: usize,
    budget: usize,
    nodes: usize,
    width: usize,
    max_depth: usize,
    top1_head: Vec<u32>,
    tree: DDTree,
}

#[derive(Debug, Clone)]
struct DraftRoundOutput {
    candidates: Vec<u32>,
    ddtree_probe: Option<DFlashDDTreeProbeRound>,
}

/// Run the dense-Qwen DFlash speculative decoder. Parallels
/// `phi4_engine::run_phi4` in shape — but drives both target and draft
/// models through the speculative loop.
pub fn run_qwen35_dflash(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    // --------- 1. Validate CLI combo -------------------------------------
    if !matches!(
        model_variant,
        ModelVariant::Qwen3_5_9B | ModelVariant::Qwen3_6_27B
    ) {
        bail!(
            "--dflash is supported for --model qwen3.5-9b and qwen3.6-27b \
             (got {model_variant})"
        );
    }
    let profile = effective_quant_profile(cli)?;
    if !matches!(
        profile,
        model_store::manifest::QuantProfile::Int4Gptq
            | model_store::manifest::QuantProfile::Int4Awq
            | model_store::manifest::QuantProfile::Int4Autoround
            | model_store::manifest::QuantProfile::Int4Hqq
            | model_store::manifest::QuantProfile::Q4Km
            | model_store::manifest::QuantProfile::Q4KmGptq
    ) {
        bail!("--dflash requires a low-bit target bake (--int4, --q4km, or --q4km-gptq)");
    }
    let draft_dir = cli
        .dflash_draft_dir
        .as_ref()
        .ok_or_else(|| anyhow!("--dflash requires --dflash-draft-dir"))?;
    if cli.batch_size != 1 {
        bail!("--dflash requires --batch-size=1 (single-sequence speculative loop)");
    }
    if cli.kv_fp8 {
        bail!("--dflash does not support --kv-fp8 at M3 (snapshot/restore covers linear only)");
    }
    if cli.oracle_prefill || cli.validate || cli.gpu_validate {
        bail!("--dflash does not support --oracle-prefill / --validate / --gpu-validate at M3");
    }

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => *p,
        FamilyParams::Qwen3Moe(_)
        | FamilyParams::Qwen36Moe(_)
        | FamilyParams::Gemma4(_)
        | FamilyParams::Phi4(_)
        | FamilyParams::Llama31(_) => {
            unreachable!("run_qwen35_dflash dispatched for non-qwen35 variant");
        }
    };
    if !params.use_4b_kernel {
        bail!("--dflash requires the 4B kernel path");
    }
    let weight_prefix: &'static str = params.weight_prefix;

    // --------- 2. Tokenizer + target config ------------------------------
    let text_config = {
        let cfg = qwen35::config::load_config(&cli.model_dir)
            .map_err(|e| anyhow!("loading target config.json: {e}"))?;
        cfg.text_config
    };
    eprintln!(
        "[dflash] target: hidden={} layers={} vocab={} heads={} kv_heads={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
    );

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = crate::load_tokenizer(&tokenizer_path)?;
    let prompt_ids = crate::resolve_prompt_token_ids(cli, &tokenizer)?;

    // --------- 3. VRAM estimate (low-bit target + ~2 GiB draft) ----------
    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    if context_tokens < prompt_ids.len() + cli.max_new_tokens {
        bail!(
            "--context-size {context_tokens} < prompt_tokens {} + max_new_tokens {}",
            prompt_ids.len(),
            cli.max_new_tokens,
        );
    }
    let kv_per_token = text_config.kv_bytes_per_token(ScalarType::BF16.size_in_bytes());
    let target_fixed = match profile {
        model_store::manifest::QuantProfile::Q4Km => (entry.vram.fixed_bytes as f64 * 0.28) as u64,
        model_store::manifest::QuantProfile::Q4KmGptq
        | model_store::manifest::QuantProfile::Int4Gptq
        | model_store::manifest::QuantProfile::Int4Awq
        | model_store::manifest::QuantProfile::Int4Autoround
        | model_store::manifest::QuantProfile::Int4Hqq => {
            (entry.vram.fixed_bytes as f64 * 0.37) as u64
        }
        _ => (entry.vram.fixed_bytes as f64 * 0.37) as u64,
    };
    let target_kv = kv_per_token * context_tokens as u64;
    let draft_fixed: u64 = 2 * 1024 * 1024 * 1024; // ~2 GiB for DFlash draft weights + scratch
    let estimated =
        ((target_fixed + target_kv + draft_fixed) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (target {profile} weights={:.2}GiB + target KV={:.2}GiB + draft={:.2}GiB) \
         available={:.1}GiB",
        gib(estimated),
        gib(target_fixed),
        gib(target_kv),
        gib(draft_fixed),
        gib(total_vram),
    );
    if estimated > total_vram {
        bail!(
            "Insufficient VRAM for DFlash at context={context_tokens}: need ~{:.2}GiB, \
             GPU has {:.1}GiB. Reduce --context-size.",
            gib(estimated),
            gib(total_vram),
        );
    }

    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    // --------- 4. Load target weights (selected low-bit bake) ------------
    let t0 = Instant::now();
    let target_weights =
        load_target_lowbit_weights(cli, model_variant, &text_config, ordinal, weight_prefix)?;
    eprintln!(
        "[weights] target ({profile}, group_size={}) loaded in {:.0}ms",
        target_weights.int4_group_size,
        t0.elapsed().as_millis(),
    );
    if !target_weights.is_int4 {
        bail!("--dflash target loader did not produce low-bit weights for {profile}");
    }

    // Grab Arc clones of embed_tokens + lm_head before moving weights into
    // the engine — the draft borrows them without owning them (docs §7).
    let target_embed: Arc<GpuBuffer> = Arc::clone(&target_weights.embed_tokens);
    let target_lm_head: Arc<GpuBuffer> = Arc::clone(&target_weights.lm_head);

    // --------- 5. Build the target DecodeEngine --------------------------
    let required_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        text_config.num_attention_heads,
        text_config.head_dim,
        context_tokens,
        params.kv_chunk_size,
    );
    let attn_scratch_floats = params.attn_scratch_floats.max(required_attn_scratch);

    let mut target_engine = DecodeEngine::new(
        target_weights,
        ordinal,
        params.proj_buf_floats,
        attn_scratch_floats,
        params.kv_chunk_size,
        /* use_4b_kernel */ true,
        cli.prefill_chunk_size,
        /* kv_fp8 */ false,
        /* batch_size */ 1,
    )?;

    // --------- 6. Load DFlash draft --------------------------------------
    let draft_config =
        dflash::load_config(draft_dir).map_err(|e| anyhow!("load draft config.json: {e}"))?;
    if draft_config.num_target_layers != text_config.num_hidden_layers {
        bail!(
            "draft num_target_layers={} != target layers={}",
            draft_config.num_target_layers,
            text_config.num_hidden_layers,
        );
    }
    let tap_layers: Vec<usize> = if let Some(override_taps) = cli.dflash_tap_layers.as_ref() {
        parse_tap_override(override_taps, draft_config.num_target_layers)?
    } else {
        draft_config
            .dflash_config
            .target_layer_ids
            .iter()
            .map(|&v| v as usize)
            .collect()
    };
    eprintln!(
        "[dflash] draft: layers={} hidden={} block_size={} taps={:?} mask_id={}",
        draft_config.num_hidden_layers,
        draft_config.hidden_size,
        draft_config.block_size,
        tap_layers,
        draft_config.dflash_config.mask_token_id,
    );
    if tap_layers.len() != draft_config.num_taps() {
        bail!(
            "tap layer count {} mismatches draft's fc.in_features implied count {}",
            tap_layers.len(),
            draft_config.num_taps(),
        );
    }
    if draft_config.hidden_size != text_config.hidden_size {
        bail!(
            "draft hidden_size {} != target hidden_size {}",
            draft_config.hidden_size,
            text_config.hidden_size,
        );
    }
    // The draft borrows the target's `embed_tokens` + `lm_head` (docs §7)
    // and uses draft-side token IDs to index them, so the two checkpoints
    // must share the same vocabulary. Mismatch would silently read past
    // the target embedding rows or truncate draft logits. Verify both
    // before spending any more time on state / scratch allocation.
    if draft_config.vocab_size != text_config.vocab_size {
        bail!(
            "draft vocab_size {} != target vocab_size {} — the draft borrows the target's \
             embed_tokens / lm_head and must share its vocabulary",
            draft_config.vocab_size,
            text_config.vocab_size,
        );
    }
    let mask_id = draft_config.dflash_config.mask_token_id;
    if (mask_id as usize) >= text_config.vocab_size {
        bail!(
            "draft mask_token_id {mask_id} is out of range for target vocab_size {} — \
             the MASK token is looked up in the target embedding table each round",
            text_config.vocab_size,
        );
    }

    let draft_weights = dflash::DFlashWeights::load(
        draft_dir,
        &draft_config,
        ordinal,
        Arc::clone(&target_embed),
        Arc::clone(&target_lm_head),
    )
    .map_err(|e| anyhow!("load draft weights: {e}"))?;
    eprintln!("[dflash] draft weights loaded");

    let draft_ctx_capacity = context_tokens.max(1);
    let draft_max_ctx = cli
        .context_size
        .map(|c| c.max(draft_config.block_size * 4))
        .unwrap_or_else(|| (draft_ctx_capacity + draft_config.block_size).max(1024));
    let draft_rotary = dflash::RotaryTables::build(&draft_config, ordinal, draft_max_ctx)
        .map_err(|e| anyhow!("build draft RoPE: {e}"))?;
    let mut draft_scratch = dflash::state::DFlashScratch::new_with_ctx_capacity(
        ordinal,
        &draft_config,
        draft_ctx_capacity,
    )
    .map_err(|e| anyhow!("alloc draft scratch: {e}"))?;
    let mut draft_noise_embedding = GpuBuffer::zeros(
        ordinal,
        ScalarType::BF16,
        &[1, draft_config.block_size, draft_config.hidden_size],
    )
    .map_err(|e| anyhow!("alloc draft noise embedding scratch: {e}"))?;
    let mut draft_state = dflash::state::DFlashState::new(ordinal, &draft_config, draft_max_ctx)
        .map_err(|e| anyhow!("alloc draft state: {e}"))?;

    // --------- 7. Prefill target + capture first-round taps --------------
    let prefill_start = Instant::now();
    let prefill_result = target_engine.prefill_native_with_taps(&prompt_ids, &tap_layers)?;
    eprintln!(
        "[prefill] {} tokens in {:.0}ms",
        prompt_ids.len(),
        prefill_start.elapsed().as_millis(),
    );
    let per_tap_row_bytes =
        tap_layers.len() * text_config.hidden_size * ScalarType::BF16.size_in_bytes();
    let mut tap_history: Vec<u8> = match prefill_result.tap_hiddens_all.as_ref() {
        Some(per_tap) => flatten_tap_history(per_tap, prompt_ids.len(), text_config.hidden_size)?,
        None => flatten_tap_history(
            &prefill_result.tap_hiddens.unwrap_or_default(),
            1,
            text_config.hidden_size,
        )?,
    };
    let mut tap_history_len: usize = if per_tap_row_bytes == 0 {
        0
    } else {
        tap_history.len() / per_tap_row_bytes
    };
    if tap_history_len == 0 {
        bail!("DFlash prefill did not produce any target tap history");
    }

    // Sample the first bonus_seed from prefill's last logits (greedy @ T=0).
    let mut bonus_seed: u32 = DecodeEngine::greedy_sample(&prefill_result.logits);

    // kv_filled count on any full-attention layer is equal to prompt_len
    // after prefill. Track committed length separately so we don't depend
    // on state internals.
    let mut committed_len: usize = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let eos_ids: Vec<u32> = text_config.eos_token_ids();
    // Qwen3.5-9B keeps the historical fused-verify default of B=3. The
    // Qwen3.6-27B comparison path defaults to the draft checkpoint's full
    // block size (16 for the Lucebox DFlash draft); if the fused verifier
    // cannot fit the model shape in LDS, the loop falls back to sequential
    // verify below.
    const DEFAULT_FUSED_BLOCK_SIZE: usize = 3;
    let default_block_size = if matches!(model_variant, ModelVariant::Qwen3_6_27B) {
        draft_config.block_size
    } else {
        DEFAULT_FUSED_BLOCK_SIZE.min(draft_config.block_size)
    };
    let block_size = cli.dflash_block.unwrap_or(default_block_size);
    if block_size == 0 || block_size > draft_config.block_size {
        bail!(
            "--dflash-block must be in 1..={} (got {block_size})",
            draft_config.block_size,
        );
    }
    let tap_history_capacity = tap_history_len + cli.max_new_tokens + block_size;
    let mut tap_history_gpu = GpuBuffer::zeros(
        ordinal,
        ScalarType::BF16,
        &[
            tap_history_capacity,
            tap_layers.len() * text_config.hidden_size,
        ],
    )
    .map_err(|e| anyhow!("alloc GPU tap history: {e}"))?;
    upload_taps_to_gpu_history(&mut tap_history_gpu, 0, per_tap_row_bytes, &tap_history)?;

    // --------- 8. Speculative loop ---------------------------------------
    let profile_ffi = env::var_os("SUPERSONIC_DFLASH_PROFILE_FFI").is_some();
    if profile_ffi {
        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
        kernel_ffi::prefill_ffi::ffi_profile_set_enabled(true);
        kernel_ffi::prefill_ffi::ffi_profile_reset();
    }
    let decode_start = Instant::now();
    let mut rounds_run: usize = 0;
    let mut accepted_total: usize = 0;
    // Per-stage timing accumulators. Reported alongside rounds summary so
    // anyone profiling DFlash can see which stage dominates wall-clock
    // before deciding what to optimize next.
    let mut ms_draft: f64 = 0.0;
    let mut ms_verify: f64 = 0.0;
    let mut ms_redecode: f64 = 0.0;
    let mut ms_rollback: f64 = 0.0;
    let trace_accept = env::var_os("SUPERSONIC_DFLASH_TRACE_ACCEPT").is_some();
    let ddtree_probe = ddtree_probe_config_from_env()?;
    let ddtree_verify = ddtree_verify_config_from_env()?;
    let ddtree_direct_rollback = env::var_os("SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK").is_some();
    if let Some(config) = ddtree_probe.as_ref() {
        eprintln!(
            "[dflash-ddtree-probe] enabled budget={} top_k={} temp={} chain_seed={}",
            config.budget, config.top_k, config.temperature, config.chain_seed
        );
    }
    if let Some(config) = ddtree_verify.as_ref() {
        let commit_mode = if ddtree_direct_rollback {
            "tree-rollback"
        } else {
            "append-reverify"
        };
        eprintln!(
            "[dflash-ddtree-verify] enabled budget={} top_k={} temp={} chain_seed={} commit={}",
            config.budget, config.top_k, config.temperature, config.chain_seed, commit_mode
        );
    }
    while generated_ids.len() < cli.max_new_tokens {
        if !cli.ignore_eos && eos_ids.contains(&bonus_seed) {
            generated_ids.push(bonus_seed);
            break;
        }

        let remaining_budget = cli.max_new_tokens - generated_ids.len();
        if remaining_budget == 1 {
            if trace_accept {
                eprintln!(
                    "[ss-trace] final carried seed l={} bonus_seed={} commit_n=1",
                    committed_len, bonus_seed
                );
            }
            generated_ids.push(bonus_seed);
            break;
        }

        rounds_run += 1;
        let l = committed_len;
        let ddtree_build_config = ddtree_verify.as_ref().or(ddtree_probe.as_ref());
        let verify_len = dflash_verify_len_for_round(remaining_budget, block_size);
        let ddtree_depth_limit = if ddtree_verify.is_some() {
            remaining_budget.saturating_sub(1)
        } else {
            block_size.saturating_sub(1)
        };

        // 8a. Draft forward: Lucebox's draft graph is stateless and attends
        // over a target-feature history window plus the current noise block.
        // Feed the tail of the target tap history, not only the newest rows.
        let draft_ctx = tap_history_len.min(draft_scratch.ctx_capacity);
        let tap_start_row = tap_history_len - draft_ctx;
        let t_draft = Instant::now();
        let draft_output = draft_forward_and_sample(
            &mut draft_state,
            &mut draft_scratch,
            &mut draft_noise_embedding,
            &draft_rotary,
            &draft_weights,
            &target_engine,
            &tap_history_gpu,
            tap_start_row,
            draft_ctx,
            bonus_seed,
            block_size,
            draft_config.dflash_config.mask_token_id,
            ddtree_build_config,
            ddtree_depth_limit,
            ordinal,
        )?;
        ms_draft += t_draft.elapsed().as_secs_f64() * 1000.0;
        if ddtree_probe.is_some() {
            if let Some(probe) = draft_output.ddtree_probe.as_ref() {
                eprintln!(
                    "[dflash-ddtree-probe] round={} L={} K={} budget={} nodes={} width={} max_depth={} top1_head={:?}",
                    rounds_run - 1,
                    probe.depth_limit,
                    probe.top_k,
                    probe.budget,
                    probe.nodes,
                    probe.width,
                    probe.max_depth,
                    probe.top1_head,
                );
            }
        }
        let draft_tree = draft_output
            .ddtree_probe
            .as_ref()
            .map(|round| round.tree.clone());
        let draft_candidates = draft_output.candidates;

        // 8b. Verify: one `persistent_decode_4b` megakernel launch at
        //     positions `[l, l+B)`. Shared-cache BatchSeqDesc aliases the
        //     live sequence's KV/linear buffers across all B batch slots
        //     with `seqlen_offset[b] = l + b`; the kernel runs the B
        //     iterations sequentially on block 0 within a single layer so
        //     position b reads the K/V written by positions 0..b of the
        //     same launch.
        let t_verify = Instant::now();
        let verify_output = if ddtree_verify.is_some() {
            let tree = draft_tree
                .as_ref()
                .ok_or_else(|| anyhow!("DDTree verify enabled but draft tree was not built"))?;
            verify_ddtree_for_dflash(
                &mut target_engine,
                draft_candidates[0],
                tree,
                l,
                &tap_layers,
                ddtree_direct_rollback,
            )?
        } else {
            let gpu_tap_history = if dflash_gpu_tap_history_enabled() {
                Some((&mut tap_history_gpu, tap_history_len, per_tap_row_bytes))
            } else {
                None
            };
            verify_block_for_dflash(
                &mut target_engine,
                &draft_candidates[..verify_len],
                l,
                &tap_layers,
                gpu_tap_history,
            )?
        };
        ms_verify += t_verify.elapsed().as_secs_f64() * 1000.0;
        let target_next_ids = verify_output.greedy_ids()?;
        let target_next: Vec<u32> = if trace_accept {
            target_next_ids.clone()
        } else {
            Vec::new()
        };

        // 8d. Accept check. Chain mode compares adjacent draft positions;
        // DDTree mode follows the target posterior through matching child
        // edges and commits that accepted branch path.
        let (accept_n, carried_seed, mut committed_block, mut accepted_tree_indices) =
            match &verify_output {
                DFlashVerifyOutput::Tree(tree_output) => {
                    let (indices, carried, _terminal_index) =
                        follow_verified_tree(&tree_output.tree, &target_next_ids);
                    let tokens = accepted_tokens_for_path(
                        tree_output.root_token,
                        &tree_output.tree,
                        &indices,
                    );
                    (tokens.len(), carried, tokens, Some(indices))
                }
                _ => {
                    let mut accept_n = 1usize;
                    while accept_n < verify_len {
                        if accept_n > target_next_ids.len() {
                            bail!(
                                "captured prefill verifier returned {} greedy IDs, insufficient for accept_n={accept_n} verify_len={verify_len}",
                                target_next_ids.len()
                            );
                        }
                        let pred = target_next_ids[accept_n - 1];
                        if pred == draft_candidates[accept_n] {
                            accept_n += 1;
                        } else {
                            break;
                        }
                    }
                    let carried_seed = *target_next_ids
                        .get(accept_n - 1)
                        .ok_or_else(|| anyhow!("target verifier returned no carried seed"))?;
                    (
                        accept_n,
                        carried_seed,
                        draft_candidates[..accept_n].to_vec(),
                        None,
                    )
                }
            };

        let accepted_len = accept_n.min(remaining_budget);
        committed_block.truncate(accepted_len);
        if let Some(indices) = accepted_tree_indices.as_mut() {
            indices.truncate(accepted_len);
        }
        let finish_after_commit = accepted_len >= remaining_budget
            || (!cli.ignore_eos && committed_block.iter().any(|t| eos_ids.contains(t)));
        if trace_accept {
            eprintln!(
                "[ss-trace] round={} l={} draft_ctx={} verify_len={} bonus_seed={} draft={:?} target_next={:?} accept_n={} carried_seed={} commit_n={} committed={:?} tree_indices={:?}",
                rounds_run - 1,
                l,
                draft_ctx,
                verify_len,
                bonus_seed,
                draft_candidates,
                target_next,
                accept_n,
                carried_seed,
                accepted_len,
                committed_block,
                accepted_tree_indices,
            );
        }

        let mut next_bonus_seed: Option<u32> = None;
        if !finish_after_commit {
            match verify_output {
                DFlashVerifyOutput::Captured(result) => {
                    let t_rollback = Instant::now();
                    if let Some(per_tap) = result.tap_hiddens_all.as_ref() {
                        let taps_bytes =
                            flatten_tap_history(per_tap, verify_len, text_config.hidden_size)?;
                        let expected_tap_bytes = verify_len * per_tap_row_bytes;
                        if taps_bytes.len() != expected_tap_bytes {
                            bail!(
                                "captured prefill verifier returned {} tap bytes, expected {}",
                                taps_bytes.len(),
                                expected_tap_bytes,
                            );
                        }
                        let committed_tap_bytes = accepted_len * per_tap_row_bytes;
                        upload_taps_to_gpu_history(
                            &mut tap_history_gpu,
                            tap_history_len,
                            per_tap_row_bytes,
                            &taps_bytes[..committed_tap_bytes],
                        )?;
                        tap_history.extend_from_slice(&taps_bytes[..committed_tap_bytes]);
                    } else if !dflash_gpu_tap_history_enabled() {
                        bail!("captured prefill verifier did not return tap history");
                    }
                    if accepted_len == verify_len {
                        target_engine.commit_prefill_append_full_accept_owned(result)?;
                    } else {
                        target_engine.commit_prefill_append_verify_owned(result, accepted_len)?;
                    }
                    tap_history_len += accepted_len;
                    next_bonus_seed = Some(carried_seed);
                    ms_rollback += t_rollback.elapsed().as_secs_f64() * 1000.0;
                }
                DFlashVerifyOutput::CapturedWindowed(result) => {
                    let t_rollback = Instant::now();
                    let mut remaining = accepted_len;
                    let mut committed_taps = Vec::with_capacity(accepted_len * per_tap_row_bytes);
                    let mut copied_tap_rows = 0usize;
                    let mut committed = false;
                    for segment in &result.segments {
                        if remaining == 0 {
                            break;
                        }
                        let expected_start = accepted_len - remaining;
                        if segment.start != expected_start {
                            bail!(
                                "windowed prefill segment start {} != expected {}",
                                segment.start,
                                expected_start
                            );
                        }
                        let take = remaining.min(segment.len);
                        if let Some(per_tap) = segment.result.tap_hiddens_all.as_ref() {
                            let taps_bytes =
                                flatten_tap_history(per_tap, segment.len, text_config.hidden_size)?;
                            let take_bytes = take * per_tap_row_bytes;
                            if taps_bytes.len() < take_bytes {
                                bail!(
                                    "windowed prefill tap segment too short: got {} need {}",
                                    taps_bytes.len(),
                                    take_bytes
                                );
                            }
                            upload_taps_to_gpu_history(
                                &mut tap_history_gpu,
                                tap_history_len + copied_tap_rows,
                                per_tap_row_bytes,
                                &taps_bytes[..take_bytes],
                            )?;
                            committed_taps.extend_from_slice(&taps_bytes[..take_bytes]);
                        } else if !dflash_gpu_tap_history_enabled() {
                            bail!("windowed prefill verifier did not return tap history");
                        }
                        copied_tap_rows += take;

                        if remaining <= segment.len {
                            if remaining == segment.len {
                                target_engine.commit_prefill_append_full_accept(&segment.result)?;
                            } else {
                                target_engine
                                    .commit_prefill_append_verify(&segment.result, remaining)?;
                            }
                            committed = true;
                            remaining = 0;
                            break;
                        }
                        remaining -= segment.len;
                    }
                    if !committed || remaining != 0 {
                        bail!(
                            "windowed prefill commit could not cover accepted_len={} with {} segments",
                            accepted_len,
                            result.segments.len()
                        );
                    }
                    if copied_tap_rows != accepted_len {
                        bail!(
                            "windowed prefill verifier copied {} tap rows, expected {}",
                            copied_tap_rows,
                            accepted_len,
                        );
                    }
                    tap_history.extend_from_slice(&committed_taps);
                    tap_history_len += accepted_len;
                    next_bonus_seed = Some(carried_seed);
                    ms_rollback += t_rollback.elapsed().as_secs_f64() * 1000.0;
                }
                DFlashVerifyOutput::Tree(result) => {
                    let t_rollback = Instant::now();
                    let indices = accepted_tree_indices
                        .as_ref()
                        .ok_or_else(|| anyhow!("DDTree commit missing accepted tree indices"))?;
                    let taps_bytes = if result.result.rollback.is_some() {
                        target_engine.commit_prefill_tree_verify(
                            &result.result,
                            indices,
                            accepted_len,
                        )?;
                        let per_tap = result.result.tap_hiddens_all.as_ref().ok_or_else(|| {
                            anyhow!("tree prefill verifier did not return tap history")
                        })?;
                        flatten_tap_history_indices(
                            per_tap,
                            result.tree.width(),
                            text_config.hidden_size,
                            indices,
                        )?
                    } else {
                        let append_result = target_engine.verify_block_prefill_append_captured(
                            &committed_block,
                            l,
                            &tap_layers,
                        )?;
                        let append_next = append_result.target_next.as_ref().ok_or_else(|| {
                            anyhow!("append reverify did not return greedy target IDs")
                        })?;
                        let append_seed = append_next
                            .get(accepted_len - 1)
                            .copied()
                            .ok_or_else(|| anyhow!("append reverify returned no carried seed"))?;
                        if append_seed != carried_seed {
                            bail!(
                                "DDTree carried seed mismatch after append reverify: tree={} append={}",
                                carried_seed,
                                append_seed
                            );
                        }
                        target_engine.commit_prefill_append_verify(&append_result, accepted_len)?;
                        let per_tap = append_result
                            .tap_hiddens_all
                            .as_ref()
                            .ok_or_else(|| anyhow!("append reverify did not return tap history"))?;
                        flatten_tap_history(per_tap, accepted_len, text_config.hidden_size)?
                    };
                    let expected_tap_bytes = accepted_len * per_tap_row_bytes;
                    if taps_bytes.len() != expected_tap_bytes {
                        bail!(
                            "tree verifier accepted tap gather returned {} tap bytes, expected {}",
                            taps_bytes.len(),
                            expected_tap_bytes,
                        );
                    }
                    upload_taps_to_gpu_history(
                        &mut tap_history_gpu,
                        tap_history_len,
                        per_tap_row_bytes,
                        &taps_bytes,
                    )?;
                    tap_history.extend_from_slice(&taps_bytes);
                    tap_history_len += accepted_len;
                    next_bonus_seed = Some(carried_seed);
                    ms_rollback += t_rollback.elapsed().as_secs_f64() * 1000.0;
                }
                DFlashVerifyOutput::Fallback { snap, .. } => {
                    // Restore linear, rewind full-attn kv_filled, then re-decode
                    // the committed tokens to rewrite full-attn K/V and capture
                    // target-feature rows for the next draft round.
                    target_engine
                        .state_mut()
                        .restore_linear(&snap, ordinal)
                        .map_err(|e| anyhow!("restore linear: {e}"))?;
                    target_engine.rewind_full_kv_filled(l);

                    let t_redecode = Instant::now();
                    let (logits, taps_bytes) = target_engine.decode_block_with_taps_kernel(
                        &committed_block,
                        l,
                        &tap_layers,
                    )?;
                    let expected_tap_bytes = accepted_len * per_tap_row_bytes;
                    if taps_bytes.len() != expected_tap_bytes {
                        bail!(
                            "decode_block_with_taps_kernel returned {} tap bytes, expected {}",
                            taps_bytes.len(),
                            expected_tap_bytes,
                        );
                    }
                    upload_taps_to_gpu_history(
                        &mut tap_history_gpu,
                        tap_history_len,
                        per_tap_row_bytes,
                        &taps_bytes,
                    )?;
                    tap_history.extend_from_slice(&taps_bytes);
                    tap_history_len += accepted_len;
                    next_bonus_seed = Some(DecodeEngine::greedy_sample(&logits));
                    ms_redecode += t_redecode.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        // 8f. Advance counters + record generated.
        // Stop as soon as any committed token is EOS — every committed
        // token goes through the target's greedy pick, same semantics
        // as the non-DFlash decode loop which bails on the first EOS
        // it samples. Without this, generation would keep rolling past
        // an EOS that appeared inside a speculative commit block.
        committed_len = l + accepted_len;
        accepted_total += accept_n;
        let mut hit_eos = false;
        for &t in committed_block.iter() {
            generated_ids.push(t);
            if !cli.ignore_eos && eos_ids.contains(&t) {
                hit_eos = true;
                break;
            }
            if generated_ids.len() >= cli.max_new_tokens {
                break;
            }
        }
        if hit_eos {
            break;
        }
        if finish_after_commit {
            break;
        }
        bonus_seed = next_bonus_seed.ok_or_else(|| anyhow!("missing DFlash next bonus seed"))?;

        draft_state.reset();
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow!("detokenize: {e}"))?;
    println!("{text}");
    println!(
        "[tokens] {}",
        generated_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mean_accepted = if rounds_run == 0 {
        0.0
    } else {
        accepted_total as f64 / rounds_run as f64
    };
    eprintln!(
        "[dflash] rounds={rounds_run} mean_accepted_per_round={mean_accepted:.2} \
         generated={} decode_ms={decode_ms:.0}",
        generated_ids.len()
    );
    let ms_per_tok = if generated_ids.is_empty() {
        0.0
    } else {
        decode_ms / generated_ids.len() as f64
    };
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} \
         ms_per_tok={ms_per_tok:.2}",
        prompt_ids.len(),
        generated_ids.len()
    );
    let ms_other = (decode_ms - ms_draft - ms_verify - ms_redecode - ms_rollback).max(0.0);
    eprintln!(
        "[dflash] breakdown ms: draft={ms_draft:.0} verify={ms_verify:.0} \
         redecode={ms_redecode:.0} rollback={ms_rollback:.0} other={ms_other:.0}",
    );
    if profile_ffi {
        let ffi = kernel_ffi::prefill_ffi::ffi_profile_snapshot();
        let hal = gpu_hal::hal_profile_snapshot();
        eprintln!(
            "[dflash-ffi-profile] calls={} total_ms={:.3}",
            ffi.total_calls, ffi.total_ms
        );
        for entry in ffi.entries.iter().take(40) {
            eprintln!(
                "[dflash-ffi-profile] op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3}",
                entry.op,
                entry.calls,
                entry.mean_ms(),
                entry.total_ms,
                entry.max_ms,
            );
        }
        eprintln!(
            "[dflash-hal-profile] calls={} total_ms={:.3} alloc_calls={} alloc_bytes={} h2d={} d2h={} d2d={} memset={} sync_calls={}",
            hal.total_calls,
            hal.total_ms,
            hal.alloc_calls,
            hal.alloc_bytes,
            hal.h2d_bytes,
            hal.d2h_bytes,
            hal.d2d_bytes,
            hal.memset_bytes,
            hal.sync_calls,
        );
        for entry in hal.entries.iter().take(20) {
            let mean_ms = if entry.calls > 0 {
                entry.total_ms / entry.calls as f64
            } else {
                0.0
            };
            eprintln!(
                "[dflash-hal-profile] op={} calls={} mean_ms={:.4} total_ms={:.3} max_ms={:.3} bytes={}",
                entry.op,
                entry.calls,
                mean_ms,
                entry.total_ms,
                entry.max_ms,
                entry.total_bytes,
            );
        }
        kernel_ffi::prefill_ffi::ffi_profile_set_enabled(false);
        gpu_hal::hal_profile_set_enabled(false);
    }

    let _ = draft_rotary; // drop order guard (rotary/scratch hold GPU buffers)
    Ok(())
}

fn ddtree_probe_config_from_env() -> Result<Option<DFlashDDTreeProbeConfig>> {
    ddtree_config_from_env("SUPERSONIC_DFLASH_DDTREE_PROBE")
}

fn ddtree_verify_config_from_env() -> Result<Option<DFlashDDTreeProbeConfig>> {
    ddtree_config_from_env("SUPERSONIC_DFLASH_DDTREE_VERIFY")
}

fn ddtree_config_from_env(trigger: &str) -> Result<Option<DFlashDDTreeProbeConfig>> {
    if env::var_os(trigger).is_none() {
        return Ok(None);
    }
    let budget = parse_env_usize("SUPERSONIC_DFLASH_DDTREE_BUDGET", DDTREE_DEFAULT_BUDGET)?;
    let top_k = parse_env_usize("SUPERSONIC_DFLASH_DDTREE_TOP_K", DDTREE_DEFAULT_TOP_K)?;
    let temperature = parse_env_f32("SUPERSONIC_DFLASH_DDTREE_TEMP", 1.0)?;
    if budget == 0 {
        bail!("SUPERSONIC_DFLASH_DDTREE_BUDGET must be > 0");
    }
    if top_k == 0 {
        bail!("SUPERSONIC_DFLASH_DDTREE_TOP_K must be > 0");
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        bail!("SUPERSONIC_DFLASH_DDTREE_TEMP must be finite and > 0");
    }

    Ok(Some(DFlashDDTreeProbeConfig {
        budget,
        top_k,
        temperature,
        chain_seed: env::var_os("SUPERSONIC_DFLASH_DDTREE_NO_CHAIN_SEED").is_none(),
    }))
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<usize>()
            .map_err(|e| anyhow!("{name} must be an unsigned integer: {e}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8"),
    }
}

fn parse_env_f32(name: &str, default: f32) -> Result<f32> {
    match env::var(name) {
        Ok(raw) => raw
            .parse::<f32>()
            .map_err(|e| anyhow!("{name} must be a float: {e}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8"),
    }
}

/// Parse `--dflash-tap-layers "1,8,15"` into a validated Vec<usize>.
fn parse_tap_override(raw: &str, num_target_layers: usize) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let t: usize = part
            .trim()
            .parse()
            .map_err(|e| anyhow!("--dflash-tap-layers: bad integer '{part}': {e}"))?;
        if t >= num_target_layers {
            bail!("tap layer {t} out of range (num_target_layers={num_target_layers})");
        }
        out.push(t);
    }
    if out.is_empty() {
        bail!("--dflash-tap-layers must list at least one integer");
    }
    Ok(out)
}

fn load_target_lowbit_weights(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    ordinal: usize,
    weight_prefix: &str,
) -> Result<Qwen35Weights> {
    load_qwen35_weights(
        cli,
        model_variant,
        text_config,
        ordinal,
        weight_prefix,
        true, // DFlash dispatcher already called ensure_hf_metadata_present.
        crate::policy::q4km_like(cli),
    )
    .map_err(|e| anyhow!("load target low-bit weights: {e}"))
}

enum DFlashVerifyOutput {
    Captured(PrefillAppendVerifyResult),
    CapturedWindowed(CapturedWindowedVerifyOutput),
    Tree(PrefillTreeVerifyOutput),
    Fallback {
        logits: Vec<Vec<f32>>,
        snap: LinearStateSnapshot,
    },
}

struct CapturedWindowedVerifyOutput {
    segments: Vec<CapturedWindowSegment>,
    target_next: Vec<u32>,
}

struct CapturedWindowSegment {
    start: usize,
    len: usize,
    result: PrefillAppendVerifyResult,
}

struct PrefillTreeVerifyOutput {
    root_token: u32,
    tree: DDTree,
    result: PrefillTreeVerifyResult,
}

impl DFlashVerifyOutput {
    fn greedy_ids(&self) -> Result<Vec<u32>> {
        match self {
            Self::Captured(result) => result.target_next.clone().ok_or_else(|| {
                anyhow!("captured prefill verifier did not return greedy target IDs")
            }),
            Self::CapturedWindowed(result) => Ok(result.target_next.clone()),
            Self::Tree(result) => Ok(result.result.target_next.clone()),
            Self::Fallback { logits, .. } => Ok(logits
                .iter()
                .map(|row| DecodeEngine::greedy_sample(row))
                .collect()),
        }
    }
}

fn dflash_prefill_window_scan_chunk() -> usize {
    env::var("SUPERSONIC_DFLASH_VERIFY_SCAN_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16)
}

fn dflash_prefill_window_min_tail() -> usize {
    env::var("SUPERSONIC_DFLASH_VERIFY_MIN_TAIL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

fn dflash_final_verify_min_len(block_size: usize) -> Option<usize> {
    if env::var_os("SUPERSONIC_DFLASH_DISABLE_FINAL_VERIFY_PAD").is_some() {
        return None;
    }
    Some(
        env::var("SUPERSONIC_DFLASH_FINAL_VERIFY_MIN_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(block_size),
    )
}

fn dflash_verify_len_for_round(remaining_budget: usize, block_size: usize) -> usize {
    let clamped = block_size.min(remaining_budget);
    if remaining_budget >= block_size {
        return clamped;
    }
    let Some(min_len) = dflash_final_verify_min_len(block_size) else {
        return clamped;
    };
    clamped.max(min_len.min(block_size))
}

fn dflash_gpu_tap_history_enabled() -> bool {
    env::var_os("SUPERSONIC_DFLASH_DISABLE_GPU_TAP_HISTORY").is_none()
}

fn dflash_window_step(remaining: usize, scan_chunk: usize, min_tail: usize) -> usize {
    let step = scan_chunk.min(remaining);
    let tail = remaining - step;
    if tail > 0 && tail < min_tail {
        remaining
    } else {
        step
    }
}

fn chain_accept_needs_more(target_next: &[u32], tokens: &[u32], verify_len: usize) -> bool {
    let mut accept_n = 1usize;
    while accept_n < verify_len {
        if accept_n > target_next.len() {
            return true;
        }
        if target_next[accept_n - 1] == tokens[accept_n] {
            accept_n += 1;
        } else {
            return false;
        }
    }
    accept_n > target_next.len()
}

fn verify_block_prefill_append_windowed(
    target_engine: &mut DecodeEngine,
    tokens: &[u32],
    pos_offset: usize,
    tap_layers: &[usize],
    scan_chunk: usize,
    mut gpu_tap_history: Option<(&mut GpuBuffer, usize, usize)>,
) -> Result<CapturedWindowedVerifyOutput> {
    if scan_chunk == 0 {
        bail!("prefill append window scan chunk must be > 0");
    }

    let mut segments = Vec::new();
    let mut target_next = Vec::new();
    let min_tail = dflash_prefill_window_min_tail();
    let mut start = 0usize;
    while start < tokens.len() {
        let step = dflash_window_step(tokens.len() - start, scan_chunk, min_tail);
        let result = if let Some((history, base_row, row_bytes)) = gpu_tap_history.as_mut() {
            target_engine.verify_block_prefill_append_captured_lazy_acceptance_gpu_taps(
                &tokens[start..start + step],
                pos_offset + start,
                tap_layers,
                &mut **history,
                *base_row + start,
                *row_bytes,
            )?
        } else {
            target_engine.verify_block_prefill_append_captured_lazy_acceptance(
                &tokens[start..start + step],
                pos_offset + start,
                tap_layers,
            )?
        };
        let ids = result
            .target_next
            .clone()
            .ok_or_else(|| anyhow!("windowed prefill verifier did not return greedy target IDs"))?;
        if ids.is_empty() {
            bail!("windowed prefill verifier returned no greedy target IDs");
        }
        target_next.extend(ids);
        segments.push(CapturedWindowSegment {
            start,
            len: step,
            result,
        });

        if !chain_accept_needs_more(&target_next, tokens, tokens.len()) {
            break;
        }
        start += step;
    }

    Ok(CapturedWindowedVerifyOutput {
        segments,
        target_next,
    })
}

fn verify_block_for_dflash(
    target_engine: &mut DecodeEngine,
    tokens: &[u32],
    pos_offset: usize,
    tap_layers: &[usize],
    mut gpu_tap_history: Option<(&mut GpuBuffer, usize, usize)>,
) -> Result<DFlashVerifyOutput> {
    let force_prefill = env::var_os("SUPERSONIC_DFLASH_PREFILL_VERIFY").is_some();
    let disable_prefill = env::var_os("SUPERSONIC_DFLASH_DISABLE_PREFILL_VERIFY").is_some();
    let chunk_size = dflash_fused_verify_chunk_size(target_engine);
    let config = &target_engine.weights().config;
    let prefer_prefill_append = config.hidden_size == 5120 && config.num_hidden_layers == 64;
    let use_prefill_append = force_prefill
        || (!disable_prefill
            && (prefer_prefill_append || (chunk_size > 0 && tokens.len() > chunk_size)));

    if use_prefill_append {
        static PREFILL_NOTICE: Once = Once::new();
        PREFILL_NOTICE.call_once(|| {
            eprintln!("[dflash] using prefill-append target verifier");
        });
        let scan_chunk = dflash_prefill_window_scan_chunk();
        if scan_chunk < tokens.len()
            && env::var_os("SUPERSONIC_DFLASH_DISABLE_VERIFY_ROW_SCAN").is_none()
        {
            match verify_block_prefill_append_windowed(
                target_engine,
                tokens,
                pos_offset,
                tap_layers,
                scan_chunk,
                gpu_tap_history
                    .as_mut()
                    .map(|(history, start_row, row_bytes)| {
                        (&mut **history, *start_row, *row_bytes)
                    }),
            ) {
                Ok(result) => return Ok(DFlashVerifyOutput::CapturedWindowed(result)),
                Err(err) if force_prefill => {
                    return Err(anyhow!("windowed prefill append verify failed: {err}"));
                }
                Err(err) => {
                    return Err(anyhow!("windowed prefill append verify failed: {err}"));
                }
            }
        }
        let captured_result =
            if let Some((history, start_row, row_bytes)) = gpu_tap_history.as_mut() {
                target_engine.verify_block_prefill_append_captured_lazy_acceptance_gpu_taps(
                    tokens,
                    pos_offset,
                    tap_layers,
                    &mut **history,
                    *start_row,
                    *row_bytes,
                )
            } else {
                target_engine.verify_block_prefill_append_captured_lazy_acceptance(
                    tokens, pos_offset, tap_layers,
                )
            };
        match captured_result {
            Ok(result) => return Ok(DFlashVerifyOutput::Captured(result)),
            Err(err) if force_prefill => {
                return Err(anyhow!("prefill append verify failed: {err}"));
            }
            Err(err) => {
                static PREFILL_FALLBACK_NOTICE: Once = Once::new();
                PREFILL_FALLBACK_NOTICE.call_once(|| {
                    eprintln!(
                        "[dflash] prefill-append verifier failed ({err}); \
                         falling back to persistent verifier"
                    );
                });
            }
        }
    }

    let snap: LinearStateSnapshot = target_engine
        .state_mut()
        .snapshot_linear()
        .map_err(|e| anyhow!("snapshot linear: {e}"))?;

    let needs_sequential = tokens.len() > kernel_ffi::MAX_BATCH_SIZE;
    if !needs_sequential {
        match target_engine.verify_block_fused_decode(tokens, pos_offset) {
            Ok(logits) => return Ok(DFlashVerifyOutput::Fallback { logits, snap }),
            Err(err) => {
                let msg = err.to_string();
                if !msg.contains("shared-memory budget exceeded") {
                    return Err(err);
                }
            }
        }
    }

    if chunk_size > 0 && chunk_size < tokens.len() {
        let mut out = Vec::with_capacity(tokens.len());
        let mut start = 0usize;
        while start < tokens.len() {
            let remaining = tokens.len() - start;
            let step = if remaining > chunk_size && remaining - chunk_size == 1 && chunk_size > 1 {
                chunk_size - 1
            } else {
                remaining.min(chunk_size)
            };
            let end = start + step;
            match target_engine.verify_block_fused_decode(&tokens[start..end], pos_offset + start) {
                Ok(mut logits) => {
                    out.append(&mut logits);
                    start = end;
                }
                Err(err) => {
                    let msg = err.to_string();
                    if start == 0 && msg.contains("shared-memory budget exceeded") {
                        break;
                    }
                    return Err(anyhow!(
                        "chunked fused verify failed at token {start}: {err}"
                    ));
                }
            }
        }
        if out.len() == tokens.len() {
            static CHUNK_NOTICE: Once = Once::new();
            CHUNK_NOTICE.call_once(|| {
                eprintln!(
                    "[dflash] fused verify split block={} into chunks of {} for LDS budget",
                    tokens.len(),
                    chunk_size
                );
            });
            return Ok(DFlashVerifyOutput::Fallback { logits: out, snap });
        }
    }

    static NOTICE: Once = Once::new();
    NOTICE.call_once(|| {
        eprintln!(
            "[dflash] fused verify does not fit this target shape/block; \
             using sequential target verify fallback"
        );
    });

    let mut logits = Vec::with_capacity(tokens.len());
    for (i, &tok) in tokens.iter().enumerate() {
        let (step_logits, _tap_bytes) = target_engine
            .decode_step_with_taps_kernel(tok, pos_offset + i, tap_layers)
            .map_err(|e| anyhow!("sequential verify decode step {i}: {e}"))?;
        logits.push(step_logits);
    }
    Ok(DFlashVerifyOutput::Fallback { logits, snap })
}

fn verify_ddtree_for_dflash(
    target_engine: &mut DecodeEngine,
    root_token: u32,
    tree: &DDTree,
    pos_offset: usize,
    tap_layers: &[usize],
    capture_rollback: bool,
) -> Result<DFlashVerifyOutput> {
    static NOTICE: Once = Once::new();
    NOTICE.call_once(|| {
        eprintln!("[dflash] using DDTree target verifier");
    });

    let plan = build_verify_plan(tree, root_token, pos_offset);
    let result = target_engine.verify_tree_prefill_captured(
        &plan.flat_tokens,
        &plan.positions,
        &plan.parent_ids,
        &plan.visibility,
        pos_offset,
        tap_layers,
        capture_rollback,
    )?;
    if result.target_next.len() != plan.flat_tokens.len() {
        bail!(
            "DDTree verifier returned {} posterior IDs, expected {}",
            result.target_next.len(),
            plan.flat_tokens.len()
        );
    }
    Ok(DFlashVerifyOutput::Tree(PrefillTreeVerifyOutput {
        root_token,
        tree: tree.clone(),
        result,
    }))
}

fn dflash_fused_verify_chunk_size(target_engine: &DecodeEngine) -> usize {
    const MAX_INPUT_CACHE_FLOATS: usize = 15872;
    let hidden_dim = target_engine.weights().config.hidden_size;
    if hidden_dim == 0 || 2 * hidden_dim > MAX_INPUT_CACHE_FLOATS {
        return 0;
    }
    (MAX_INPUT_CACHE_FLOATS / hidden_dim).min(kernel_ffi::MAX_BATCH_SIZE)
}

fn upload_taps_to_gpu_history(
    tap_history_gpu: &mut GpuBuffer,
    start_row: usize,
    row_bytes: usize,
    taps: &[u8],
) -> Result<()> {
    if taps.is_empty() {
        return Ok(());
    }
    if row_bytes == 0 || taps.len() % row_bytes != 0 {
        bail!(
            "tap upload byte length {} is not a multiple of row_bytes {}",
            taps.len(),
            row_bytes
        );
    }
    let dst_offset = start_row * row_bytes;
    if dst_offset + taps.len() > tap_history_gpu.len_bytes() {
        bail!(
            "GPU tap history write exceeds buffer: offset {} + len {} > {}",
            dst_offset,
            taps.len(),
            tap_history_gpu.len_bytes()
        );
    }
    let dst = unsafe {
        (tap_history_gpu.as_mut_ptr() as *mut u8).add(dst_offset) as *mut std::ffi::c_void
    };
    gpu_hal::copy_h2d(
        tap_history_gpu.device_ordinal(),
        dst,
        taps.as_ptr() as *const std::ffi::c_void,
        taps.len(),
    )
    .map_err(|e| anyhow!("upload tap history: {e}"))
}

/// Convert tap-major BF16 history into draft input layout.
///
/// Input is one `[num_positions, hidden_dim]` byte vector per tap layer.
/// Output is `[num_positions, num_taps * hidden_dim]`, row-major by target
/// position, matching Lucebox's `target_hidden_cat`.
fn flatten_tap_history(
    per_tap: &[Vec<u8>],
    num_positions: usize,
    hidden_dim: usize,
) -> Result<Vec<u8>> {
    if per_tap.is_empty() {
        bail!("flatten_tap_history requires at least one tap layer");
    }
    let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
    let expected = num_positions * row_bytes;
    for (idx, tap) in per_tap.iter().enumerate() {
        if tap.len() != expected {
            bail!(
                "tap history {idx} has {} bytes, expected {expected} \
                 ({num_positions} positions * {hidden_dim} hidden * 2)",
                tap.len(),
            );
        }
    }
    let mut out = Vec::with_capacity(num_positions * per_tap.len() * row_bytes);
    for pos in 0..num_positions {
        let start = pos * row_bytes;
        for tap in per_tap {
            out.extend_from_slice(&tap[start..start + row_bytes]);
        }
    }
    Ok(out)
}

fn flatten_tap_history_indices(
    per_tap: &[Vec<u8>],
    num_positions: usize,
    hidden_dim: usize,
    indices: &[usize],
) -> Result<Vec<u8>> {
    if per_tap.is_empty() {
        bail!("flatten_tap_history_indices requires at least one tap layer");
    }
    let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
    let expected = num_positions * row_bytes;
    for (idx, tap) in per_tap.iter().enumerate() {
        if tap.len() != expected {
            bail!(
                "indexed tap history {idx} has {} bytes, expected {expected} \
                 ({num_positions} positions * hidden_dim {hidden_dim} * bf16)",
                tap.len(),
            );
        }
    }
    let mut out = Vec::with_capacity(indices.len() * per_tap.len() * row_bytes);
    for &row in indices {
        if row >= num_positions {
            bail!("tap gather row {row} out of range num_positions={num_positions}");
        }
        let start = row * row_bytes;
        for tap in per_tap {
            out.extend_from_slice(&tap[start..start + row_bytes]);
        }
    }
    Ok(out)
}

/// Drive the DFlash draft's forward pass for one round and sample B
/// candidate tokens via the target's `lm_head`.
///
/// The draft path mirrors Lucebox's stateless draft graph: every round sees a
/// target-feature history window plus the current `[bonus_seed, MASK, ...]`
/// noise block, with positions local to that window.
fn draft_forward_and_sample(
    draft_state: &mut dflash::state::DFlashState,
    draft_scratch: &mut dflash::state::DFlashScratch,
    noise_embedding: &mut GpuBuffer,
    draft_rotary: &dflash::RotaryTables,
    draft_weights: &dflash::DFlashWeights,
    target_engine: &DecodeEngine,
    tap_history_gpu: &GpuBuffer,
    tap_start_row: usize,
    round_taps_len: usize,
    bonus_seed: u32,
    block_size: usize,
    mask_token_id: u32,
    ddtree_probe_config: Option<&DFlashDDTreeProbeConfig>,
    ddtree_depth_limit: usize,
    ordinal: usize,
) -> Result<DraftRoundOutput> {
    if round_taps_len == 0 {
        bail!("draft_forward: round_taps_len must be > 0");
    }
    let profile = env::var_os("SUPERSONIC_DFLASH_PROFILE_DRAFT").is_some();
    let mut ms_noise = 0.0_f64;
    let mut ms_tap_copy = 0.0_f64;
    let mut ms_forward = 0.0_f64;
    let mut ms_lm_head = 0.0_f64;
    let mut ms_argmax = 0.0_f64;
    let mut ms_tree_probe = 0.0_f64;
    let hidden = draft_weights.config.hidden_size;
    let num_taps = draft_weights.config.num_taps();
    let tap_row_bytes = num_taps * hidden * ScalarType::BF16.size_in_bytes();
    let expected_bytes = round_taps_len * tap_row_bytes;
    let src_offset = tap_start_row * tap_row_bytes;
    if src_offset + expected_bytes > tap_history_gpu.len_bytes() {
        bail!(
            "GPU tap history read exceeds buffer: offset {} + len {} > {}",
            src_offset,
            expected_bytes,
            tap_history_gpu.len_bytes()
        );
    }

    // 1) Build noise_embedding = embed([bonus_seed, MASK, …, MASK]).
    let t_noise = Instant::now();
    let target_embed = &target_engine.weights().embed_tokens;
    let row_bytes = hidden * ScalarType::BF16.size_in_bytes();
    let expected_noise_bytes = block_size * row_bytes;
    if noise_embedding.len_bytes() < expected_noise_bytes {
        bail!(
            "draft noise embedding scratch has {} bytes, need {}",
            noise_embedding.len_bytes(),
            expected_noise_bytes
        );
    }
    for i in 0..block_size {
        let tok = if i == 0 { bonus_seed } else { mask_token_id };
        let src_off = tok as usize * row_bytes;
        let dst_off = i * row_bytes;
        gpu_hal::copy_d2d(
            ordinal,
            unsafe {
                (noise_embedding.as_mut_ptr() as *mut u8).add(dst_off) as *mut std::ffi::c_void
            },
            target_embed.offset_ptr(src_off),
            row_bytes,
        )
        .map_err(|e| anyhow!("noise_embedding gather slot {i}: {e}"))?;
    }
    if profile {
        ms_noise += t_noise.elapsed().as_secs_f64() * 1000.0;
    }

    // 2) Copy target_hidden_raw [1, round_taps_len, num_taps*hidden] into
    // reusable draft scratch. `forward()` reads only the leading ctx rows.
    let t_tap_copy = Instant::now();
    if expected_bytes > draft_scratch.fuser_input.len_bytes() {
        bail!(
            "draft fuser_input scratch has {} bytes, need {}",
            draft_scratch.fuser_input.len_bytes(),
            expected_bytes
        );
    }
    gpu_hal::copy_d2d(
        ordinal,
        draft_scratch.fuser_input.as_mut_ptr(),
        tap_history_gpu.offset_ptr(src_offset),
        expected_bytes,
    )
    .map_err(|e| anyhow!("copy target_hidden_raw: {e}"))?;
    if profile {
        ms_tap_copy += t_tap_copy.elapsed().as_secs_f64() * 1000.0;
    }

    // 3) Draft forward.
    let t_forward = Instant::now();
    let pos_offset = 0;
    dflash::forward::forward(
        draft_weights,
        draft_state,
        draft_scratch,
        draft_rotary,
        noise_embedding,
        dflash::ForwardParams {
            ctx_len: round_taps_len,
            q_len: block_size,
            pos_offset,
        },
    )
    .map_err(|e| anyhow!("draft forward: {e}"))?;
    if profile {
        ms_forward += t_forward.elapsed().as_secs_f64() * 1000.0;
    }

    // 4) lm_head projection -> persistent draft logits scratch [block_size, vocab].
    let t_lm_head = Instant::now();
    let target_weights = target_engine.weights();
    let lm_head = &target_weights.lm_head;
    let lm_head_buf: &GpuBuffer = lm_head.as_ref();
    let vocab = draft_weights.config.vocab_size;
    if draft_scratch.logits.elem_count() < block_size * vocab {
        bail!(
            "draft logits scratch has {} elems, need {}",
            draft_scratch.logits.elem_count(),
            block_size * vocab
        );
    }
    if let Some((lm_head_qtype, scale, zero)) = target_weights.lm_head_lowbit_params(hidden) {
        kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,          // batch
            block_size, // m
            vocab,      // n
            hidden,     // k
            &draft_scratch.final_hidden,
            lm_head_buf,
            scale,
            zero,
            target_weights.lm_head_awq_inv_scale.as_ref(),
            target_weights.int4_group_size,
            lm_head_qtype,
            &mut draft_scratch.logits,
        )
        .map_err(|e| anyhow!("draft lm_head low-bit matmul: {e}"))?;
    } else {
        kernel_ffi::matmul_rhs_transposed_4b(
            ordinal,
            ScalarType::BF16,
            1,          // batch
            block_size, // m
            vocab,      // n
            hidden,     // k
            &draft_scratch.final_hidden,
            lm_head_buf,
            &mut draft_scratch.logits,
        )
        .map_err(|e| anyhow!("draft lm_head: {e}"))?;
    }
    if profile {
        ms_lm_head += t_lm_head.elapsed().as_secs_f64() * 1000.0;
    }

    // 5) GPU argmax per position, then D2H only the token IDs.
    let t_argmax = Instant::now();
    kernel_ffi::prefill_ffi::argmax_bf16_rows(
        ordinal,
        block_size,
        vocab,
        &draft_scratch.logits,
        &mut draft_scratch.argmax_indices,
    )
    .map_err(|e| anyhow!("draft logits argmax: {e}"))?;
    let argmax_bytes = draft_scratch
        .argmax_indices
        .to_host_bytes()
        .map_err(|e| anyhow!("draft argmax indices D2H: {e}"))?;
    let mut candidates: Vec<u32> = argmax_bytes
        .chunks_exact(4)
        .take(block_size)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if candidates.len() != block_size {
        bail!(
            "draft argmax returned {} candidates, expected {block_size}",
            candidates.len()
        );
    }
    if let Some(first) = candidates.first_mut() {
        // Lucebox DFlash treats slot 0 as the previous target token, not a
        // free draft proposal: noise_ids[0] = bonus_seed and then
        // draft_tok[0] is forced back to that same token after projection.
        *first = bonus_seed;
    }
    if profile {
        ms_argmax += t_argmax.elapsed().as_secs_f64() * 1000.0;
    }

    let t_tree_probe = Instant::now();
    let ddtree_probe = match ddtree_probe_config {
        Some(config) => Some(probe_ddtree_round(
            &candidates,
            &draft_scratch.logits,
            block_size,
            ddtree_depth_limit,
            vocab,
            config,
        )?),
        None => None,
    };
    if profile {
        ms_tree_probe += t_tree_probe.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[dflash-profile] draft outer ctx={} q={} noise={:.2}ms tap_copy={:.2}ms forward={:.2}ms lm_head={:.2}ms argmax={:.2}ms tree_probe={:.2}ms",
            round_taps_len,
            block_size,
            ms_noise,
            ms_tap_copy,
            ms_forward,
            ms_lm_head,
            ms_argmax,
            ms_tree_probe,
        );
    }

    Ok(DraftRoundOutput {
        candidates,
        ddtree_probe,
    })
}

fn probe_ddtree_round(
    candidates: &[u32],
    draft_logits: &GpuBuffer,
    block_size: usize,
    depth_limit: usize,
    vocab: usize,
    config: &DFlashDDTreeProbeConfig,
) -> Result<DFlashDDTreeProbeRound> {
    let depth_limit = depth_limit.min(block_size.saturating_sub(1));
    let top_k = if config.budget > depth_limit {
        config.top_k.min(vocab)
    } else {
        1
    };
    let (top_log_probs, top_token_ids) = if depth_limit == 0 {
        (Vec::new(), Vec::new())
    } else if top_k == 1 {
        let ids = candidates
            .iter()
            .skip(1)
            .take(depth_limit)
            .copied()
            .collect::<Vec<_>>();
        (vec![0.0; depth_limit], ids)
    } else {
        let row_bytes = vocab * ScalarType::BF16.size_in_bytes();
        let needed = block_size * row_bytes;
        let logits_bytes = draft_logits
            .to_host_bytes()
            .map_err(|e| anyhow!("dflash ddtree probe logits D2H: {e}"))?;
        if logits_bytes.len() < needed {
            bail!(
                "dflash ddtree probe logits D2H returned {} bytes, expected at least {}",
                logits_bytes.len(),
                needed
            );
        }
        extract_draft_topk_bf16(
            &logits_bytes[row_bytes..row_bytes + depth_limit * row_bytes],
            depth_limit,
            vocab,
            top_k,
            config.temperature,
        )
    };

    let tree = build_ddtree(
        &top_log_probs,
        &top_token_ids,
        depth_limit,
        top_k,
        config.budget,
        config.chain_seed,
    );
    let top1_head = top_token_ids
        .chunks(top_k)
        .take(6)
        .filter_map(|row| row.first().copied())
        .collect::<Vec<_>>();
    let max_depth = tree.depths.iter().copied().max().unwrap_or(0);

    Ok(DFlashDDTreeProbeRound {
        depth_limit,
        top_k,
        budget: config.budget,
        nodes: tree.n_nodes(),
        width: tree.width(),
        max_depth,
        top1_head,
        tree,
    })
}

#[cfg(test)]
mod tests {
    use super::{chain_accept_needs_more, dflash_verify_len_for_round, dflash_window_step};
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn verify_len_pads_final_partial_rounds_by_default() {
        let _guard = env_lock().lock().unwrap();
        std::env::remove_var("SUPERSONIC_DFLASH_DISABLE_FINAL_VERIFY_PAD");
        std::env::remove_var("SUPERSONIC_DFLASH_FINAL_VERIFY_MIN_LEN");
        assert_eq!(dflash_verify_len_for_round(3, 16), 16);
        assert_eq!(dflash_verify_len_for_round(8, 16), 16);
        assert_eq!(dflash_verify_len_for_round(20, 16), 16);
    }

    #[test]
    fn verify_len_can_override_final_partial_minimum() {
        let _guard = env_lock().lock().unwrap();
        std::env::remove_var("SUPERSONIC_DFLASH_DISABLE_FINAL_VERIFY_PAD");
        std::env::set_var("SUPERSONIC_DFLASH_FINAL_VERIFY_MIN_LEN", "8");
        assert_eq!(dflash_verify_len_for_round(3, 16), 8);
        assert_eq!(dflash_verify_len_for_round(8, 16), 8);
        assert_eq!(dflash_verify_len_for_round(12, 16), 12);
        assert_eq!(dflash_verify_len_for_round(20, 16), 16);
        std::env::remove_var("SUPERSONIC_DFLASH_FINAL_VERIFY_MIN_LEN");
    }

    #[test]
    fn verify_len_can_restore_budget_clamp() {
        let _guard = env_lock().lock().unwrap();
        std::env::set_var("SUPERSONIC_DFLASH_DISABLE_FINAL_VERIFY_PAD", "1");
        std::env::remove_var("SUPERSONIC_DFLASH_FINAL_VERIFY_MIN_LEN");
        assert_eq!(dflash_verify_len_for_round(3, 16), 3);
        assert_eq!(dflash_verify_len_for_round(8, 16), 8);
        assert_eq!(dflash_verify_len_for_round(20, 16), 16);
        std::env::remove_var("SUPERSONIC_DFLASH_DISABLE_FINAL_VERIFY_PAD");
    }

    #[test]
    fn window_step_absorbs_tiny_tail() {
        assert_eq!(dflash_window_step(10, 8, 4), 10);
        assert_eq!(dflash_window_step(11, 8, 4), 11);
    }

    #[test]
    fn window_step_keeps_normal_tail() {
        assert_eq!(dflash_window_step(12, 8, 4), 8);
        assert_eq!(dflash_window_step(16, 8, 4), 8);
    }

    #[test]
    fn window_step_handles_short_remaining() {
        assert_eq!(dflash_window_step(1, 8, 4), 1);
        assert_eq!(dflash_window_step(7, 8, 4), 7);
    }

    #[test]
    fn chain_accept_fetches_carried_seed_after_full_window_match() {
        let tokens = [10, 20, 30, 40];
        let target_next = [20, 30, 40];

        assert!(chain_accept_needs_more(&target_next, &tokens, tokens.len()));
    }

    #[test]
    fn chain_accept_stops_when_mismatch_is_known() {
        let tokens = [10, 20, 30, 40];
        let target_next = [20, 31, 99];

        assert!(!chain_accept_needs_more(
            &target_next,
            &tokens,
            tokens.len()
        ));
    }

    #[test]
    fn chain_accept_has_enough_for_single_token_verify() {
        let tokens = [10];
        let target_next = [20];

        assert!(!chain_accept_needs_more(
            &target_next,
            &tokens,
            tokens.len()
        ));
    }
}
