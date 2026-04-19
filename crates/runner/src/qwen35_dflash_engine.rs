//! Qwen3.5-9B DFlash speculative-decoding engine (M3.3).
//!
//! Drives the target (Qwen3.5-9B INT4) and the DFlash draft together through
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

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::state::LinearStateSnapshot;
use qwen35::weights::Qwen35Weights;
use qwen35_dflash as dflash;

use crate::decode_engine::DecodeEngine;
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};
use crate::Cli;

/// Run the Qwen3.5-9B DFlash speculative decoder. Parallels
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
    if !cli.int4 {
        bail!("--dflash requires --int4 (Qwen3.5-9B INT4 target)");
    }
    if model_variant.to_string() != "qwen3.5-9b" {
        bail!(
            "--dflash is only supported for --model qwen3.5-9b (got {model_variant})"
        );
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
        FamilyParams::Gemma4(_) | FamilyParams::Phi4(_) => {
            unreachable!("run_qwen35_dflash dispatched for non-qwen35 variant");
        }
    };
    if !params.use_4b_kernel {
        bail!("--dflash requires the 4B kernel path (qwen3.5-9b INT4)");
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
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        bail!("empty prompt after tokenization");
    }

    // --------- 3. VRAM estimate (target INT4 + ~2 GiB draft) -------------
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
    let target_fixed = (entry.vram.fixed_bytes as f64 * 0.37) as u64; // INT4 scaling
    let target_kv = kv_per_token * context_tokens as u64;
    let draft_fixed: u64 = 2 * 1024 * 1024 * 1024; // ~2 GiB for DFlash draft weights + scratch
    let estimated =
        ((target_fixed + target_kv + draft_fixed) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (target weights={:.2}GiB + target KV={:.2}GiB + draft={:.2}GiB) \
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

    // --------- 4. Load target weights (INT4 bake) ------------------------
    let t0 = Instant::now();
    let target_weights =
        load_target_int4_weights(cli, entry, &text_config, ordinal, weight_prefix)?;
    eprintln!(
        "[weights] target (INT4, group_size={}) loaded in {:.0}ms",
        target_weights.int4_group_size,
        t0.elapsed().as_millis(),
    );

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
    let draft_config = dflash::load_config(draft_dir)
        .map_err(|e| anyhow!("load draft config.json: {e}"))?;
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

    let draft_weights = dflash::DFlashWeights::load(
        draft_dir,
        &draft_config,
        ordinal,
        Arc::clone(&target_embed),
        Arc::clone(&target_lm_head),
    )
    .map_err(|e| anyhow!("load draft weights: {e}"))?;
    eprintln!("[dflash] draft weights loaded");

    let draft_max_ctx = cli
        .context_size
        .map(|c| c.max(draft_config.block_size * 4))
        .unwrap_or_else(|| (context_tokens + draft_config.block_size).max(1024));
    let draft_rotary = dflash::RotaryTables::build(&draft_config, ordinal, draft_max_ctx)
        .map_err(|e| anyhow!("build draft RoPE: {e}"))?;
    let mut draft_scratch = dflash::state::DFlashScratch::new(ordinal, &draft_config)
        .map_err(|e| anyhow!("alloc draft scratch: {e}"))?;
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
    let mut round_taps: Vec<u8> = flatten_tap_blobs(&prefill_result.tap_hiddens.unwrap_or_default());
    let mut round_taps_len: usize = 1; // T=1 after prefill (last prompt token).

    // Sample the first bonus_seed from prefill's last logits (greedy @ T=0).
    let mut bonus_seed: u32 = DecodeEngine::greedy_sample(&prefill_result.logits);

    // kv_filled count on any full-attention layer is equal to prompt_len
    // after prefill. Track committed length separately so we don't depend
    // on state internals.
    let mut committed_len: usize = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let eos_ids: Vec<u32> = text_config.eos_token_ids();
    // Default block_size = 3: the fused verify path (now the only verify
    // path after M4.3c) caps B at 3 on Qwen3.5-9B because the 4B
    // megakernel's 64 KiB LDS budget has to hold (block_size + B*hidden
    // + fp8_lut) * 4 bytes. 4*4096 = 16384 floats exceeds the 15872
    // float budget — verify_block_fused_decode errors out with a
    // diagnostic if the user forces --dflash-block >= 4 on 9B.
    // Historical context: the M4.1 prefill verify path had no such cap
    // and used DEFAULT_BLOCK_SIZE=4 per project_m4_2_findings; see git
    // history before M4.3c for the B=4 vs B=16 sweep.
    const DEFAULT_BLOCK_SIZE: usize = 3;
    let block_size = cli
        .dflash_block
        .unwrap_or(DEFAULT_BLOCK_SIZE.min(draft_config.block_size));
    if block_size == 0 || block_size > draft_config.block_size {
        bail!(
            "--dflash-block must be in 1..={} (got {block_size})",
            draft_config.block_size,
        );
    }

    // --------- 8. Speculative loop ---------------------------------------
    let decode_start = Instant::now();
    let mut rounds_run: usize = 0;
    let mut accepted_total: usize = 0;
    // Per-stage timing accumulators. Reported alongside rounds summary so
    // anyone profiling DFlash can see which stage dominates wall-clock
    // before deciding what to optimize next.
    let mut ms_draft: f64 = 0.0;
    let mut ms_verify: f64 = 0.0;
    let mut ms_redecode: f64 = 0.0;
    while generated_ids.len() < cli.max_new_tokens {
        if eos_ids.contains(&bonus_seed) {
            generated_ids.push(bonus_seed);
            break;
        }

        rounds_run += 1;
        let l = committed_len;

        // 8a. Draft forward: noise_embedding = embed([bonus_seed, MASK, …])
        // of length B; target_hidden_raw carries ctx_len=round_taps_len tap
        // rows (1 after prefill, accepted_len after every round). Output
        // hidden is projected through target's lm_head → B block logits →
        // argmax → B candidates at positions L..L+B-1.
        let t_draft = Instant::now();
        let (draft_candidates, draft_final_hidden_bytes) = draft_forward_and_sample(
            &mut draft_state,
            &mut draft_scratch,
            &draft_rotary,
            &draft_weights,
            &target_engine,
            &round_taps,
            round_taps_len,
            bonus_seed,
            block_size,
            draft_config.dflash_config.mask_token_id,
            ordinal,
        )?;
        let _ = draft_final_hidden_bytes; // retained for future logits caching
        ms_draft += t_draft.elapsed().as_secs_f64() * 1000.0;

        // 8b. Snapshot linear state (before verify mutates it).
        let snap: LinearStateSnapshot = target_engine
            .state_mut()
            .snapshot_linear()
            .map_err(|e| anyhow!("snapshot linear: {e}"))?;

        // 8c. Verify: one `persistent_decode_4b` megakernel launch at
        //     positions `[l, l+B)`. Shared-cache BatchSeqDesc aliases the
        //     live sequence's KV/linear buffers across all B batch slots
        //     with `seqlen_offset[b] = l + b`; the kernel runs the B
        //     iterations sequentially on block 0 within a single layer so
        //     position b reads the K/V written by positions 0..b of the
        //     same launch.
        let t_verify = Instant::now();
        let verify_logits = target_engine.verify_block_fused_decode(&draft_candidates, l)?;
        ms_verify += t_verify.elapsed().as_secs_f64() * 1000.0;

        // 8d. Accept check, full protocol (docs/dflash.md §6):
        //   preds[0]   = argmax(prev_logits)            (target's pick at L;
        //                                                = bonus_seed)
        //   preds[i+1] = argmax(verify_logits[i])       (target's pick at L+i+1
        //                                                given draft_candidates[..=i] at
        //                                                L..L+i)
        //   accepted = longest j in 0..=B where
        //              preds[0..j] == draft_candidates[0..j].
        //   bonus = preds[accepted]      (target's pick at L+accepted).
        //
        // Note: preds[0] is meaningful. If draft_candidates[0] != bonus_seed
        // we reject immediately (accepted=0) and commit just bonus_seed at
        // position L. This is the "d_0 verified" path — the target's pick
        // at L is authoritative regardless of what the draft said.
        let pred_at = |i: usize, verify: &[Vec<f32>]| -> u32 {
            if i == 0 {
                bonus_seed
            } else {
                DecodeEngine::greedy_sample(&verify[i - 1])
            }
        };
        // Cap accepted at block_size-1 so accepted_len <= block_size — the
        // draft's scratch buffers are sized for ctx_len <= block_size and
        // our next-round round_taps_len = accepted_len.
        let mut accepted = 0usize;
        while accepted < block_size - 1 {
            let pred = pred_at(accepted, &verify_logits);
            if pred == draft_candidates[accepted] {
                accepted += 1;
            } else {
                break;
            }
        }
        // bonus lives at position L+accepted. verify_logits has len = block_size
        // so pred_at(accepted) indexes at most verify_logits[block_size-2].
        let bonus = pred_at(accepted, &verify_logits);

        // 8e. Restore linear, rewind full-attn kv_filled, then re-decode
        //     the committed tokens to rewrite full-attn K/V and capture a
        //     tap row per committed position so the next round's
        //     target_hidden carries `ctx_len = accepted_len` context.
        target_engine
            .state_mut()
            .restore_linear(&snap, ordinal)
            .map_err(|e| anyhow!("restore linear: {e}"))?;
        target_engine.rewind_full_kv_filled(l);

        // Committed sequence this round: [d_0, ..., d_{accepted-1}, bonus]
        // at positions [L, L+1, ..., L+accepted] — length accepted+1.
        let accepted_len = accepted + 1;
        let mut committed_block: Vec<u32> = Vec::with_capacity(accepted_len);
        committed_block.extend_from_slice(&draft_candidates[..accepted]);
        committed_block.push(bonus);

        // Capture taps at every re-decoded position so the next round sees
        // `ctx_len = accepted_len` tap rows covering the newly committed
        // block. target_hidden_raw layout expected by the draft:
        // `[1, ctx_len, num_taps * hidden]` BF16, row-major per ctx pos.
        let per_tap_row_bytes =
            tap_layers.len() * text_config.hidden_size * ScalarType::BF16.size_in_bytes();
        let mut stacked_taps: Vec<u8> =
            Vec::with_capacity(accepted_len * per_tap_row_bytes);
        let mut final_round_logits: Option<Vec<f32>> = None;
        let t_redecode = Instant::now();
        for (i, &tok) in committed_block.iter().enumerate() {
            let seqlen_offset = l + i;
            let (logits, taps_bytes) = target_engine
                .decode_step_with_taps_kernel(tok, seqlen_offset, &tap_layers)?;
            if taps_bytes.len() != per_tap_row_bytes {
                bail!(
                    "decode_step_with_taps_kernel returned {} bytes, expected {}",
                    taps_bytes.len(),
                    per_tap_row_bytes,
                );
            }
            stacked_taps.extend_from_slice(&taps_bytes);
            if i + 1 == committed_block.len() {
                final_round_logits = Some(logits);
            }
        }
        ms_redecode += t_redecode.elapsed().as_secs_f64() * 1000.0;
        round_taps = stacked_taps;
        round_taps_len = accepted_len;

        // 8f. Advance counters + record generated.
        // Stop as soon as any committed token is EOS — every committed
        // token goes through the target's greedy pick, same semantics
        // as the non-DFlash decode loop which bails on the first EOS
        // it samples. Without this, generation would keep rolling past
        // an EOS that appeared inside a speculative commit block.
        committed_len = l + accepted_len;
        accepted_total += accepted;
        let mut hit_eos = false;
        for &t in committed_block.iter() {
            generated_ids.push(t);
            if eos_ids.contains(&t) {
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
        bonus_seed = match final_round_logits {
            Some(logits) => DecodeEngine::greedy_sample(&logits),
            None => bonus, // fallback; should always be Some since accepted_len >= 1
        };

        // 8g. Crop the draft's KV cache in draft-coordinate space.
        // `DFlashState::crop` truncates physical rows, so it must be passed
        // a draft-side row count — not the target-coord `committed_len`
        // (which includes prompt_len and is larger than the draft cursor
        // for any non-empty prompt, silently no-op'ing the crop and
        // leaving the rejected noise tail in cache). Post-forward,
        //   draft_state.kv_filled = kv_pre + ctx_len + block_size
        // and the draft-side noise rows committed this round are the
        // `accepted` rows at the head of the noise block (the bonus is
        // target-picked and has no draft noise row). So the keeper is
        //   kv_pre + ctx_len + accepted
        //     = draft_state.kv_filled - (block_size - accepted)
        // which drops exactly the rejected noise tail.
        let rejected_tail = block_size - accepted;
        let draft_keep = draft_state.kv_filled.saturating_sub(rejected_tail);
        draft_state.crop(draft_keep);
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
    let ms_other = (decode_ms - ms_draft - ms_verify - ms_redecode).max(0.0);
    eprintln!(
        "[dflash] breakdown ms: draft={ms_draft:.0} verify={ms_verify:.0} \
         redecode={ms_redecode:.0} other={ms_other:.0}",
    );

    let _ = draft_rotary; // drop order guard (rotary/scratch hold GPU buffers)
    Ok(())
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

fn load_target_int4_weights(
    cli: &Cli,
    _entry: &RegistryEntry,
    text_config: &qwen35::config::TextConfig,
    ordinal: usize,
    weight_prefix: &str,
) -> Result<Qwen35Weights> {
    let bake_dir = model_store::fetch::BakeVariant::Int4Gptq.bake_dir(&cli.model_dir);
    if !model_store::version_ok(&bake_dir) {
        bail!(
            "Qwen3.5-9B INT4 bake not found at {}. Run:\n  python oracle/bake_int4.py --model-dir {}\n\
             (or run once without --dflash to let the release-bake downloader populate it).",
            bake_dir.display(),
            cli.model_dir.display(),
        );
    }
    let store = model_store::BakedStore::open(&bake_dir)
        .map_err(|e| anyhow!("open target INT4 bake: {e}"))?;
    Qwen35Weights::load_baked(&store, text_config, ordinal, weight_prefix)
        .map_err(|e| anyhow!("load target INT4 weights: {e}"))
}

/// Concatenate per-tap `[hidden_dim]` BF16 blobs into a single
/// `[num_taps * hidden_dim]` byte vector (the draft's `target_hidden_raw`
/// expects this layout for a single ctx position).
fn flatten_tap_blobs(per_tap: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(per_tap.iter().map(|v| v.len()).sum());
    for v in per_tap {
        out.extend_from_slice(v);
    }
    out
}

/// Drive the DFlash draft's forward pass for one round and sample B
/// candidate tokens via the target's `lm_head`. Returns
/// `(candidates, draft_final_hidden_bytes)`.
///
/// Simplifications for M3:
///   * `ctx_len = round_taps_len` (always 1 here post-M3, matching §5.2's
///     first-round T=1 case).
///   * `noise_embedding = embed_tokens([bonus_seed, MASK, …])` — B entries.
fn draft_forward_and_sample(
    draft_state: &mut dflash::state::DFlashState,
    draft_scratch: &mut dflash::state::DFlashScratch,
    draft_rotary: &dflash::RotaryTables,
    draft_weights: &dflash::DFlashWeights,
    target_engine: &DecodeEngine,
    round_taps: &[u8],
    round_taps_len: usize,
    bonus_seed: u32,
    block_size: usize,
    mask_token_id: u32,
    ordinal: usize,
) -> Result<(Vec<u32>, Vec<u8>)> {
    if round_taps_len == 0 {
        bail!("draft_forward: round_taps_len must be > 0");
    }
    let hidden = draft_weights.config.hidden_size;
    let num_taps = draft_weights.config.num_taps();
    let expected_bytes = round_taps_len * num_taps * hidden * ScalarType::BF16.size_in_bytes();
    if round_taps.len() != expected_bytes {
        bail!(
            "round_taps byte length {} != expected {} ({}ctx × {}tap × {}hidden × 2)",
            round_taps.len(),
            expected_bytes,
            round_taps_len,
            num_taps,
            hidden,
        );
    }

    // 1) Build noise_embedding = embed([bonus_seed, MASK, …, MASK]).
    let target_embed = &target_engine.weights().embed_tokens;
    let row_bytes = hidden * ScalarType::BF16.size_in_bytes();
    let noise_embedding =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, block_size, hidden])
            .map_err(|e| anyhow!("alloc noise_embedding: {e}"))?;
    for i in 0..block_size {
        let tok = if i == 0 { bonus_seed } else { mask_token_id };
        let src_off = tok as usize * row_bytes;
        let dst_off = i * row_bytes;
        gpu_hal::copy_d2d(
            ordinal,
            unsafe {
                (noise_embedding.as_ptr() as *mut u8).add(dst_off) as *mut std::ffi::c_void
            },
            target_embed.offset_ptr(src_off),
            row_bytes,
        )
        .map_err(|e| anyhow!("noise_embedding gather slot {i}: {e}"))?;
    }

    // 2) Upload target_hidden_raw [1, round_taps_len, num_taps*hidden].
    let target_hidden_raw = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, round_taps_len, num_taps * hidden],
        round_taps,
    )
    .map_err(|e| anyhow!("upload target_hidden_raw: {e}"))?;

    // 3) Draft forward.
    let pos_offset = draft_state.kv_filled;
    let final_hidden = dflash::forward::forward(
        draft_weights,
        draft_state,
        draft_scratch,
        draft_rotary,
        &noise_embedding,
        &target_hidden_raw,
        dflash::ForwardParams {
            ctx_len: round_taps_len,
            q_len: block_size,
            pos_offset,
        },
    )
    .map_err(|e| anyhow!("draft forward: {e}"))?;

    // 4) lm_head projection → block_logits [block_size, vocab].
    let lm_head = &target_engine.weights().lm_head;
    let vocab = draft_weights.config.vocab_size;
    let mut block_logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, vocab])
        .map_err(|e| anyhow!("alloc block_logits: {e}"))?;
    kernel_ffi::matmul_rhs_transposed_4b(
        ordinal,
        ScalarType::BF16,
        1,          // batch
        block_size, // m
        vocab,      // n
        hidden,     // k
        final_hidden,
        &**lm_head,
        &mut block_logits,
    )
    .map_err(|e| anyhow!("draft lm_head: {e}"))?;

    // 5) D2H + argmax per position.
    let logits_bytes = block_logits
        .to_host_bytes()
        .map_err(|e| anyhow!("block_logits D2H: {e}"))?;
    let mut candidates = Vec::with_capacity(block_size);
    let row_elems = vocab;
    let row_stride_bytes = vocab * ScalarType::BF16.size_in_bytes();
    for i in 0..block_size {
        let start = i * row_stride_bytes;
        let slice = &logits_bytes[start..start + row_stride_bytes];
        let mut best_idx: u32 = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (j, chunk) in slice.chunks_exact(2).enumerate() {
            let v = half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
            if v > best_val {
                best_val = v;
                best_idx = j as u32;
            }
        }
        candidates.push(best_idx);
        let _ = row_elems;
    }

    // Retain the final-hidden bytes for potential debugging; cost is a
    // single D2H of [block_size * hidden] BF16 bytes.
    let draft_final_hidden_bytes = final_hidden
        .to_host_bytes()
        .map_err(|e| anyhow!("draft final_hidden D2H: {e}"))?;

    Ok((candidates, draft_final_hidden_bytes))
}
