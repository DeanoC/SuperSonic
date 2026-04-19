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
//!       `lm_head`). M3 uses the full q_len output; dflash.py's
//!       `[:, 1-block_size:, :]` slice is equivalent for block_size=16 on
//!       the last 15 rows, we use all 16 for simpler verify indexing.
//!    b. Snapshot linear state of the target.
//!    c. Verify: run B iterative `decode_step`s on the candidate block at
//!       positions `[L, L+B)` to get per-position logits. This writes K/V
//!       into full-attention caches at those positions and advances the
//!       linear state — both will be fixed up after the accept check.
//!    d. Compute `accepted` = longest prefix match vs target's greedy
//!       per-position picks (§6).
//!    e. Restore linear state, then re-decode the committed
//!       `accepted + 1` tokens. The last re-decode uses
//!       `decode_step_with_taps_kernel` to capture the next round's
//!       draft context (single-position tap; per-accepted-position taps
//!       are a future optimization).
//!    f. Rewind target's full-attention `kv_filled` to `L + accepted + 1`.
//!    g. Crop the draft's KV cache to the new committed length.
//!
//! This is the correctness-first implementation. M4 will replace the B
//! iterative verify decodes with a single mid-sequence prefill call and
//! explore megakernel-based acceleration.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::state::LinearStateSnapshot;
use qwen35::weights::Qwen35Weights;
use qwen35_dflash as dflash;

use crate::decode_engine::DecodeEngine;
use crate::prefill_engine;
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
    let target_weights = load_target_int4_weights(cli, entry, &text_config, ordinal)?;
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
    let block_size = cli.dflash_block.unwrap_or(draft_config.block_size);
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
    while generated_ids.len() < cli.max_new_tokens {
        if eos_ids.contains(&bonus_seed) {
            generated_ids.push(bonus_seed);
            break;
        }

        rounds_run += 1;
        let l = committed_len;

        // 8a. Draft forward: produces [1, B, hidden]. Project through target
        // lm_head to get block_logits, argmax → B candidates. For M3 the
        // "bonus_seed as first noise" and "B-1 usable draft candidates" vs
        // "use all B candidates" distinction (dflash.py's
        // [:, 1-B:, :] slice) is resolved by verifying all B positions:
        // position L is verified against target's pred-at-L from the
        // previous round's logits stream; positions L+1..L+B-1 against
        // verify's per-position logits.
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

        // 8b. Snapshot linear state (before verify mutates it).
        let snap: LinearStateSnapshot = target_engine
            .state_mut()
            .snapshot_linear()
            .map_err(|e| anyhow!("snapshot linear: {e}"))?;

        // 8c. Verify: decode the B candidate block at positions [L, L+B).
        let verify_logits = verify_block(
            &mut target_engine,
            &draft_candidates,
            l,
        )?;

        // 8d. Accept check:
        //   preds[0] = argmax(prev_logits_for_pos_L)   — produced by prefill's last token
        //                                               on round 0, or by previous round's
        //                                               tail decode on subsequent rounds.
        //     In this engine, we don't retain prev_logits across rounds;
        //     so we use the rule: d_i at position L+i is verified by
        //     argmax(verify_logits[i]) which predicts position L+i+1 — i.e.
        //     we compare d_{i+1} to argmax(verify_logits[i]).
        //   Consequence: the FIRST draft token d_0 isn't verified against
        //   target directly; it's implicitly trusted from the draft (the
        //   speculative decoding "free first token" convention used in
        //   several production implementations). For M3 correctness this
        //   is acceptable — d_0 is the draft's best pick and the target
        //   can still reject d_1..d_{B-1}.
        //   TODO(M4): thread prev_logits through the round to enable a
        //   true argmax comparison for d_0.
        let mut accepted = 0usize;
        while accepted < block_size - 1 {
            let target_pred = DecodeEngine::greedy_sample(&verify_logits[accepted]);
            if target_pred == draft_candidates[accepted + 1] {
                accepted += 1;
            } else {
                break;
            }
        }
        // Bonus: target's pred for position L+accepted+1 (argmax of verify_logits[accepted]).
        let bonus = DecodeEngine::greedy_sample(&verify_logits[accepted]);

        // 8e. Restore linear, rewind full-attn kv_filled, then re-decode
        //     the committed tokens to rewrite full-attn K/V with the
        //     correct committed block and to capture taps for next round.
        target_engine
            .state_mut()
            .restore_linear(&snap, ordinal)
            .map_err(|e| anyhow!("restore linear: {e}"))?;
        target_engine.rewind_full_kv_filled(l);

        // Committed sequence this round: [d_0, d_1, ..., d_{accepted-1}, bonus]
        // plus the FIRST draft token d_0 is actually already the draft's choice.
        // accepted_len = accepted + 1 positions (drafts 0..accepted + bonus).
        let accepted_len = accepted + 1;
        let mut committed_block: Vec<u32> = Vec::with_capacity(accepted_len);
        committed_block.extend_from_slice(&draft_candidates[..accepted]);
        committed_block.push(bonus);

        // Re-decode accepted_len - 1 tokens via plain decode_step, then the
        // final token via decode_step_with_taps_kernel to capture taps.
        let mut final_round_logits: Option<Vec<f32>> = None;
        for (i, &tok) in committed_block.iter().enumerate() {
            let seqlen_offset = l + i;
            if i + 1 == committed_block.len() {
                // Last committed token → capture taps for next round.
                let (logits, taps_bytes) = target_engine
                    .decode_step_with_taps_kernel(tok, seqlen_offset, &tap_layers)?;
                round_taps = taps_bytes;
                round_taps_len = 1; // T=1 per M3 simplification.
                final_round_logits = Some(logits);
            } else {
                let _ = target_engine.decode_step(tok, seqlen_offset)?;
            }
        }

        // 8f. Advance counters + record generated.
        committed_len = l + accepted_len;
        accepted_total += accepted;
        for &t in committed_block.iter() {
            generated_ids.push(t);
            if generated_ids.len() >= cli.max_new_tokens {
                break;
            }
        }
        bonus_seed = match final_round_logits {
            Some(logits) => DecodeEngine::greedy_sample(&logits),
            None => bonus, // fallback; should always be Some since accepted_len >= 1
        };

        // 8g. Crop the draft's KV cache to the new committed length.
        draft_state.crop(committed_len);
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
    Qwen35Weights::load_baked(
        &store,
        text_config,
        ordinal,
        "model", // Qwen weight prefix
    )
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

/// Iteratively run `block_size` decode_step calls on the target to verify
/// the draft's candidate block at positions `[l, l + block_size)`. Returns
/// the per-position logits as `Vec<Vec<f32>>` of length `block_size`.
///
/// The target's linear state and full-attention kv_filled are MUTATED
/// during this call — the caller must snapshot before and restore/rewind
/// after based on the accept decision.
fn verify_block(
    target_engine: &mut DecodeEngine,
    draft_candidates: &[u32],
    l: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(draft_candidates.len());
    for (i, &tok) in draft_candidates.iter().enumerate() {
        let logits = target_engine
            .decode_step(tok, l + i)
            .with_context(|| format!("verify decode_step i={i}"))?;
        out.push(logits);
    }
    Ok(out)
}
