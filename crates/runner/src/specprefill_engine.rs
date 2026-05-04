//! SpecPrefill (arXiv 2502.02789) end-to-end engine for Qwen3.5-9B target
//! + Qwen3.5-0.8B draft. Greedy decode only. HIP backend.
//!
//! Pattern mirrors `qwen35_dflash_engine.rs`: load both models, drive a
//! custom prefill phase (speculator + selection + sparse target), then
//! hand off to the standard decode loop.

use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::ScalarType;

use qwen35::weights::Qwen35Weights;

use crate::decode_engine::DecodeEngine;
use crate::registry::{self, FamilyParams, ModelVariant, RegistryEntry};
use crate::specprefill::{select_kept_positions, SelectionConfig};
use crate::Cli;

fn score_blocks_cosine(
    draft_engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    block_size: usize,
) -> Result<Vec<f32>> {
    let prompt_len = prompt_ids.len();
    if prompt_len == 0 {
        anyhow::bail!("score_blocks_cosine: empty prompt");
    }

    // 1. Pick scoring layer(s). Resolve BEFORE prefill so we can stop
    //    drafter execution after the deepest scoring layer.
    // Clone the config out of the borrow chain so we can re-borrow draft_engine
    // mutably below (state_mut + cosine kernel launch).
    let config = draft_engine.weights().config.clone();
    let full_layer_idxs: Vec<usize> = (0..config.num_hidden_layers)
        .filter(|&i| config.is_full_attention(i))
        .collect();
    if full_layer_idxs.is_empty() {
        anyhow::bail!("score_blocks_cosine: no full-attention layers in drafter");
    }
    let shallowest = full_layer_idxs[0];
    let layer_override = std::env::var("SUPERSONIC_SPECPREFILL_SCORE_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|i| full_layer_idxs.contains(i));
    let mode_env = std::env::var("SUPERSONIC_SPECPREFILL_LAYERS").ok();
    let mode = mode_env.as_deref().unwrap_or("shallowest");

    let layers_to_score: Vec<usize> = match mode {
        "all_max" => full_layer_idxs.clone(),
        // "shallowest" or any other value: single layer (override or shallowest).
        _ => vec![layer_override.unwrap_or(shallowest)],
    };

    // 2. Drafter prefill, stopping after the deepest scored layer.
    //    For shallowest mode (default) on Qwen3.5-0.8B this prunes most
    //    drafter layers — only layers 0..=shallowest execute. For
    //    all_max mode, runs through the deepest full-attention layer.
    //    Skips final norm + lm_head + KV-FP8 conversion entirely.
    let last_layer = *layers_to_score
        .iter()
        .max()
        .expect("layers_to_score non-empty");
    draft_engine.prefill_native_kv_through(prompt_ids, last_layer)?;

    let n_blocks = (prompt_len + block_size - 1) / block_size;
    let last_pos = prompt_len - 1;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let ordinal = draft_engine.ordinal();

    // 3. Per-layer cosine launch; max-reduce across layers element-wise.
    //
    // K access has to be virtual-aware. With VMM KV enabled (the default
    // on HIP whenever `vmm_is_supported` and `SUPERSONIC_VMM_KV != 0`),
    // the drafter's K is held in `virtual_kv_cache_k` and `kv_cache_k`
    // is None. Mirror the lookahead path's strategy: prefer the dense
    // GpuBuffer when it exists (zero-copy), otherwise materialize a
    // fresh `[1, kv_heads, cap, head_dim]` BF16 GpuBuffer and copy from
    // the virtual backing in one D2D (~kv_heads × cap × head_dim × 2
    // bytes; ~1-2 MB for the 0.8B drafter — negligible vs TTFT).
    let mut block_scores = vec![f32::NEG_INFINITY; n_blocks];
    for li in &layers_to_score {
        let cap = draft_engine.state_mut().layers[*li].kv_capacity();
        let materialized_k: Option<gpu_hal::GpuBuffer> = {
            let layer = &draft_engine.state_mut().layers[*li];
            if layer.kv_cache_k.is_some() {
                None
            } else if layer.has_virtual_kv_cache() {
                let src_ptr = layer.kv_cache_k_offset_ptr(0).ok_or_else(|| {
                    anyhow::anyhow!(
                        "score_blocks_cosine: layer {li}: virtual K cache pointer missing after prefill"
                    )
                })?;
                let bytes_total = kv_heads * cap * head_dim * 2; // BF16
                let mut buf = gpu_hal::GpuBuffer::zeros(
                    ordinal,
                    gpu_hal::ScalarType::BF16,
                    &[1, kv_heads, cap, head_dim],
                )
                .map_err(|e| {
                    anyhow::anyhow!("score_blocks_cosine: layer {li}: kv_k_contig alloc: {e}")
                })?;
                gpu_hal::copy_d2d(ordinal, buf.as_mut_ptr(), src_ptr, bytes_total).map_err(
                    |e| anyhow::anyhow!("score_blocks_cosine: layer {li}: kv_k_contig copy: {e}"),
                )?;
                Some(buf)
            } else {
                anyhow::bail!(
                    "score_blocks_cosine: layer {li}: K cache missing after prefill"
                );
            }
        };
        let k_kernel_input: &gpu_hal::GpuBuffer = if let Some(b) = materialized_k.as_ref() {
            b
        } else {
            draft_engine.state_mut().layers[*li]
                .kv_cache_k
                .as_ref()
                .expect("kv_cache_k present on this branch")
        };

        let mut scores_buf =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::F32, &[n_blocks])
                .map_err(|e| anyhow::anyhow!("scores alloc layer {li}: {e}"))?;

        kernel_ffi::prefill_ffi::pflash_cosine_score(
            ordinal,
            gpu_hal::ScalarType::BF16,
            prompt_len,
            kv_heads,
            cap,
            head_dim,
            block_size,
            last_pos,
            k_kernel_input,
            &mut scores_buf,
        )?;

        let bytes = scores_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("scores d2h layer {li}: {e}"))?;
        let layer_scores: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for (i, s) in layer_scores.iter().enumerate() {
            if *s > block_scores[i] {
                block_scores[i] = *s;
            }
        }
    }

    // 4. Project per-block scores → per-token vector.
    let mut importance = vec![0.0_f32; prompt_len];
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = ((b + 1) * block_size).min(prompt_len);
        for pos in start..end {
            importance[pos] = block_scores[b];
        }
    }

    Ok(importance)
}

/// Cross-family SpecPrefill orchestrator: Qwen3.5-0.8B drafter scoring
/// against a Qwen3.6-MoE target. The drafter is loaded, runs cosine
/// scoring on the prompt, and is dropped before the MoE target loads —
/// 24 GiB on gfx1100 cannot fit drafter + 35B-A3B target + KV
/// concurrently. The keep-mask handed to `qwen36_moe::run_with_sparse_prefill`
/// is what the MoE engine's prefill loop will follow when the
/// loop-side sparse stepping is wired (R1 Step 3).
fn run_specprefill_qwen36_moe(
    cli: &Cli,
    entry: &RegistryEntry,
    ordinal: usize,
    _total_vram: u64,
    draft_dir: &std::path::Path,
) -> Result<()> {
    // Drafter kernel params: Qwen3.5-0.8B is in the Qwen35 family, so
    // we look it up directly rather than try to reuse the Qwen36Moe
    // target's params (which are Qwen36MoeKernelParams, a different
    // shape). Backend + arch match the target's so the right HIP
    // build is selected.
    let drafter_entry =
        registry::lookup(&ModelVariant::Qwen3_5_0_8B, &entry.backend, &entry.arch)
            .ok_or_else(|| {
                anyhow!(
                    "specprefill cross-family: no Qwen3.5-0.8B drafter entry registered \
                     for backend={:?} arch={:?}",
                    entry.backend,
                    entry.arch,
                )
            })?;
    let drafter_params = match &drafter_entry.params {
        FamilyParams::Qwen35(p) => *p,
        _ => unreachable!("Qwen3.5-0.8B drafter not in Qwen35 family"),
    };

    // Tokenize prompt with target's tokenizer. Vocab parity between
    // Qwen3.5 and Qwen3.6 (vocab_size=248320) is what makes cross-family
    // drafting work without a tokenizer bridge.
    //
    // `add_special_tokens = true` mirrors `qwen36_moe_cli::prompt::prepare_prompt`
    // which hard-codes `tok.encode(prompt, true)`. The Qwen3.6-MoE engine
    // re-tokenizes the prompt downstream when it loads the model, so the
    // scorer-side tokenization MUST match the engine-side tokenization or
    // the keep-mask we hand off would be aligned with a different token
    // sequence than the engine processes — the keep-mask length / last-token
    // checks in `decode_text` would then trip even on a valid prompt. We
    // intentionally ignore `cli.prompt_no_special_tokens` here for the same
    // reason; the MoE engine ignores it too.
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        bail!("specprefill: empty prompt");
    }
    eprintln!(
        "[specprefill] cross-family: prompt_tokens={} (Qwen3.5-0.8B drafter → {} target)",
        prompt_ids.len(),
        entry.model,
    );

    let cfg = SelectionConfig {
        keep_ratio: cli.specprefill_keep_ratio.unwrap_or(0.50),
        chunk_size: cli.specprefill_chunk_size.unwrap_or(32),
        pool_window: cli.specprefill_pool_window.unwrap_or(5),
        always_keep_prefix: cli.specprefill_always_keep_prefix.unwrap_or(4),
        always_keep_suffix: cli.specprefill_always_keep_suffix.unwrap_or(4),
    };

    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    // Drafter setup, scoring, and selection — scoped so the drafter
    // weights and decode engine are dropped before we hand off to the
    // MoE engine. On 24 GiB the MoE target won't load if the drafter
    // is still resident.
    let kept_positions: Vec<u32> = {
        let draft_text_config = qwen35::config::load_config(draft_dir)
            .map_err(|e| anyhow!("load draft config: {e}"))?
            .text_config;

        // Validate the drafter is actually Qwen3.5-0.8B before applying
        // 0.8B-tuned kernel sizing (`drafter_params.proj_buf_floats`,
        // `attn_scratch_floats`). Qwen3.5-0.8B is the only same-family
        // drafter that fits on 24 GiB alongside Qwen3.6-MoE INT4 + KV;
        // a larger drafter (4B/9B) pointed at by `--specprefill-draft-dir`
        // would still load but be undersized for its scratch and projection
        // buffers, leading to opaque kernel launch / memory errors deep
        // inside the chained-decode dispatch. Fail-fast here with a clear
        // error.
        const QWEN35_0_8B_HIDDEN: usize = 1024;
        const QWEN35_0_8B_LAYERS: usize = 24;
        if draft_text_config.hidden_size != QWEN35_0_8B_HIDDEN
            || draft_text_config.num_hidden_layers != QWEN35_0_8B_LAYERS
        {
            bail!(
                "specprefill cross-family: --specprefill-draft-dir must point at \
                 Qwen3.5-0.8B (hidden_size={}, num_hidden_layers={}). Got \
                 hidden_size={}, num_hidden_layers={} from {}. Larger Qwen3.5 \
                 variants don't fit on 24 GiB alongside the Qwen3.6-MoE target \
                 and would receive undersized 0.8B-tuned kernel scratch buffers.",
                QWEN35_0_8B_HIDDEN,
                QWEN35_0_8B_LAYERS,
                draft_text_config.hidden_size,
                draft_text_config.num_hidden_layers,
                draft_dir.display(),
            );
        }

        let t0 = Instant::now();
        let draft_weights = Qwen35Weights::load(
            draft_dir,
            &draft_text_config,
            ordinal,
            drafter_params.weight_prefix,
        )
        .map_err(|e| anyhow!("load draft weights: {e}"))?;
        eprintln!(
            "[specprefill] draft weights loaded in {:.0}ms",
            t0.elapsed().as_millis()
        );

        let lookahead_count = cli.specprefill_lookahead.unwrap_or(4) + 1;
        let draft_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
            draft_text_config.num_attention_heads,
            draft_text_config.head_dim,
            prompt_ids.len() + lookahead_count,
            drafter_params.kv_chunk_size,
        );
        let mut draft_engine = DecodeEngine::new(
            draft_weights,
            ordinal,
            drafter_params.proj_buf_floats,
            draft_attn_scratch.max(drafter_params.attn_scratch_floats),
            drafter_params.kv_chunk_size,
            drafter_params.use_4b_kernel,
            cli.prefill_chunk_size,
            false, // kv_fp8 — drafter doesn't carry FP8 KV
            1,     // batch_size
        )?;
        draft_engine.set_decode_context_limit(prompt_ids.len() + lookahead_count);

        let block_size_for_scoring = cli.specprefill_chunk_size.unwrap_or(32);
        let speculator_start = Instant::now();
        let importance: Vec<f32> = match cli.specprefill_algorithm.as_str() {
            "cosine" => {
                let imp = score_blocks_cosine(
                    &mut draft_engine,
                    &prompt_ids,
                    block_size_for_scoring,
                )?;
                eprintln!(
                    "[specprefill] speculator (cosine) done in {:.0}ms (block_size={})",
                    speculator_start.elapsed().as_millis(),
                    block_size_for_scoring,
                );
                imp
            }
            other => bail!(
                "specprefill cross-family on Qwen3.6-MoE supports only \
                 --specprefill-algorithm=cosine (got {other:?}). The lookahead \
                 path runs drafter decode and is gated to same-family targets."
            ),
        };

        let kept = select_kept_positions(&importance, &cfg);
        if kept.last().copied() != Some((prompt_ids.len() - 1) as u32) {
            bail!(
                "specprefill: kept_positions must include the last prompt position \
                 (got last={:?}); the first decode logits come from that slot",
                kept.last(),
            );
        }
        eprintln!(
            "[specprefill] kept {}/{} tokens ({:.1}%)",
            kept.len(),
            prompt_ids.len(),
            100.0 * kept.len() as f32 / prompt_ids.len() as f32
        );
        kept
        // draft_engine and draft_weights drop here, freeing drafter VRAM.
    };

    let mut keep_mask = vec![false; prompt_ids.len()];
    for &p in &kept_positions {
        keep_mask[p as usize] = true;
    }

    // Hand off to the MoE engine. It re-tokenizes (same tokenizer file +
    // same prompt → identical IDs), sets up the target weights with VMM,
    // and runs the prefill+decode loop with sparse stepping driven by
    // `keep_mask`.
    crate::qwen36_moe_cli::run_with_sparse_prefill(cli, entry, _total_vram, keep_mask)
}

pub fn run_specprefill(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    // Validation already happened in policy::validate_specprefill_flags;
    // we just unwrap the draft dir here.
    let draft_dir = cli
        .specprefill_draft_dir
        .as_ref()
        .ok_or_else(|| anyhow!("specprefill: missing --specprefill-draft-dir"))?;

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => *p,
        FamilyParams::Qwen36Moe(_) => {
            // Cross-family path: Qwen3.5-0.8B drafter → Qwen3.6-MoE target.
            // Target setup is structurally different (MoE expert paging via
            // VMM, single-token-at-a-time prefill in qwen36_moe::engine),
            // so we score with the drafter here and hand the resulting
            // keep-mask off to qwen36_moe::run_with_sparse_prefill instead
            // of falling through to the Qwen3.5 target path below.
            return run_specprefill_qwen36_moe(cli, entry, ordinal, total_vram, draft_dir);
        }
        _ => unreachable!("specprefill dispatched for non-qwen35-family variant"),
    };

    // ---- Tokeniser + target config ----
    let target_text_config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow!("load target config: {e}"))?
        .text_config;
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
        .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        bail!("specprefill: empty prompt");
    }
    eprintln!("[specprefill] prompt_tokens={}", prompt_ids.len());

    // ---- Selection config ----
    let cfg = SelectionConfig {
        keep_ratio: cli.specprefill_keep_ratio.unwrap_or(0.50),
        chunk_size: cli.specprefill_chunk_size.unwrap_or(32),
        pool_window: cli.specprefill_pool_window.unwrap_or(5),
        always_keep_prefix: cli.specprefill_always_keep_prefix.unwrap_or(4),
        always_keep_suffix: cli.specprefill_always_keep_suffix.unwrap_or(4),
    };

    // ---- VRAM budget (rough) ----
    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    let kv_per_token = target_text_config.kv_bytes_per_token(ScalarType::BF16.size_in_bytes());
    let target_fixed = entry.vram.fixed_bytes;
    let draft_fixed: u64 = 2 * 1024 * 1024 * 1024; // ~1.6 GiB BF16 + scratch
    let kv_budget = kv_per_token * context_tokens as u64;
    let estimated =
        ((target_fixed + draft_fixed + kv_budget) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[specprefill] vram estimate target={:.2}GiB draft={:.2}GiB kv={:.2}GiB total={:.2}GiB available={:.2}GiB",
        gib(target_fixed),
        gib(draft_fixed),
        gib(kv_budget),
        gib(estimated),
        gib(total_vram),
    );
    if estimated > total_vram {
        bail!(
            "SpecPrefill VRAM budget exceeded: need ~{:.2} GiB, have {:.2} GiB. \
             Try --specprefill-unload-draft or reduce --context-size.",
            gib(estimated),
            gib(total_vram),
        );
    }

    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    // ---- Load draft (BF16 from safetensors directly — no bakes for the speculator) ----
    let draft_text_config = qwen35::config::load_config(draft_dir)
        .map_err(|e| anyhow!("load draft config: {e}"))?
        .text_config;
    if draft_text_config.vocab_size != target_text_config.vocab_size {
        bail!(
            "draft vocab_size {} != target vocab_size {} — same-family check failed",
            draft_text_config.vocab_size,
            target_text_config.vocab_size,
        );
    }
    let t0 = Instant::now();
    let draft_weights =
        Qwen35Weights::load(draft_dir, &draft_text_config, ordinal, params.weight_prefix)
            .map_err(|e| anyhow!("load draft weights: {e}"))?;
    eprintln!(
        "[specprefill] draft weights loaded in {:.0}ms",
        t0.elapsed().as_millis()
    );

    // Build draft decode engine. lookahead_count is used by both the
    // legacy lookahead path AND for sizing the draft engine's decode
    // context (in case the cosine path also exercises decode in future).
    let lookahead_count = cli.specprefill_lookahead.unwrap_or(4) + 1;
    let draft_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        draft_text_config.num_attention_heads,
        draft_text_config.head_dim,
        prompt_ids.len() + lookahead_count,
        params.kv_chunk_size,
    );
    let mut draft_engine = DecodeEngine::new(
        draft_weights,
        ordinal,
        params.proj_buf_floats,
        draft_attn_scratch.max(params.attn_scratch_floats),
        params.kv_chunk_size,
        params.use_4b_kernel,
        cli.prefill_chunk_size,
        false, // kv_fp8
        1,     // batch_size
    )?;
    draft_engine.set_decode_context_limit(prompt_ids.len() + lookahead_count);

    // ---- Speculator phase: route on --specprefill-algorithm ----
    let block_size_for_scoring = cli.specprefill_chunk_size.unwrap_or(32);
    let speculator_start = Instant::now();
    let importance: Vec<f32> = match cli.specprefill_algorithm.as_str() {
        "cosine" => {
            let imp = score_blocks_cosine(
                &mut draft_engine,
                &prompt_ids,
                block_size_for_scoring,
            )?;
            eprintln!(
                "[specprefill] speculator (cosine) done in {:.0}ms (block_size={})",
                speculator_start.elapsed().as_millis(),
                block_size_for_scoring,
            );
            imp
        }
        _ => {
            // Legacy Phase C lookahead path.
            let look = draft_engine
                .prefill_with_lookahead_attention(&prompt_ids, lookahead_count)?;
            eprintln!(
                "[specprefill] speculator (lookahead) done in {:.0}ms (full layers={})",
                speculator_start.elapsed().as_millis(),
                look.layer_scores.len(),
            );
            // Aggregation: max over heads → max over layers → mean over lookahead steps.
            let t = prompt_ids.len();
            let q_heads = draft_text_config.num_attention_heads;
            let mut imp = vec![0.0_f32; t];
            for q_row in 0..lookahead_count {
                let mut row = vec![f32::NEG_INFINITY; t];
                for layer_scores in &look.layer_scores {
                    for q_head in 0..q_heads {
                        let base = (q_head * lookahead_count + q_row) * t;
                        for k_pos in 0..t {
                            let v = layer_scores[base + k_pos];
                            if v > row[k_pos] {
                                row[k_pos] = v;
                            }
                        }
                    }
                }
                for k_pos in 0..t {
                    imp[k_pos] += row[k_pos];
                }
            }
            for v in imp.iter_mut() {
                *v /= lookahead_count as f32;
            }
            imp
        }
    };

    // ---- Selection ----
    let kept_positions: Vec<u32> = select_kept_positions(&importance, &cfg);

    // Invariant: the last prompt position must always be kept so that the first
    // decode token is sampled from the correct context.  The validator rejects
    // --specprefill-always-keep-suffix 0, but this assert catches any future
    // regression in select_kept_positions itself.
    if kept_positions.last().copied() != Some((prompt_ids.len() - 1) as u32) {
        bail!(
            "specprefill: kept_positions must include the last prompt position \
             (got last={:?}, expected={}). Increase --specprefill-always-keep-suffix.",
            kept_positions.last(),
            prompt_ids.len() - 1
        );
    }

    eprintln!(
        "[specprefill] kept {}/{} tokens ({:.1}%)",
        kept_positions.len(),
        prompt_ids.len(),
        100.0 * kept_positions.len() as f32 / prompt_ids.len() as f32
    );

    // ---- (Optional) unload draft to recover VRAM ----
    if cli.specprefill_unload_draft {
        drop(draft_engine);
        eprintln!("[specprefill] draft unloaded");
    }

    // ---- Load target ----
    let bootstrap_downloaded = crate::bakes::ensure_hf_metadata_present(cli, model_variant)?;
    let q4km_like = crate::policy::q4km_like(cli);
    let target_weights = crate::bakes::load_qwen35_weights(
        cli,
        model_variant,
        &target_text_config,
        ordinal,
        params.weight_prefix,
        bootstrap_downloaded,
        q4km_like,
    )?;

    let target_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        target_text_config.num_attention_heads,
        target_text_config.head_dim,
        context_tokens,
        params.kv_chunk_size,
    );
    let mut target_engine = DecodeEngine::new(
        target_weights,
        ordinal,
        params.proj_buf_floats,
        target_attn_scratch.max(params.attn_scratch_floats),
        params.kv_chunk_size,
        params.use_4b_kernel,
        cli.prefill_chunk_size,
        cli.kv_fp8,
        1, // batch_size
    )?;
    target_engine.set_decode_context_limit(context_tokens);

    // ---- Target sparse prefill ----
    let prefill_start = Instant::now();
    let prefill_result = target_engine.prefill_kept_native(&prompt_ids, &kept_positions)?;
    eprintln!(
        "[specprefill] target prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );

    // ---- LAST_LOGITS dump for parity tests ----
    if cli.dump_last_logits {
        use std::io::Write as _;
        print!("\nLAST_LOGITS: ");
        for (i, x) in prefill_result.logits.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    // ---- Decode loop. Two paths:
    //
    // (1) Default (BF16 KV cache): incremental decode via
    //     `decode_step_with_rope_pos`. KV slot and RoPE position are
    //     decoupled:
    //       - kv_slot = kept_count + step  (next available slot in the sparse cache)
    //       - rope_pos = prompt_len + step  (actual sequence position for RoPE)
    //     The KV cache only has kept_count rows after sparse prefill, so each new
    //     token writes to slot kept_count+step with its true RoPE rotation.
    //
    // (2) `--kv-fp8`: replay-prefill per step. Component decode (the path
    //     `decode_step_with_rope_pos` takes for use_4b_kernel models) writes
    //     K/V via `copy_step_bf16` and reads via the BF16 attention kernel
    //     `full_attention_decode_flat`; neither has an FP8 path. Plain
    //     `--kv-fp8` already replays prefill per decode step (see main.rs's
    //     `replay_kv_fp8_enabled`); we mirror that here. The first prefill
    //     keeps SpecPrefill's TTFT win (sparse `prefill_kept`); each decode
    //     step then runs a dense replay-prefill on the full unsparsified
    //     history. Decode tokens-per-second is bounded by replay-prefill
    //     cost, same as plain `--kv-fp8` decode on Qwen3.5-9B. ----
    let kept_count = kept_positions.len();
    let prompt_len = prompt_ids.len();
    let mut next_id = DecodeEngine::greedy_sample(&prefill_result.logits);
    let mut generated: Vec<u32> = Vec::new();
    let mut step: usize = 0;
    let eos_ids = target_text_config.eos_token_ids();
    let kv_fp8_replay = cli.kv_fp8;
    while generated.len() < cli.max_new_tokens {
        if eos_ids.contains(&next_id) {
            generated.push(next_id);
            break;
        }
        generated.push(next_id);
        if generated.len() >= cli.max_new_tokens {
            break;
        }
        let logits = if kv_fp8_replay {
            let mut token_ids: Vec<u32> = Vec::with_capacity(prompt_len + generated.len());
            token_ids.extend_from_slice(&prompt_ids);
            token_ids.extend_from_slice(&generated);
            target_engine.rebuild_prefill_state(&token_ids, false)?
        } else {
            let kv_slot = kept_count + step;
            let rope_pos = prompt_len + step;
            target_engine.decode_step_with_rope_pos(next_id, kv_slot, rope_pos)?
        };
        next_id = DecodeEngine::greedy_sample(&logits);
        step += 1;
    }

    // ---- Detokenise + print ----
    let all: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all, true)
        .map_err(|e| anyhow!("detokenize: {e}"))?;
    println!("{text}");
    Ok(())
}
