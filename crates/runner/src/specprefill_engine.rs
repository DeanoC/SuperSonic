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
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};
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

    // 1. Drafter dense prefill (fast megakernel path; no decode steps).
    let _base = draft_engine.prefill_native_with_final_norm(prompt_ids)?;

    // 2. Pick scoring layer + mode.
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
        _ => unreachable!("specprefill dispatched for non-qwen35 variant"),
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

    // ---- Decode loop. KV slot and RoPE position are decoupled:
    //      - kv_slot = kept_count + step  (next available slot in the sparse KV cache)
    //      - rope_pos = prompt_len + step  (actual sequence position for RoPE rotation)
    //      This fixes attention math after sparse target prefill: the KV cache only
    //      has kept_count rows, so each new token must write to slot kept_count+step,
    //      but must use its true sequence position for RoPE. ----
    let kept_count = kept_positions.len();
    let prompt_len = prompt_ids.len();
    let mut next_id = DecodeEngine::greedy_sample(&prefill_result.logits);
    let mut generated: Vec<u32> = Vec::new();
    let mut step: usize = 0;
    let eos_ids = target_text_config.eos_token_ids();
    while generated.len() < cli.max_new_tokens {
        if eos_ids.contains(&next_id) {
            generated.push(next_id);
            break;
        }
        generated.push(next_id);
        if generated.len() >= cli.max_new_tokens {
            break;
        }
        let kv_slot = kept_count + step;
        let rope_pos = prompt_len + step;
        let logits = target_engine.decode_step_with_rope_pos(next_id, kv_slot, rope_pos)?;
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
