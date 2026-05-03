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
    let context_tokens = cli.context_size.unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    let kv_per_token = target_text_config.kv_bytes_per_token(ScalarType::BF16.size_in_bytes());
    let target_fixed = entry.vram.fixed_bytes;
    let draft_fixed: u64 = 2 * 1024 * 1024 * 1024; // ~1.6 GiB BF16 + scratch
    let kv_budget = kv_per_token * context_tokens as u64;
    let estimated = ((target_fixed + draft_fixed + kv_budget) as f64 * entry.vram.overhead_factor) as u64;
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
    let draft_weights = Qwen35Weights::load(
        draft_dir,
        &draft_text_config,
        ordinal,
        params.weight_prefix,
    )
    .map_err(|e| anyhow!("load draft weights: {e}"))?;
    eprintln!("[specprefill] draft weights loaded in {:.0}ms", t0.elapsed().as_millis());

    // Build draft decode engine.
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
        false,  // kv_fp8
        1,      // batch_size
    )?;
    draft_engine.set_decode_context_limit(prompt_ids.len() + lookahead_count);

    // ---- Speculator phase: prefill + lookahead + per-layer attention export ----
    let speculator_start = Instant::now();
    let look = draft_engine.prefill_with_lookahead_attention(&prompt_ids, lookahead_count)?;
    eprintln!(
        "[specprefill] speculator done in {:.0}ms (full layers={})",
        speculator_start.elapsed().as_millis(),
        look.layer_scores.len(),
    );

    // ---- Aggregate per-token importance host-side ----
    // Formula (paper §3.3, oracle/specprefill_oracle.py):
    // max over heads → max over layers → mean over lookahead steps.
    let t = prompt_ids.len();
    let q_heads = draft_text_config.num_attention_heads;
    let mut importance = vec![0.0_f32; t];
    for q_row in 0..lookahead_count {
        let mut row = vec![f32::NEG_INFINITY; t];
        for layer_scores in &look.layer_scores {
            // [q_heads, lookahead_count, t] flat
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
            importance[k_pos] += row[k_pos];
        }
    }
    for v in importance.iter_mut() {
        *v /= lookahead_count as f32;
    }

    // ---- Selection ----
    let kept_positions: Vec<u32> = select_kept_positions(&importance, &cfg);
    eprintln!(
        "[specprefill] kept {}/{} tokens ({:.1}%)",
        kept_positions.len(),
        t,
        100.0 * kept_positions.len() as f32 / t as f32
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
        1,  // batch_size
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
    let all: Vec<u32> = prompt_ids.iter().copied().chain(generated.iter().copied()).collect();
    let text = tokenizer
        .decode(&all, true)
        .map_err(|e| anyhow!("detokenize: {e}"))?;
    println!("{text}");
    Ok(())
}
