use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;
use std::time::Instant;

use crate::bakes::{ensure_gemma4_int4_bake, ensure_hf_metadata_present};
use crate::profiling::PrefillProfileScope;
use crate::registry::{self, Backend, FamilyParams, Gemma4KernelParams, ModelVariant};
use crate::{gemma4_engine, gemma4_int4_engine, oracle, validate, Cli};

pub(crate) struct Gemma4Startup {
    pub(crate) cfg: gemma4::config::Config,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) context_tokens: usize,
}

/// Dispatcher over Gemma 4 runtime engines. BF16 runs through the persistent
/// megakernel; INT4 runs through the primitive chain backed by the GPTQ bake.
pub(crate) enum Gemma4Runtime {
    Bf16(gemma4_engine::Gemma4Engine),
    Int4(gemma4_int4_engine::Gemma4Int4Engine),
}

impl Gemma4Runtime {
    pub(crate) fn prefill(&mut self, prompt_token_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.prefill(prompt_token_ids),
            Self::Int4(e) => e.prefill(prompt_token_ids),
        }
    }

    /// Run one decode step on every sequence in the batch. Both BF16 and
    /// INT4 engines honour `--batch-size > 1` via their batched persistent
    /// megakernels (BF16: `g4::persistent_decode_batch`, INT4:
    /// `g4::persistent_decode_batch_int4`).
    pub(crate) fn decode_step_batch(
        &mut self,
        input_tokens: &[u32],
        positions: &[usize],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Self::Bf16(e) => e.decode_step_batch(input_tokens, positions),
            Self::Int4(e) => e.decode_step_batch(input_tokens, positions),
        }
    }

    pub(crate) fn decode_step_batch_greedy_cuda(
        &mut self,
        input_tokens: &[u32],
        positions: &[usize],
    ) -> anyhow::Result<Option<Vec<u32>>> {
        match self {
            Self::Bf16(e) => {
                if input_tokens.len() == 1 && positions.len() == 1 {
                    Ok(
                        e.decode_step_seq_greedy_cuda(0, input_tokens[0], positions[0])?
                            .map(|tok| vec![tok]),
                    )
                } else {
                    Ok(None)
                }
            }
            Self::Int4(_) => Ok(None),
        }
    }

    /// Replicate seq 0's K/V cache contents into every other sequence's
    /// caches. Applies to both BF16 and INT4 engines.
    pub(crate) fn replicate_seq0_kv(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Bf16(e) => e.replicate_seq0_kv(),
            Self::Int4(e) => e.replicate_seq0_kv(),
        }
    }

    /// Run one decode step on sequence 0 with the given token and position.
    /// Returns the full logit vector for sequence 0.
    pub(crate) fn decode_step(&mut self, input_token: u32, pos: usize) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.decode_step(input_token, pos),
            Self::Int4(e) => e.decode_step(input_token, pos),
        }
    }

    pub(crate) fn batch_size(&self) -> usize {
        match self {
            Self::Bf16(e) => e.batch_size(),
            Self::Int4(e) => e.batch_size(),
        }
    }

    pub(crate) fn greedy_sample(logits: &[f32]) -> u32 {
        gemma4_engine::Gemma4Engine::greedy_sample(logits)
    }
}

pub(crate) fn load_gemma4_runtime(
    cli: &Cli,
    model_variant: &ModelVariant,
    params: &Gemma4KernelParams,
    context_tokens: usize,
    ordinal: usize,
    bootstrap_downloaded: bool,
) -> Result<Gemma4Runtime> {
    if cli.int4 {
        ensure_gemma4_int4_bake(cli, model_variant, bootstrap_downloaded)?;
        eprintln!("[gemma4] loading INT4 GPTQ bake (primitive-chain decode)");
        Ok(Gemma4Runtime::Int4(
            gemma4_int4_engine::Gemma4Int4Engine::load_with_batch(
                &cli.model_dir,
                params.weight_prefix,
                context_tokens,
                ordinal,
                cli.batch_size,
            )?,
        ))
    } else {
        Ok(Gemma4Runtime::Bf16(
            gemma4_engine::Gemma4Engine::load_with_quant(
                &cli.model_dir,
                params.weight_prefix,
                context_tokens,
                ordinal,
                cli.batch_size,
                cli.kv_fp8,
                cli.fp8_runtime,
            )?,
        ))
    }
}

pub(crate) fn run_gemma4(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &registry::RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    let params = match &entry.params {
        FamilyParams::Gemma4(p) => p,
        FamilyParams::Qwen35(_) => unreachable!("dispatch filtered to Gemma4"),
        FamilyParams::Qwen3Moe(_) => unreachable!("dispatch filtered to Gemma4"),
        FamilyParams::Qwen36Moe(_) => unreachable!("dispatch filtered to Gemma4"),
        FamilyParams::Phi4(_) => unreachable!("dispatch filtered to Gemma4"),
        FamilyParams::Llama31(_) => unreachable!("dispatch filtered to Gemma4"),
    };

    validate_gemma4_startup(cli, model_variant, entry.backend)?;

    // Fetch first if --model-dir is pristine so HF metadata lands before config load.
    let bootstrap_downloaded = ensure_hf_metadata_present(cli, model_variant)?;

    let Gemma4Startup {
        cfg,
        tokenizer,
        prompt_ids,
        context_tokens,
    } = load_gemma4_startup(cli, model_variant, params)?;
    let t = &cfg.text_config;

    check_gemma4_vram(cli, t, &entry.vram, context_tokens, total_vram)?;

    let t0 = Instant::now();
    let mut engine = load_gemma4_runtime(
        cli,
        model_variant,
        params,
        context_tokens,
        ordinal,
        bootstrap_downloaded,
    )?;
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    // Teacher-forced scoring: do not proceed to normal prefill+decode.
    if cli.teacher_forced {
        return run_gemma4_teacher_forced(
            cli,
            model_variant,
            entry.backend,
            &mut engine,
            &prompt_ids,
        );
    }

    let oracle_output = if cli.validate {
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("oracle/gemma4_oracle.py");
        let oracle = oracle::run_gemma4_oracle(
            &oracle_script,
            &cli.model_dir,
            &cli.prompt,
            cli.max_new_tokens,
            &cli.oracle_dtype,
        )?;
        if let Some(ref oracle_ids) = oracle.prompt_token_ids {
            if oracle_ids != &prompt_ids {
                anyhow::bail!(
                    "tokenizer mismatch between Rust and Python oracle: rust={prompt_ids:?} oracle={oracle_ids:?}"
                );
            }
        }
        Some(oracle)
    } else {
        None
    };

    let prefill_start = Instant::now();
    let profile = PrefillProfileScope::new(
        cli.profile_prefill,
        cli.profile_prefill_json.as_deref(),
        "gemma4",
        &cli.model,
        &cli.backend,
        prompt_ids.len(),
    );
    let prefill_logits = engine.prefill(&prompt_ids)?;
    let prefill_token = Gemma4Runtime::greedy_sample(&prefill_logits);
    eprintln!(
        "[prefill] native GPU prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );
    profile.finish()?;

    let batch_size = engine.batch_size();
    if batch_size > 1 {
        eprintln!(
            "[batch] replicating prefill K/V across {} sequences",
            batch_size
        );
        engine.replicate_seq0_kv()?;
    }

    if let Some(ref oracle) = oracle_output {
        let prefill_delta = validate::max_abs_delta(&prefill_logits, &oracle.prefill_logits);
        eprintln!("[validate] prefill logit delta={prefill_delta:.4}");
        if let Some(&oracle_first) = oracle.generated_token_ids.first() {
            if oracle_first != prefill_token {
                eprintln!(
                    "[validate] WARNING: prefill token mismatch! native={prefill_token} oracle={oracle_first}"
                );
            }
        }
        if batch_size > 1 {
            eprintln!("[validate] WARNING: --validate compares oracle vs sequence 0 only when --batch-size > 1");
        }
    }

    let seqlen_start = prompt_ids.len();
    let eos_ids = t.eos_token_ids();
    let mut max_delta = 0.0f32;
    let mut token_mismatches = 0usize;

    // Per-sequence decode state. All sequences start from the same prefill
    // token; greedy sampling will keep them identical unless something
    // diverges (useful sanity check until Phase 2 adds true per-sequence
    // prompts).
    let mut next_tokens: Vec<u32> = vec![prefill_token; batch_size];
    let mut generated_per_seq: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
    let mut seq_done: Vec<bool> = vec![false; batch_size];
    let mut steps_done: usize = 0;

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        // Mark any newly-EOSed sequences but keep stepping until ALL sequences
        // have stopped - the megakernel still has to handle the active ones.
        for b in 0..batch_size {
            if !seq_done[b] && eos_ids.contains(&next_tokens[b]) {
                seq_done[b] = true;
            }
        }
        if seq_done.iter().all(|d| *d) {
            break;
        }
        let pos = seqlen_start + step;
        let positions: Vec<usize> = vec![pos; batch_size];
        let fast_sampled = if oracle_output.is_none() {
            engine.decode_step_batch_greedy_cuda(&next_tokens, &positions)?
        } else {
            None
        };
        let logits_per_seq = if fast_sampled.is_none() {
            Some(engine.decode_step_batch(&next_tokens, &positions)?)
        } else {
            None
        };

        if let (Some(ref oracle), Some(ref logits_per_seq)) = (&oracle_output, &logits_per_seq) {
            if step < oracle.decode_logits.len() {
                let oracle_logits = &oracle.decode_logits[step];
                // Always compare against sequence 0 (canonical run).
                let delta = validate::max_abs_delta(&logits_per_seq[0], oracle_logits);
                if delta > max_delta {
                    max_delta = delta;
                }
                let oracle_next = if step + 1 < oracle.generated_token_ids.len() {
                    Some(oracle.generated_token_ids[step + 1])
                } else {
                    None
                };
                let rust_next = Gemma4Runtime::greedy_sample(&logits_per_seq[0]);
                let mismatch_tag = match oracle_next {
                    Some(ot) if ot != rust_next => {
                        token_mismatches += 1;
                        format!(" MISMATCH (oracle_next={ot})")
                    }
                    _ => String::new(),
                };
                eprintln!(
                    "[validate] step={step} pos={pos} delta={delta:.4} input_tok={} rust_next={rust_next}{mismatch_tag}",
                    next_tokens[0]
                );
            }
        }

        // Sample per sequence and roll forward - but only record sampled
        // tokens for sequences that haven't already hit EOS.
        for b in 0..batch_size {
            if seq_done[b] {
                continue;
            }
            let sampled = if let Some(ref sampled_tokens) = fast_sampled {
                sampled_tokens[b]
            } else {
                let logits_per_seq = logits_per_seq
                    .as_ref()
                    .expect("full logits populated when fast sampling is unavailable");
                Gemma4Runtime::greedy_sample(&logits_per_seq[b])
            };
            generated_per_seq[b].push(next_tokens[b]);
            next_tokens[b] = sampled;
        }
        steps_done = step + 1;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    // Print every sequence. For batch_size == 1 the output matches the
    // pre-batched format (no `[seq=N]` prefix).
    for b in 0..batch_size {
        let all_ids: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(generated_per_seq[b].iter().copied())
            .collect();
        let text = tokenizer
            .decode(&all_ids, true)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
        let generated_text = tokenizer
            .decode(&generated_per_seq[b], true)
            .map_err(|e| anyhow::anyhow!("detokenize generated suffix: {e}"))?;
        if batch_size == 1 {
            println!("{text}");
            if cli.emit_generated_json {
                println!(
                    "[generated_json] {}",
                    serde_json::to_string(&generated_text)?
                );
            }
            println!(
                "[tokens] {}",
                generated_per_seq[b]
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        } else {
            println!("[seq={b}] {text}");
            if cli.emit_generated_json {
                println!(
                    "[seq={b}][generated_json] {}",
                    serde_json::to_string(&generated_text)?
                );
            }
            println!(
                "[seq={b}][tokens] {}",
                generated_per_seq[b]
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }

    let total_generated: usize = generated_per_seq.iter().map(|v| v.len()).sum();
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} steps={steps_done} batch_size={batch_size} decode_ms={decode_ms:.0} ms_per_step={:.0}{}",
        prompt_ids.len(),
        total_generated,
        if steps_done == 0 {
            0.0
        } else {
            decode_ms / steps_done as f64
        },
        if oracle_output.is_some() {
            format!(" decode_max_delta={max_delta:.4} token_mismatches={token_mismatches}")
        } else {
            String::new()
        },
    );
    Ok(())
}

pub(crate) fn validate_gemma4_startup(
    cli: &Cli,
    model_variant: &ModelVariant,
    backend: Backend,
) -> Result<()> {
    if backend == Backend::Cuda {
        if !matches!(model_variant, ModelVariant::Gemma4_E2B) {
            anyhow::bail!("Gemma 4 CUDA v1 supports only --model gemma4-e2b on sm86");
        }
        if cli.int4 {
            anyhow::bail!("Gemma 4 CUDA v1 supports BF16 only; --int4 is not wired");
        }
        if cli.fp8_runtime {
            anyhow::bail!("Gemma 4 CUDA v1 supports BF16 only; --fp8-runtime is not wired");
        }
        if cli.kv_fp8 {
            anyhow::bail!("Gemma 4 CUDA v1 supports BF16 KV only; --kv-fp8 is not wired");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("Gemma 4 CUDA v1 supports only --batch-size=1");
        }
    }

    if cli.fp8_runtime && cli.int4 {
        anyhow::bail!(
            "Gemma 4 --fp8-runtime cannot combine with --int4 (the INT4 kernel \
             does not yet route the FP8 weight-dequant path)"
        );
    }
    if cli.fp8_runtime && cli.batch_size != 1 {
        anyhow::bail!(
            "Gemma 4 --fp8-runtime currently requires --batch-size=1 (FP8 weight \
             dequant is wired into the single-batch persistent decode kernel only)"
        );
    }
    if cli.kv_fp8 && cli.int4 {
        anyhow::bail!(
            "Gemma 4 --kv-fp8 cannot combine with --int4 yet (kernel FP8 KV \
             path lives in the BF16 single-batch persistent decode kernel; \
             INT4 + FP8-KV would need the INT4 kernel updated too)"
        );
    }
    if cli.kv_fp8 && cli.batch_size != 1 {
        anyhow::bail!(
            "Gemma 4 --kv-fp8 currently requires --batch-size=1 (FP8 KV path \
             only wired into the single-batch persistent decode kernel)"
        );
    }
    if cli.batch_size < 1 || cli.batch_size > kernel_ffi::MAX_BATCH_SIZE {
        anyhow::bail!("--batch-size must be 1..{}", kernel_ffi::MAX_BATCH_SIZE);
    }
    if cli.oracle_prefill || cli.gpu_validate {
        anyhow::bail!("Gemma 4 does not yet support --oracle-prefill / --gpu-validate");
    }
    if cli.prefill_chunk_size != 0 {
        anyhow::bail!(
            "Gemma 4 does not yet support --prefill-chunk-size (single-shot prefill only)"
        );
    }
    if cli.no_bake && !cli.int4 {
        eprintln!(
            "[gemma4] note: --no-bake is implied for BF16 (Gemma 4 has no BF16 bake format). \
             Loading directly from safetensors."
        );
    }
    Ok(())
}

pub(crate) fn load_gemma4_startup(
    cli: &Cli,
    model_variant: &ModelVariant,
    params: &Gemma4KernelParams,
) -> Result<Gemma4Startup> {
    let cfg = gemma4::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading Gemma 4 config.json: {e}"))?;
    let t = &cfg.text_config;
    eprintln!(
        "[gemma4] variant={model_variant} weight_prefix={} kv_chunk={}",
        params.weight_prefix, params.kv_chunk_size
    );
    eprintln!(
        "[gemma4] hidden={} layers={} vocab={} heads={}/{} head_dim={}/{} window={} kv_shared_layers={} softcap={:?} ple_dim={} tied_lm_head={}",
        t.hidden_size,
        t.num_hidden_layers,
        t.vocab_size,
        t.num_attention_heads,
        t.num_key_value_heads,
        t.head_dim,
        t.global_head_dim,
        t.sliding_window,
        t.num_kv_shared_layers,
        t.final_logit_softcapping,
        t.hidden_size_per_layer_input,
        cfg.tie_word_embeddings || t.tie_word_embeddings,
    );

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        anyhow::bail!("empty prompt after tokenization");
    }

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    if context_tokens < prompt_ids.len() + cli.max_new_tokens {
        anyhow::bail!(
            "--context-size {context_tokens} < prompt_tokens {} + max_new_tokens {}",
            prompt_ids.len(),
            cli.max_new_tokens,
        );
    }

    Ok(Gemma4Startup {
        cfg,
        tokenizer,
        prompt_ids,
        context_tokens,
    })
}

pub(crate) fn check_gemma4_vram(
    cli: &Cli,
    text_config: &gemma4::config::TextConfig,
    vram: &registry::VramBudget,
    context_tokens: usize,
    total_vram: u64,
) -> Result<()> {
    // Per-token KV element count across owned layers; shared layers alias.
    let mut kv_elems_per_token: u64 = 0;
    let mut owned_layers: u64 = 0;
    for l in 0..text_config.num_hidden_layers {
        if text_config.kv_source_layer(l).is_none() {
            let kind = text_config
                .attn_kind(l)
                .ok_or_else(|| anyhow::anyhow!("layer {l}: no attention kind"))?;
            let hd = text_config.head_dim_for(kind);
            kv_elems_per_token += (text_config.num_key_value_heads * hd * 2) as u64;
            owned_layers += 1;
        }
    }

    let kv_dtype_bytes: u64 = if cli.kv_fp8 { 1 } else { 2 };
    let scale_bytes_per_seq: u64 = if cli.kv_fp8 {
        2 * (text_config.num_key_value_heads as u64) * (context_tokens as u64) * 4 * owned_layers
    } else {
        0
    };
    let kv_bytes_per_seq =
        kv_elems_per_token * context_tokens as u64 * kv_dtype_bytes + scale_bytes_per_seq;
    let kv_bytes = kv_bytes_per_seq * cli.batch_size as u64;

    // The registry budget is BF16 weights + scratch. Quantized modes shrink
    // weights but still keep scratch, so use the same conservative factors as
    // the other runner preflights.
    let quant_fixed_bytes = if cli.int4 {
        (vram.fixed_bytes as f64 * 0.37) as u64
    } else if cli.fp8_runtime {
        (vram.fixed_bytes as f64 * 0.6) as u64
    } else {
        vram.fixed_bytes
    };
    let estimated_vram = ((quant_fixed_bytes + kv_bytes) as f64 * vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let weight_label = if cli.int4 {
        "weights+scratch (INT4-scaled)"
    } else if cli.fp8_runtime {
        "weights+scratch (FP8-scaled)"
    } else {
        "weights+scratch"
    };
    eprintln!(
        "[vram] estimated={:.2}GiB ({}={:.2}GiB + kv_cache={:.2}GiB for {}tok x B={}) available={:.1}GiB",
        gib(estimated_vram),
        weight_label,
        gib(quant_fixed_bytes),
        gib(kv_bytes),
        context_tokens,
        cli.batch_size,
        gib(total_vram),
    );
    if estimated_vram > total_vram {
        let reduce_hint = if cli.batch_size > 1 {
            "Reduce --context-size, --max-new-tokens, or --batch-size."
        } else {
            "Reduce --context-size or --max-new-tokens."
        };
        anyhow::bail!(
            "Insufficient VRAM for {context_tokens}-token context at batch_size={}: need ~{:.2}GiB, GPU has {:.1}GiB. {reduce_hint}",
            cli.batch_size,
            gib(estimated_vram),
            gib(total_vram),
        );
    }
    Ok(())
}

/// Compute the negative log-likelihood of `target_token` under the distribution
/// implied by `logits` (numerically stable log-softmax).
fn gemma4_target_nll(logits: &[f32], target_token: u32) -> Result<f64> {
    let target_idx = target_token as usize;
    let target_logit = logits.get(target_idx).ok_or_else(|| {
        anyhow!(
            "target token {target_token} outside logits len {}",
            logits.len()
        )
    })?;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp = logits
        .iter()
        .map(|&x| ((x as f64) - max_logit).exp())
        .sum::<f64>();
    Ok(max_logit + sum_exp.ln() - *target_logit as f64)
}

/// Score `prompt_ids` with teacher forcing: run GPU prefill over the first
/// token, then decode step-by-step while feeding the *true* next token rather
/// than the model's argmax.  Emits a human-readable `[teacher_forced]` line
/// to stderr and a machine-parseable `[teacher_forced_json]` line to stdout
/// with the same fields as the Qwen3.5 scorer.
///
/// Note: Gemma 4 does not currently have a `prefill_native_with_target_nll`
/// path (that method is specific to the Qwen3.5 `DecodeEngine`).  This
/// implementation therefore uses the decode-step-by-step path exclusively
/// (equivalent to Qwen3.5's `--teacher-forced-decode-step` branch).
/// All tokens from position 1 onward are scored, giving the same NLL/PPL
/// semantics; only `teacher_forced_scoring` is different in the JSON.
pub(crate) fn run_gemma4_teacher_forced(
    cli: &Cli,
    model_variant: &ModelVariant,
    backend: Backend,
    engine: &mut Gemma4Runtime,
    prompt_ids: &[u32],
) -> Result<()> {
    if prompt_ids.len() < 2 {
        bail!("--teacher-forced requires at least 2 prompt tokens");
    }
    if cli.validate || cli.gpu_validate {
        bail!("--teacher-forced does not currently support --validate or --gpu-validate");
    }

    let score_start = Instant::now();
    let prefill_start = Instant::now();

    // Gemma 4 uses the decode-step-by-step path: prefill on a single token,
    // then step through the remaining tokens scoring each from its logits.
    let mut logits = engine.prefill(&prompt_ids[..1])?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut total_nll = 0.0f64;
    let mut scored_tokens = 0usize;
    let mut decode_steps = 0usize;

    for target_idx in 1..prompt_ids.len() {
        total_nll += gemma4_target_nll(&logits, prompt_ids[target_idx])?;
        scored_tokens += 1;
        if target_idx + 1 < prompt_ids.len() {
            let input_token = prompt_ids[target_idx];
            let pos = target_idx;
            logits = engine.decode_step(input_token, pos)?;
            decode_steps += 1;
        }
    }

    if scored_tokens == 0 {
        bail!("--teacher-forced scored zero tokens");
    }

    let total_ms = score_start.elapsed().as_secs_f64() * 1000.0;
    let avg_nll = total_nll / scored_tokens as f64;
    let perplexity = avg_nll.exp();
    let bits_per_token = avg_nll / std::f64::consts::LN_2;
    let ms_per_token = total_ms / scored_tokens as f64;
    eprintln!(
        "[teacher_forced] tokens={} scored_tokens={} decode_steps={} nll={:.6} avg_nll={:.6} ppl={:.6} bpt={:.6} prefill_ms={:.1} total_ms={:.1} ms_per_token={:.2}",
        prompt_ids.len(),
        scored_tokens,
        decode_steps,
        total_nll,
        avg_nll,
        perplexity,
        bits_per_token,
        prefill_ms,
        total_ms,
        ms_per_token,
    );
    println!(
        "[teacher_forced_json] {}",
        serde_json::to_string(&serde_json::json!({
            "backend": backend_label(backend),
            "model": model_variant.to_string(),
            "mode": "dense",
            "teacher_forced_scoring": "decode_step_logits",
            "prompt_tokens": prompt_ids.len(),
            "scored_tokens": scored_tokens,
            "skipped_boundary_tokens": 0,
            "dense_prefix_len": 0,
            "decode_steps": decode_steps,
            "certified_decode_steps": 0,
            "total_nll": total_nll,
            "avg_nll": avg_nll,
            "perplexity": perplexity,
            "bits_per_token": bits_per_token,
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "ms_per_token": ms_per_token,
        }))?
    );
    Ok(())
}

/// See `qwen35_runtime::backend_label` — emit the actual backend in
/// `[teacher_forced_json]` rather than a stale literal.
fn backend_label(backend: Backend) -> &'static str {
    match backend {
        Backend::Hip => "hip",
        Backend::Cuda => "cuda",
        Backend::Metal => "metal",
    }
}
