use anyhow::Result;

use crate::bakes::ensure_gemma4_int4_bake;
use crate::registry::{self, Backend, Gemma4KernelParams, ModelVariant};
use crate::{gemma4_engine, gemma4_int4_engine, Cli};

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
