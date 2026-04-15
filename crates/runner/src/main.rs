mod decode_engine;
mod oracle;
mod registry;
mod validate;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use decode_engine::DecodeEngine;
use registry::{Backend, GpuArch, ModelVariant};

#[derive(Parser)]
#[command(name = "supersonic", about = "SuperSonic — optimized LLM inference")]
struct Cli {
    /// Model variant (e.g. "qwen3.5-0.8b")
    #[arg(long, default_value = "qwen3.5-0.8b")]
    model: String,

    /// Path to HuggingFace model directory (containing config.json + safetensors)
    #[arg(long)]
    model_dir: PathBuf,

    /// Text prompt (will be tokenized)
    #[arg(long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "8")]
    max_new_tokens: usize,

    /// Maximum context size in tokens (prompt + generated). Used for VRAM estimation.
    /// Defaults to prompt length + max_new_tokens if not specified.
    #[arg(long)]
    context_size: Option<usize>,

    /// HIP device ordinal
    #[arg(long, default_value = "0")]
    device: usize,

    /// Run PyTorch oracle and compare logits
    #[arg(long)]
    validate: bool,

    /// Oracle dtype (bf16 or fp32)
    #[arg(long, default_value = "bf16")]
    oracle_dtype: String,

    /// HuggingFace model ID (for oracle; defaults based on model variant)
    #[arg(long)]
    model_id: Option<String>,

    /// Skip baked format and load directly from safetensors (for debugging)
    #[arg(long)]
    no_bake: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ordinal = cli.device;

    // 1. Parse model variant
    let model_variant = ModelVariant::from_cli_str(&cli.model).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown model '{}'. Supported models: {}",
            cli.model,
            registry::supported_models_list().join(", ")
        )
    })?;

    // 2. Detect GPU
    let (arch_name, total_vram) = kernel_ffi::query_gpu_info(ordinal)
        .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
    let gpu_arch = GpuArch::from_rocm_name(&arch_name);
    eprintln!(
        "[gpu] device={ordinal} arch={arch_name} vram={:.1}GiB",
        total_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // 3. Registry lookup
    let backend = Backend::Hip;
    let entry = registry::lookup(&model_variant, &backend, &gpu_arch).ok_or_else(|| {
        let supported_archs = registry::supported_archs_for(&model_variant, &backend);
        anyhow::anyhow!(
            "No optimized kernel for model={model_variant} backend={backend} arch={gpu_arch}. \
             Supported GPU architectures for this model: [{}]",
            supported_archs.join(", ")
        )
    })?;

    let params = &entry.params;

    // Load config
    let config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
    let text_config = config.text_config;
    eprintln!(
        "[config] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    );

    // Tokenize
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());

    // 4. VRAM check (needs config + prompt length for KV cache estimation)
    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    let kv_per_token = text_config.kv_bytes_per_token(gpu_hal::ScalarType::BF16.size_in_bytes());
    let estimated_vram = entry.vram.estimate_total(context_tokens, kv_per_token);
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(entry.vram.fixed_bytes),
        gib(kv_per_token * context_tokens as u64),
        context_tokens,
        gib(total_vram),
    );
    if estimated_vram > total_vram {
        anyhow::bail!(
            "Insufficient VRAM for {context_tokens}-token context: \
             need ~{:.2}GiB (weights {:.2}GiB + KV cache {:.2}GiB), \
             GPU has {:.1}GiB. Reduce --context-size or --max-new-tokens.",
            gib(estimated_vram),
            gib(entry.vram.fixed_bytes),
            gib(kv_per_token * context_tokens as u64),
            gib(total_vram),
        );
    }

    // Load weights (baked format with auto-bake, or raw safetensors with --no-bake)
    let t0 = Instant::now();
    let weights = if cli.no_bake {
        eprintln!("[weights] loading from safetensors (--no-bake)...");
        qwen35::weights::Qwen35Weights::load(
            &cli.model_dir,
            &text_config,
            ordinal,
            params.weight_prefix,
        )
        .map_err(|e| anyhow::anyhow!("load weights: {e}"))?
    } else {
        let bake_dir = model_store::bake_dir(&cli.model_dir);
        if !model_store::version_ok(&bake_dir) {
            eprintln!("[bake] no baked package found — baking weights (one-time)...");
            let bake_start = Instant::now();
            let layer_is_full: Vec<bool> = (0..text_config.num_hidden_layers)
                .map(|i| text_config.is_full_attention(i))
                .collect();
            model_store::bake_qwen35(
                &cli.model_dir,
                params.weight_prefix,
                text_config.num_hidden_layers,
                &layer_is_full,
                &|msg| eprintln!("{msg}"),
            )
            .map_err(|e| anyhow::anyhow!("bake weights: {e}"))?;
            eprintln!(
                "[bake] done in {:.1}s",
                bake_start.elapsed().as_secs_f64()
            );
        } else {
            eprintln!("[weights] found baked package at {}", bake_dir.display());
        }
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow::anyhow!("open baked store: {e}"))?;
        qwen35::weights::Qwen35Weights::load_baked(
            &store,
            &text_config,
            ordinal,
            params.weight_prefix,
        )
        .map_err(|e| anyhow::anyhow!("load baked weights: {e}"))?
    };
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    // Create decode engine
    let mut engine = DecodeEngine::new(
        weights,
        ordinal,
        params.proj_buf_floats,
        params.attn_scratch_floats,
        params.kv_chunk_size,
    )?;

    // Run oracle for prefill (and optionally validation)
    let model_id = cli
        .model_id
        .clone()
        .unwrap_or_else(|| model_variant.hf_model_id().to_string());
    let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("oracle/run_oracle.py");

    let oracle_output = oracle::run_oracle(
        &oracle_script,
        &model_id,
        &prompt_ids,
        cli.max_new_tokens,
        &cli.oracle_dtype,
        true, // always emit state for prefill
    )?;

    // Load prefill state into GPU
    engine.load_prefill_state(&oracle_output)?;
    eprintln!("[engine] prefill state loaded to GPU");

    // Decode loop
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;

    // First token comes from prefill argmax
    let mut next_token = oracle_output.generated_token_ids[0];

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        let seqlen_offset = seqlen_start + step;
        let logits = engine.decode_step(next_token, seqlen_offset)?;

        // Validate against oracle if available
        if cli.validate && step < oracle_output.decode_logits.len() {
            let oracle_logits = &oracle_output.decode_logits[step];
            let delta = validate::max_abs_delta(&logits, oracle_logits);
            if delta > max_delta {
                max_delta = delta;
            }
            eprintln!(
                "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
            );
        }

        let sampled = DecodeEngine::greedy_sample(&logits);
        generated_ids.push(next_token);
        next_token = sampled;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    // Decode generated tokens to text
    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;

    println!("{text}");
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_tok={:.0} decode_max_delta={max_delta:.4}",
        prompt_ids.len(),
        generated_ids.len(),
        decode_ms / generated_ids.len() as f64,
    );

    Ok(())
}
