mod decode_engine;
mod gemma4_engine;
mod gemma4_int4_engine;
mod oracle;
mod prefill_engine;
mod registry;
mod validate;

/// Dispatcher over Gemma 4 runtime engines. BF16 runs through the persistent
/// megakernel; INT4 runs through the primitive chain backed by the GPTQ bake.
enum Gemma4Runtime {
    Bf16(gemma4_engine::Gemma4Engine),
    Int4(gemma4_int4_engine::Gemma4Int4Engine),
}

impl Gemma4Runtime {
    fn prefill(&mut self, prompt_token_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.prefill(prompt_token_ids),
            Self::Int4(e) => e.prefill(prompt_token_ids),
        }
    }

    fn decode_step(&mut self, token: u32, pos: usize) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.decode_step(token, pos),
            Self::Int4(e) => e.decode_step(token, pos),
        }
    }

    fn greedy_sample(logits: &[f32]) -> u32 {
        // Sampling is engine-independent; delegate to the BF16 engine's impl.
        gemma4_engine::Gemma4Engine::greedy_sample(logits)
    }
}

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use decode_engine::DecodeEngine;
use registry::{Backend, FamilyParams, GpuArch, ModelFamily, ModelVariant};

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

    /// Use oracle (Python) for prefill instead of native GPU prefill
    #[arg(long)]
    oracle_prefill: bool,

    /// Keep FP8 weights in native format on GPU for runtime dequantization.
    /// Halves weight VRAM (~8.8→4.8 GiB for 4B). Requires FP8 model weights.
    #[arg(long)]
    fp8_runtime: bool,

    /// Quantize weights to INT4 (4-bit) with group quantization for ~4x weight compression.
    /// Bakes BF16→INT4 on first run. Targets ~200 ms/tok on bandwidth-limited GPUs.
    #[arg(long)]
    int4: bool,

    /// Process prompt in chunks of this size (0 = no chunking, process entire prompt at once).
    /// Reduces activation VRAM for long prompts. Typical values: 128, 256, 512.
    #[arg(long, default_value = "0")]
    prefill_chunk_size: usize,

    /// Store KV cache in FP8 E4M3 with dynamic per-head scaling.
    /// Halves KV cache VRAM, nearly doubling max context length.
    #[arg(long)]
    kv_fp8: bool,

    /// Validate megakernel decode against GPU component-kernel oracle.
    /// Runs each decode step through both the megakernel and the prefill engine's
    /// component kernels, comparing logits. No external oracle needed.
    #[arg(long)]
    gpu_validate: bool,

    /// Batch size for decode (number of sequences decoded in parallel).
    /// Default 1. B=2 amortizes weight loading for ~1.8x throughput.
    /// Requires 4B kernel (2B/4B/9B models).
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// Run on an arch without a registry entry by reusing another arch's kernel.
    /// Pass the arch name whose kernel you want to reuse (e.g. "gfx1150"). Emits
    /// a loud warning — correctness is not guaranteed. Intended for archs that
    /// are binary-compatible (same wavefront size, similar CU/LDS) but haven't
    /// been explicitly tuned.
    #[arg(long)]
    allow_untested_gpu: Option<String>,

    /// Disable downloading pre-baked weights from GitHub releases when the
    /// local bake is missing. Prints the "run bake_int4.py" error instead of
    /// touching the network. Use on air-gapped machines.
    #[arg(long)]
    no_download: bool,

    /// Force downloading a pre-baked package even if a valid local bake exists.
    /// Useful for reproducing release content on a big-box producer.
    #[arg(long)]
    download_bake: bool,

    /// Override the GitHub release to fetch pre-baked weights from. Accepts a
    /// bare tag (e.g. "bakes-v1") or a full release URL. Also readable from
    /// the `SUPERSONIC_BAKE_RELEASE` env var.
    #[arg(long)]
    bake_release: Option<String>,
}

/// Resolve the release source from CLI flags and env var.
fn resolve_release_source(cli: &Cli) -> Result<model_store::fetch::ReleaseSource> {
    let raw = cli
        .bake_release
        .clone()
        .or_else(|| std::env::var("SUPERSONIC_BAKE_RELEASE").ok());
    match raw {
        Some(s) if !s.is_empty() => model_store::fetch::ReleaseSource::from_override(&s)
            .map_err(|e| anyhow::anyhow!("invalid --bake-release: {e}")),
        _ => Ok(model_store::fetch::ReleaseSource::default_for_format_version()),
    }
}

/// Pretty-print fetch progress. Throttles the per-byte update so the log
/// doesn't flood; prints on each ~5% crossing or on part transitions.
fn log_fetch_progress() -> impl Fn(model_store::fetch::FetchProgress) {
    use std::cell::Cell;
    let last_pct = Cell::new(i32::MIN);
    let last_part = Cell::new(u32::MAX);
    move |p| {
        use model_store::fetch::FetchProgress::*;
        match p {
            ResolvingIndex => eprintln!("[fetch] resolving release index..."),
            Downloading {
                part,
                total_parts,
                bytes_done,
                bytes_total,
            } => {
                let pct = if bytes_total > 0 {
                    (bytes_done * 100 / bytes_total) as i32
                } else {
                    0
                };
                if part != last_part.get() {
                    last_part.set(part);
                    last_pct.set(i32::MIN);
                    eprintln!(
                        "[fetch] downloading part {part}/{total_parts} ({} MiB)",
                        bytes_total / (1024 * 1024)
                    );
                }
                if pct / 5 != last_pct.get() / 5 {
                    last_pct.set(pct);
                    eprintln!(
                        "[fetch]   {pct}% ({} / {} MiB)",
                        bytes_done / (1024 * 1024),
                        bytes_total / (1024 * 1024)
                    );
                }
            }
            Verifying => eprintln!("[fetch] verifying SHA-256..."),
            Extracting => eprintln!("[fetch] extracting tarball..."),
            Done => eprintln!("[fetch] done"),
        }
    }
}

/// Attempt to download a bake. Returns `Ok(true)` if the download succeeded
/// and the bake is now present; `Ok(false)` if downloads are disabled by
/// `--no-download`; `Err` on any fetch error (with an instructive message).
fn try_download_bake(
    cli: &Cli,
    variant: model_store::fetch::BakeVariant,
    model_cli_name: &str,
    target: &std::path::Path,
) -> Result<bool> {
    if cli.no_download {
        return Ok(false);
    }
    let source = resolve_release_source(cli)?;
    eprintln!(
        "[fetch] downloading {model_cli_name} {variant} from {}/{}",
        source.repo_slug, source.tag
    );
    let progress = log_fetch_progress();
    let req = model_store::fetch::FetchRequest {
        source: &source,
        model_cli_name,
        variant,
        target_bake_dir: target,
        progress: &progress,
    };
    model_store::fetch::fetch_bake(req)
        .map_err(|e| anyhow::anyhow!("fetch bake: {e}"))?;
    Ok(true)
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
    let entry = match registry::lookup(&model_variant, &backend, &gpu_arch) {
        Some(e) => e,
        None => {
            if let Some(override_arch) = cli.allow_untested_gpu.as_deref() {
                let reuse_arch = GpuArch::from_rocm_name(override_arch);
                let e = registry::lookup(&model_variant, &backend, &reuse_arch).ok_or_else(|| {
                    let supported_archs = registry::supported_archs_for(&model_variant, &backend);
                    anyhow::anyhow!(
                        "--allow-untested-gpu={override_arch}: no registry entry for \
                         model={model_variant} backend={backend} arch={reuse_arch}. \
                         Pass one of: [{}]",
                        supported_archs.join(", ")
                    )
                })?;
                eprintln!(
                    "[gpu] WARNING: detected arch={gpu_arch} has no registry entry; \
                     reusing {reuse_arch} kernel as requested by --allow-untested-gpu. \
                     Correctness is not guaranteed."
                );
                e
            } else {
                let supported_archs = registry::supported_archs_for(&model_variant, &backend);
                anyhow::bail!(
                    "No optimized kernel for model={model_variant} backend={backend} arch={gpu_arch}. \
                     Supported GPU architectures for this model: [{}]. \
                     To force-reuse another arch's kernel, pass --allow-untested-gpu=<arch>.",
                    supported_archs.join(", ")
                );
            }
        }
    };

    // Dispatch on family. Gemma 4 has its own load/prefill/decode pipeline via
    // the persistent megakernel; return early before the Qwen-shaped path.
    if model_variant.family() == ModelFamily::Gemma4 {
        return run_gemma4(&cli, &model_variant, entry, ordinal, total_vram);
    }

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => p,
        FamilyParams::Gemma4(_) => unreachable!("gemma4 handled above"),
    };

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
    let kv_dtype_bytes = if cli.kv_fp8 { 1usize } else { gpu_hal::ScalarType::BF16.size_in_bytes() };
    let kv_per_token = text_config.kv_bytes_per_token(kv_dtype_bytes);
    // FP8 runtime dequant halves weight VRAM; INT4 quarters it.
    let effective_fixed = if cli.int4 {
        // INT4: weights ~= fixed * 0.9, scratch ~= fixed * 0.1
        // INT4 weights = weights / 4 + ~5% scale/zero overhead
        // total ≈ fixed * 0.9 * 0.3 + fixed * 0.1 = fixed * 0.37
        (entry.vram.fixed_bytes as f64 * 0.37) as u64
    } else if cli.fp8_runtime {
        // FP8: weights / 2
        (entry.vram.fixed_bytes as f64 * 0.55) as u64
    } else {
        entry.vram.fixed_bytes
    };
    let estimated_vram = {
        let kv_bytes = kv_per_token * context_tokens as u64;
        ((effective_fixed + kv_bytes) as f64 * entry.vram.overhead_factor) as u64
    };
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(effective_fixed),
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
            gib(effective_fixed),
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
        // Select bake directory + variant: INT4 > FP8-native > BF16 (priority order)
        let variant = if cli.int4 {
            model_store::fetch::BakeVariant::Int4Gptq
        } else if cli.fp8_runtime {
            model_store::fetch::BakeVariant::Fp8Native
        } else {
            model_store::fetch::BakeVariant::Bf16
        };
        let bake_dir = variant.bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;

        if cli.download_bake || !model_store::version_ok(&bake_dir) {
            // Choose provisioning strategy: for BF16 we have a local baker;
            // for INT4/FP8-native (Qwen) the only way to get a bake is to
            // download or run the Python baker manually.
            let local_bake_ok = matches!(variant, model_store::fetch::BakeVariant::Bf16 | model_store::fetch::BakeVariant::Fp8Native);
            let canonical_model = model_variant.to_string();
            let downloaded = try_download_bake(&cli, variant, &canonical_model, &bake_dir);
            match downloaded {
                Ok(true) => {
                    eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                }
                Ok(false) => {
                    if local_bake_ok {
                        // --no-download: fall through to local bake below.
                    } else {
                        anyhow::bail!(
                            "no {variant} bake at {} and --no-download set.\n\
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                            bake_dir.display(),
                            cli.model_dir.display(),
                        );
                    }
                }
                Err(e) => {
                    if local_bake_ok {
                        eprintln!("[fetch] {e}; falling back to local bake");
                    } else {
                        anyhow::bail!(
                            "could not obtain {variant} bake: {e}\n\n\
                             INT4 baking requires a GPTQ calibration pass in Python. \
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}\n\n\
                             then `python oracle/upload_bake.py --model {} --int4 --model-dir {}` to publish.",
                            cli.model_dir.display(),
                            cli.model,
                            cli.model_dir.display(),
                        );
                    }
                }
            }

            if !model_store::version_ok(&bake_dir) && local_bake_ok {
                let mode_str = if cli.fp8_runtime { " (FP8 native)" } else { "" };
                eprintln!(
                    "[bake] no baked package found — baking weights{mode_str} (one-time)..."
                );
                let bake_start = Instant::now();
                let layer_is_full: Vec<bool> = (0..text_config.num_hidden_layers)
                    .map(|i| text_config.is_full_attention(i))
                    .collect();
                model_store::bake_qwen35(
                    &cli.model_dir,
                    params.weight_prefix,
                    text_config.num_hidden_layers,
                    &layer_is_full,
                    cli.fp8_runtime,
                    &|msg| eprintln!("{msg}"),
                )
                .map_err(|e| anyhow::anyhow!("bake weights: {e}"))?;
                eprintln!("[bake] done in {:.1}s", bake_start.elapsed().as_secs_f64());
            }
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
    if weights.is_fp8 {
        eprintln!(
            "[weights] FP8 runtime dequant active (block_size={})",
            weights.fp8_block_size
        );
    }
    if weights.is_int4 {
        eprintln!(
            "[weights] INT4 runtime dequant active (group_size={})",
            weights.int4_group_size
        );
    }
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    // Validate batch_size
    if cli.batch_size > 1 && !params.use_4b_kernel {
        anyhow::bail!("--batch-size > 1 requires 4B kernel (2B/4B/9B models)");
    }
    if cli.batch_size < 1 || cli.batch_size > kernel_ffi::MAX_BATCH_SIZE {
        anyhow::bail!("--batch-size must be 1..{}", kernel_ffi::MAX_BATCH_SIZE);
    }

    // Create decode engine
    let mut engine = DecodeEngine::new(
        weights,
        ordinal,
        params.proj_buf_floats,
        params.attn_scratch_floats,
        params.kv_chunk_size,
        params.use_4b_kernel,
        cli.prefill_chunk_size,
        cli.kv_fp8,
        cli.batch_size,
    )?;

    // When using FP8 runtime weights, tell the oracle to use the same FP8 weights
    // (dequanted to BF16) so we compare apples-to-apples.
    let fp8_oracle_dir = if cli.fp8_runtime {
        Some(cli.model_dir.clone())
    } else {
        None
    };

    // Run prefill (native GPU or oracle)
    let prefill_start = Instant::now();
    let (prefill_logits, mut next_token) = if cli.oracle_prefill {
        let model_id = cli
            .model_id
            .clone()
            .unwrap_or_else(|| model_variant.hf_model_id().to_string());
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("oracle/run_oracle.py");
        let output = oracle::run_oracle(
            &oracle_script, &model_id, &prompt_ids, cli.max_new_tokens,
            &cli.oracle_dtype, true,
            fp8_oracle_dir.as_deref(),
        )?;
        engine.load_prefill_state(&output)?;
        let first = output.generated_token_ids[0];
        eprintln!("[prefill] oracle prefill done in {:.0}ms", prefill_start.elapsed().as_millis());
        (output.prefill_logits, first)
    } else {
        let logits = engine.prefill_native(&prompt_ids)?;
        let first = DecodeEngine::greedy_sample(&logits);
        eprintln!("[prefill] native GPU prefill done in {:.0}ms", prefill_start.elapsed().as_millis());
        (logits, first)
    };

    // Optionally run oracle for validation
    let oracle_output = if cli.validate {
        let model_id = cli
            .model_id
            .clone()
            .unwrap_or_else(|| model_variant.hf_model_id().to_string());
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("oracle/run_oracle.py");

        let output = oracle::run_oracle(
            &oracle_script,
            &model_id,
            &prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            false, // only need logits for comparison
            fp8_oracle_dir.as_deref(),
        )?;

        // Compare prefill logits
        let prefill_delta = validate::max_abs_delta(&prefill_logits, &output.prefill_logits);
        eprintln!("[validate] prefill logit delta={prefill_delta:.4}");

        // Check if oracle and native agree on first token
        let oracle_first = output.generated_token_ids[0];
        if oracle_first != next_token {
            eprintln!(
                "[validate] WARNING: prefill token mismatch! native={next_token} oracle={oracle_first}"
            );
        }

        Some(output)
    } else {
        None
    };

    // Replicate prefill state to batch items if batch_size > 1
    if cli.batch_size > 1 {
        eprintln!("[batch] replicating prefill state to {} sequences", cli.batch_size);
        engine.replicate_state_to_batch()?;
    }

    // GPU oracle: clone model state for independent component-kernel decode
    let mut gpu_oracle_state = if cli.gpu_validate && cli.batch_size == 1 {
        eprintln!("[gpu-validate] cloning model state for GPU oracle...");
        Some(engine.clone_state()?)
    } else {
        if cli.gpu_validate && cli.batch_size > 1 {
            eprintln!("[gpu-validate] GPU oracle disabled for batch_size > 1");
        }
        None
    };

    // Decode loop
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;
    let mut gpu_max_delta = 0.0f32;
    let eos_ids = text_config.eos_token_ids();

    // For batched decode, track per-sequence tokens
    let mut batch_next_tokens: Vec<u32> = vec![next_token; cli.batch_size];

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        // Stop on EOS token (sequence 0 drives the output)
        if eos_ids.contains(&next_token) {
            break;
        }

        let seqlen_offset = seqlen_start + step;

        if cli.batch_size > 1 {
            // Batched decode
            let batch_logits = engine.decode_step_batch(&batch_next_tokens, seqlen_offset)?;

            // Use sequence 0's logits for output and validation
            let logits = &batch_logits[0];

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(logits, oracle_logits);
                    if delta > max_delta { max_delta = delta; }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token} batch_size={}",
                        cli.batch_size
                    );
                }
            }

            // Sample next tokens for all sequences
            for (bi, seq_logits) in batch_logits.iter().enumerate() {
                batch_next_tokens[bi] = DecodeEngine::greedy_sample(seq_logits);
            }

            generated_ids.push(next_token);
            next_token = batch_next_tokens[0];
        } else {
            // Single-sequence decode (original path)
            let logits = engine.decode_step(next_token, seqlen_offset)?;

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(&logits, oracle_logits);
                    if delta > max_delta { max_delta = delta; }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
                    );
                }
            }

            if let Some(ref mut oracle_state) = gpu_oracle_state {
                let gpu_logits = engine.decode_step_on_state(
                    oracle_state, next_token, seqlen_offset,
                )?;
                let delta = validate::max_abs_delta(&logits, &gpu_logits);
                let gpu_token = DecodeEngine::greedy_sample(&gpu_logits);
                let token_match = if gpu_token == DecodeEngine::greedy_sample(&logits) { "" } else { " MISMATCH" };
                if delta > gpu_max_delta { gpu_max_delta = delta; }
                eprintln!(
                    "[gpu-validate] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token} gpu_token={gpu_token}{token_match}"
                );
            }

            let sampled = DecodeEngine::greedy_sample(&logits);
            generated_ids.push(next_token);
            next_token = sampled;
        }
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
    println!(
        "[tokens] {}",
        generated_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_tok={:.0} decode_max_delta={max_delta:.4} gpu_oracle_max_delta={gpu_max_delta:.4} batch_size={}",
        prompt_ids.len(),
        generated_ids.len(),
        if generated_ids.is_empty() { 0.0 } else { decode_ms / generated_ids.len() as f64 },
        cli.batch_size,
    );

    Ok(())
}

/// Gemma 4 end-to-end decode: tokenize → native prefill → megakernel decode
/// → detokenize. Rejects unsupported flags (INT4, FP8, KV-FP8, batch>1) with
/// clear errors — the Qwen-family flags do not plumb through yet.
fn run_gemma4(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &registry::RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    let params = match &entry.params {
        FamilyParams::Gemma4(p) => p,
        FamilyParams::Qwen35(_) => unreachable!("dispatch filtered to Gemma4"),
    };

    if cli.fp8_runtime || cli.kv_fp8 {
        anyhow::bail!(
            "Gemma 4 does not yet support --fp8-runtime / --kv-fp8"
        );
    }
    if cli.batch_size != 1 {
        anyhow::bail!("Gemma 4 does not yet support --batch-size > 1");
    }
    if cli.oracle_prefill || cli.gpu_validate {
        anyhow::bail!(
            "Gemma 4 does not yet support --oracle-prefill / --gpu-validate"
        );
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

    // Load config for a config summary + VRAM / context bookkeeping.
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

    // Tokenize
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
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

    // VRAM estimate — Gemma 4 BF16 KV cache: owning layers only (shared layers
    // alias the source), each layer's slot is `num_kv_heads * head_dim * 2`
    // elements (×2 for K + V tensors). Multiply by the BF16 byte size to get
    // actual bytes.
    const BF16_BYTES: u64 = 2;
    let mut kv_per_token: u64 = 0;
    for l in 0..t.num_hidden_layers {
        if t.kv_source_layer(l).is_none() {
            let kind = t
                .attn_kind(l)
                .ok_or_else(|| anyhow::anyhow!("layer {l}: no attention kind"))?;
            let hd = t.head_dim_for(kind);
            kv_per_token += (t.num_key_value_heads * hd * 2) as u64;
        }
    }
    let kv_bytes = kv_per_token * context_tokens as u64 * BF16_BYTES;
    let estimated_vram =
        ((entry.vram.fixed_bytes + kv_bytes) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights+scratch={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(entry.vram.fixed_bytes),
        gib(kv_bytes),
        context_tokens,
        gib(total_vram),
    );
    if estimated_vram > total_vram {
        anyhow::bail!(
            "Insufficient VRAM for {context_tokens}-token context: need ~{:.2}GiB, \
             GPU has {:.1}GiB. Reduce --context-size or --max-new-tokens.",
            gib(estimated_vram),
            gib(total_vram),
        );
    }

    // Load weights + build engine. INT4 uses a different loader path (reads
    // the GPTQ bake under .supersonic/v1-int4-gptq/) and runs the primitive
    // chain for decode instead of the persistent megakernel.
    let t0 = Instant::now();
    let mut engine: Gemma4Runtime = if cli.int4 {
        let target = gemma4_int4_engine::int4_bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
        if cli.download_bake || !gemma4_int4_engine::int4_bake_ok(&cli.model_dir) {
            let canonical_model = model_variant.to_string();
            match try_download_bake(
                cli,
                model_store::fetch::BakeVariant::Int4Gptq,
                &canonical_model,
                &target,
            ) {
                Ok(true) => eprintln!("[fetch] installed Gemma 4 INT4 bake at {}", target.display()),
                Ok(false) => {
                    anyhow::bail!(
                        "No INT4 bake at {} and --no-download set.\n\
                         Run on a bigger machine:\n  \
                         python oracle/bake_int4_gemma4.py --model-dir {}",
                        target.display(),
                        cli.model_dir.display(),
                    );
                }
                Err(e) => {
                    anyhow::bail!(
                        "could not obtain Gemma 4 INT4 bake: {e}\n\n\
                         Run on a bigger machine:\n  \
                         python oracle/bake_int4_gemma4.py --model-dir {}\n\
                         then `python oracle/upload_bake.py --model {} --int4 --model-dir {}` to publish.",
                        cli.model_dir.display(),
                        cli.model,
                        cli.model_dir.display(),
                    );
                }
            }
        }
        eprintln!("[gemma4] loading INT4 GPTQ bake (primitive-chain decode)");
        Gemma4Runtime::Int4(gemma4_int4_engine::Gemma4Int4Engine::load(
            &cli.model_dir,
            params.weight_prefix,
            context_tokens,
            ordinal,
        )?)
    } else {
        Gemma4Runtime::Bf16(gemma4_engine::Gemma4Engine::load(
            &cli.model_dir,
            params.weight_prefix,
            context_tokens,
            ordinal,
        )?)
    };
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    // Optional oracle (runs BEFORE prefill so we can compare every step).
    // The Gemma 4 oracle does its own CPU forward pass — expensive (~seconds to
    // minutes depending on prompt length) but fully self-consistent with HF.
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
        // Cross-check tokenization — if the Python tokenizer disagrees with
        // ours, logit comparison is meaningless, so fail loudly.
        if let Some(ref oracle_ids) = oracle.prompt_token_ids {
            if oracle_ids != &prompt_ids {
                anyhow::bail!(
                    "tokenizer mismatch between Rust and Python oracle: \
                     rust={prompt_ids:?} oracle={oracle_ids:?}"
                );
            }
        }
        Some(oracle)
    } else {
        None
    };

    // Native prefill.
    let prefill_start = Instant::now();
    let prefill_logits = engine.prefill(&prompt_ids)?;
    let mut next_token = Gemma4Runtime::greedy_sample(&prefill_logits);
    eprintln!(
        "[prefill] native GPU prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );

    // Prefill delta vs oracle (if --validate).
    if let Some(ref oracle) = oracle_output {
        let prefill_delta = validate::max_abs_delta(&prefill_logits, &oracle.prefill_logits);
        eprintln!("[validate] prefill logit delta={prefill_delta:.4}");
        // Oracle may return an empty generation when --max-new-tokens 0 was
        // requested; only compare the first token when the oracle actually
        // produced one.
        if let Some(&oracle_first) = oracle.generated_token_ids.first() {
            if oracle_first != next_token {
                eprintln!(
                    "[validate] WARNING: prefill token mismatch! native={next_token} oracle={oracle_first}"
                );
            }
        }
    }

    // Decode loop.
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let eos_ids = t.eos_token_ids();
    let mut max_delta = 0.0f32;
    let mut token_mismatches = 0usize;

    let decode_start = Instant::now();
    for step in 0..cli.max_new_tokens {
        if eos_ids.contains(&next_token) {
            break;
        }
        let pos = seqlen_start + step;
        let logits = engine.decode_step(next_token, pos)?;

        if let Some(ref oracle) = oracle_output {
            if step < oracle.decode_logits.len() {
                let oracle_logits = &oracle.decode_logits[step];
                let delta = validate::max_abs_delta(&logits, oracle_logits);
                if delta > max_delta {
                    max_delta = delta;
                }
                // Step k's input token is the one we just sampled at the end
                // of step k-1 (or prefill for step 0). The oracle picks the
                // same input greedy path, so generated_token_ids[step+1] is
                // the oracle's next-output for this step's logits.
                let oracle_next = if step + 1 < oracle.generated_token_ids.len() {
                    Some(oracle.generated_token_ids[step + 1])
                } else {
                    None
                };
                let rust_next = gemma4_engine::Gemma4Engine::greedy_sample(&logits);
                let mismatch_tag = match oracle_next {
                    Some(ot) if ot != rust_next => {
                        token_mismatches += 1;
                        format!(" MISMATCH (oracle_next={ot})")
                    }
                    _ => String::new(),
                };
                eprintln!(
                    "[validate] step={step} pos={pos} delta={delta:.4} input_tok={next_token} rust_next={rust_next}{mismatch_tag}"
                );
            }
        }

        let sampled = gemma4_engine::Gemma4Engine::greedy_sample(&logits);
        generated_ids.push(next_token);
        next_token = sampled;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;

    println!("{text}");
    println!(
        "[tokens] {}",
        generated_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_tok={:.0}{}",
        prompt_ids.len(),
        generated_ids.len(),
        if generated_ids.is_empty() {
            0.0
        } else {
            decode_ms / generated_ids.len() as f64
        },
        if oracle_output.is_some() {
            format!(" decode_max_delta={max_delta:.4} token_mismatches={token_mismatches}")
        } else {
            String::new()
        },
    );
    Ok(())
}
