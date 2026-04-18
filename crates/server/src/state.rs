//! Server startup: detect GPU, look up registry entry, load weights, build
//! the [`InferenceSession`]. Boiled-down version of the full `supersonic`
//! CLI flow in `crates/runner/src/main.rs` — the server skips the oracle,
//! tracing, and fallback paths the CLI exposes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use runner::decode_engine::DecodeEngine;
use runner::gemma4_engine::Gemma4Engine;
use runner::gemma4_int4_engine::{self, Gemma4Int4Engine};
use runner::registry::{self, FamilyParams, GpuArch, ModelFamily, ModelVariant};

use gpu_hal::Backend;

use crate::chat_template::ChatTemplate;
use crate::session::InferenceSession;

/// Per-process state shared across every HTTP request. Everything here is
/// built once at startup.
pub struct ServerState {
    pub model_id: String,
    pub model_family: ModelFamily,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<Arc<ChatTemplate>>,
    pub session: Arc<Mutex<InferenceSession>>,
    pub eos_ids: Vec<u32>,
    pub max_context: usize,
    pub api_key: Option<String>,
}

/// Arguments captured from the CLI and forwarded into the loader.
pub struct LoaderConfig {
    pub model: String,
    pub model_dir: PathBuf,
    pub backend: String,
    pub device: usize,
    pub max_context: usize,
    pub int4: bool,
    pub fp8_runtime: bool,
    pub kv_fp8: bool,
    pub api_key: Option<String>,
    /// Disable automatic bake download from the GitHub release. Air-gapped
    /// or reproducibility-focused deploys should set this.
    pub no_download: bool,
}

pub fn build(cfg: LoaderConfig) -> Result<ServerState> {
    /* ---- backend + GPU detection ---- */
    let backend = resolve_backend(&cfg.backend, cfg.device)?;
    gpu_hal::set_backend(backend);

    let variant = ModelVariant::from_cli_str(&cfg.model).ok_or_else(|| {
        anyhow!(
            "unknown --model '{}' (supported: {})",
            cfg.model,
            registry::supported_models_list().join(", ")
        )
    })?;

    let (arch_name, total_vram, warp_size) = match backend {
        Backend::Hip => {
            let (a, v) = kernel_ffi::query_gpu_info(cfg.device)
                .map_err(|e| anyhow!("GPU query failed for device {}: {}", cfg.device, e))?;
            (a, v, 32)
        }
        Backend::Cuda => {
            let info = gpu_hal::query_device_info(backend, cfg.device)
                .map_err(|e| anyhow!("GPU query failed for device {}: {}", cfg.device, e))?;
            (info.arch_name, info.total_vram_bytes, info.warp_size)
        }
    };
    let gpu_arch = GpuArch::from_backend_name(&backend, &arch_name);
    tracing::info!(
        backend = %backend,
        device = cfg.device,
        arch = %arch_name,
        warp = warp_size,
        vram_gib = total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
        "GPU detected"
    );

    let entry = registry::lookup(&variant, &backend, &gpu_arch).ok_or_else(|| {
        let archs = registry::supported_archs_for(&variant, &backend);
        anyhow!(
            "no registry entry for model={} backend={} arch={}; supported archs: [{}]",
            variant,
            backend,
            gpu_arch,
            archs.join(", ")
        )
    })?;

    /* ---- HF metadata preflight ---- */
    ensure_hf_metadata_present(&cfg, &variant)?;

    /* ---- tokenizer + chat template ---- */
    let tokenizer_path = cfg.model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!(
            "missing {} — cannot build tokenizer",
            tokenizer_path.display()
        );
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let chat_template = ChatTemplate::try_load(&cfg.model_dir)?;
    if chat_template.is_none() {
        tracing::warn!(
            "no chat_template in tokenizer_config.json — /v1/chat/completions will reject \
             requests; /v1/completions still works"
        );
    }

    /* ---- engine construction ---- */
    let max_context = cfg.max_context.max(8);
    let (session, eos_ids) = match variant.family() {
        ModelFamily::Qwen35 => build_qwen(&cfg, entry, max_context)?,
        ModelFamily::Gemma4 => build_gemma4(&cfg, entry, max_context)?,
    };

    tracing::info!(
        model = %variant,
        family = %variant.family(),
        max_context,
        "server state ready"
    );

    Ok(ServerState {
        model_id: variant.to_string(),
        model_family: variant.family(),
        tokenizer: Arc::new(tokenizer),
        chat_template,
        session: Arc::new(Mutex::new(session)),
        eos_ids,
        max_context,
        api_key: cfg.api_key,
    })
}

/// When `--model-dir` has no `config.json`, trigger a bake download so the
/// tarball can populate HF metadata (config.json, tokenizer.json, etc.)
/// before we attempt to read them. Mirrors the CLI's preflight.
fn ensure_hf_metadata_present(cfg: &LoaderConfig, variant: &ModelVariant) -> Result<()> {
    if cfg.no_download {
        return Ok(());
    }
    if cfg.model_dir.join("config.json").exists() {
        return Ok(());
    }
    let bake_variant = if cfg.int4 {
        model_store::fetch::BakeVariant::Int4Gptq
    } else if cfg.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        model_store::fetch::BakeVariant::Bf16
    };
    let bake_dir = match variant.family() {
        ModelFamily::Gemma4 if cfg.int4 => gemma4_int4_engine::int4_bake_dir(&cfg.model_dir),
        _ => bake_variant.bake_dir(&cfg.model_dir),
    };
    let _lock = model_store::BakeLock::acquire(&cfg.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    if cfg.model_dir.join("config.json").exists() {
        return Ok(());
    }
    tracing::info!(
        "--model-dir has no config.json; fetching {} bake to populate HF metadata + weights",
        bake_variant
    );
    let _ = try_download_bake(cfg, bake_variant, &bake_dir)?;
    Ok(())
}

/// Attempt to fetch the requested bake from the GitHub `bakes-v{FORMAT_VERSION}`
/// release. Returns `Ok(true)` on success, `Ok(false)` when download is
/// disabled via `--no-download`, `Err(_)` when the fetch itself failed.
fn try_download_bake(
    cfg: &LoaderConfig,
    variant: model_store::fetch::BakeVariant,
    target_bake_dir: &std::path::Path,
) -> Result<bool> {
    if cfg.no_download {
        return Ok(false);
    }
    let source = model_store::fetch::ReleaseSource::default_for_format_version();
    tracing::info!(
        "downloading {} bake for {} from bakes-v{} release...",
        variant,
        cfg.model,
        model_store::manifest::FORMAT_VERSION
    );
    let req = model_store::fetch::FetchRequest {
        source: &source,
        model_cli_name: &cfg.model,
        variant,
        target_bake_dir,
        target_model_dir: &cfg.model_dir,
        progress: &fetch_progress_logger(),
    };
    model_store::fetch::fetch_bake(req)
        .map_err(|e| anyhow!("fetch {} bake: {}", variant, e))?;
    Ok(true)
}

fn fetch_progress_logger() -> impl Fn(model_store::fetch::FetchProgress) {
    use std::cell::Cell;
    let last_pct = Cell::new(i32::MIN);
    let last_part = Cell::new(u32::MAX);
    move |p| {
        use model_store::fetch::FetchProgress::*;
        match p {
            ResolvingIndex => tracing::info!("[fetch] resolving release index..."),
            Downloading { part, total_parts, bytes_done, bytes_total } => {
                let pct = if bytes_total > 0 {
                    (bytes_done * 100 / bytes_total) as i32
                } else {
                    0
                };
                if part != last_part.get() {
                    last_part.set(part);
                    last_pct.set(i32::MIN);
                    tracing::info!(
                        "[fetch] part {}/{} — {} MiB",
                        part + 1,
                        total_parts,
                        bytes_total / (1024 * 1024)
                    );
                }
                if pct >= last_pct.get() + 10 {
                    last_pct.set(pct);
                    tracing::info!("[fetch]   {}% ({} / {} MiB)",
                        pct,
                        bytes_done / (1024 * 1024),
                        bytes_total / (1024 * 1024));
                }
            }
            Verifying => tracing::info!("[fetch] verifying checksums..."),
            Extracting => tracing::info!("[fetch] extracting archive..."),
            Done => tracing::info!("[fetch] done"),
        }
    }
}

fn resolve_backend(choice: &str, ordinal: usize) -> Result<Backend> {
    match choice.to_ascii_lowercase().as_str() {
        "hip" => Ok(Backend::Hip),
        "cuda" => Ok(Backend::Cuda),
        "auto" | "" => {
            // Prefer HIP when a HIP device is reachable; fall back to CUDA.
            if kernel_ffi::query_gpu_info(ordinal).is_ok() {
                Ok(Backend::Hip)
            } else {
                Ok(Backend::Cuda)
            }
        }
        other => bail!("unknown --backend '{other}' (auto | hip | cuda)"),
    }
}

fn build_qwen(
    cfg: &LoaderConfig,
    entry: &'static registry::RegistryEntry,
    context_tokens: usize,
) -> Result<(InferenceSession, Vec<u32>)> {
    let mut params = match &entry.params {
        FamilyParams::Qwen35(p) => *p,
        FamilyParams::Gemma4(_) => unreachable!("caller filtered to Qwen"),
    };

    // INT4 decode lives in the 4B kernel; force-route 0.8B through it.
    if cfg.int4 && !params.use_4b_kernel && matches!(entry.backend, Backend::Hip) {
        params.use_4b_kernel = true;
    }
    let params = &params;

    let config = qwen35::config::load_config(&cfg.model_dir)
        .map_err(|e| anyhow!("loading config.json: {e}"))?;
    let text_config = config.text_config;
    let eos_ids = text_config.eos_token_ids();

    // Prefer the baked format; auto-bake BF16/FP8 if missing, fail with a
    // clear message for INT4 (calibration must happen offline).
    let t0 = Instant::now();
    let variant_bake = if cfg.int4 {
        model_store::fetch::BakeVariant::Int4Gptq
    } else if cfg.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        model_store::fetch::BakeVariant::Bf16
    };
    let bake_dir = variant_bake.bake_dir(&cfg.model_dir);
    let _lock = model_store::BakeLock::acquire(&cfg.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;

    if !model_store::version_ok(&bake_dir) {
        let local_bake_ok = matches!(
            variant_bake,
            model_store::fetch::BakeVariant::Bf16 | model_store::fetch::BakeVariant::Fp8Native
        );
        let downloaded = try_download_bake(cfg, variant_bake, &bake_dir)?;
        if !downloaded && !local_bake_ok {
            bail!(
                "no {variant_bake} bake at {} and download unavailable. INT4 calibration \
                 must happen offline — run `python oracle/bake_int4.py --model-dir {}` on a \
                 machine with spare RAM or rerun without --no-download to fetch from the \
                 GitHub bakes-v1 release.",
                bake_dir.display(),
                cfg.model_dir.display(),
            );
        }
        if !model_store::version_ok(&bake_dir) && local_bake_ok {
            tracing::info!("baking Qwen3.5 {} weights (one-time)...", variant_bake);
            let bake_start = Instant::now();
            let layer_is_full: Vec<bool> = (0..text_config.num_hidden_layers)
                .map(|i| text_config.is_full_attention(i))
                .collect();
            model_store::bake_qwen35(
                &cfg.model_dir,
                params.weight_prefix,
                text_config.num_hidden_layers,
                &layer_is_full,
                cfg.fp8_runtime,
                &|m| tracing::info!("{m}"),
            )
            .map_err(|e| anyhow!("bake weights: {e}"))?;
            tracing::info!("bake done in {:.1}s", bake_start.elapsed().as_secs_f64());
        }
    }

    let store = model_store::BakedStore::open(&bake_dir)
        .map_err(|e| anyhow!("open baked store: {e}"))?;
    let weights = qwen35::weights::Qwen35Weights::load_baked(
        &store,
        &text_config,
        cfg.device,
        params.weight_prefix,
    )
    .map_err(|e| anyhow!("load baked weights: {e}"))?;
    tracing::info!("weights loaded in {:.0}ms", t0.elapsed().as_millis());

    let attn_scratch_floats = params.attn_scratch_floats.max(
        qwen35::scratch::required_attn_scratch_floats(
            text_config.num_attention_heads,
            text_config.head_dim,
            context_tokens,
            params.kv_chunk_size,
        ),
    );

    let engine = DecodeEngine::new(
        weights,
        cfg.device,
        params.proj_buf_floats,
        attn_scratch_floats,
        params.kv_chunk_size,
        params.use_4b_kernel,
        0, // prefill_chunk_size — 0 = no chunking; server handles one prompt at a time
        cfg.kv_fp8,
        1, // batch_size — serial model for v1
    )
    .with_context(|| "build Qwen3.5 DecodeEngine")?;

    Ok((InferenceSession::Qwen(engine), eos_ids))
}

fn build_gemma4(
    cfg: &LoaderConfig,
    entry: &'static registry::RegistryEntry,
    context_tokens: usize,
) -> Result<(InferenceSession, Vec<u32>)> {
    if cfg.fp8_runtime || cfg.kv_fp8 {
        bail!("Gemma 4 does not yet support --fp8-runtime / --kv-fp8");
    }
    let params = match &entry.params {
        FamilyParams::Gemma4(p) => p,
        FamilyParams::Qwen35(_) => unreachable!("caller filtered to Gemma 4"),
    };

    let g_cfg = gemma4::config::load_config(&cfg.model_dir)
        .map_err(|e| anyhow!("loading Gemma 4 config.json: {e}"))?;
    let eos_ids = g_cfg.text_config.eos_token_ids();

    let t0 = Instant::now();
    let session = if cfg.int4 {
        if !gemma4_int4_engine::int4_bake_ok(&cfg.model_dir) {
            let target = gemma4_int4_engine::int4_bake_dir(&cfg.model_dir);
            let downloaded = try_download_bake(
                cfg,
                model_store::fetch::BakeVariant::Int4Gptq,
                &target,
            )?;
            if !downloaded || !gemma4_int4_engine::int4_bake_ok(&cfg.model_dir) {
                bail!(
                    "no Gemma 4 INT4 bake at {} and download unavailable. Run \
                     `python oracle/bake_int4_gemma4.py --model-dir {}` on a bigger machine \
                     or rerun without --no-download to fetch from the GitHub bakes-v1 release.",
                    target.display(),
                    cfg.model_dir.display(),
                );
            }
        }
        let engine = Gemma4Int4Engine::load_with_batch(
            &cfg.model_dir,
            params.weight_prefix,
            context_tokens,
            cfg.device,
            1, // batch_size — serial model for v1
        )?;
        InferenceSession::Gemma4Int4(engine)
    } else {
        let engine = Gemma4Engine::load_with_batch(
            &cfg.model_dir,
            params.weight_prefix,
            context_tokens,
            cfg.device,
            1,
        )?;
        InferenceSession::Gemma4Bf16(engine)
    };
    tracing::info!("weights loaded in {:.0}ms", t0.elapsed().as_millis());

    Ok((session, eos_ids))
}
