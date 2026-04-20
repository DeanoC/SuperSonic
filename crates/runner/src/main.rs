mod decode_engine;
mod gemma4_engine;
mod gemma4_int4_engine;
mod oracle;
mod phi4_engine;
mod prefill_engine;
mod qwen35_dflash_engine;
mod registry;
mod validate;

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use base64::Engine as _;
use clap::Parser;

use decode_engine::{DecodeEngine, DecodeStageTimings};
use qwen35::state::{LayerState, ModelState};
use registry::{Backend, FamilyParams, GpuArch, ModelFamily, ModelVariant};

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

    /// Run one decode step on every sequence in the batch. Both BF16 and
    /// INT4 engines honour `--batch-size > 1` via their batched persistent
    /// megakernels (BF16: `g4::persistent_decode_batch`, INT4:
    /// `g4::persistent_decode_batch_int4`).
    fn decode_step_batch(
        &mut self,
        input_tokens: &[u32],
        positions: &[usize],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Self::Bf16(e) => e.decode_step_batch(input_tokens, positions),
            Self::Int4(e) => e.decode_step_batch(input_tokens, positions),
        }
    }

    /// Replicate seq 0's K/V cache contents into every other sequence's
    /// caches. Applies to both BF16 and INT4 engines.
    fn replicate_seq0_kv(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Bf16(e) => e.replicate_seq0_kv(),
            Self::Int4(e) => e.replicate_seq0_kv(),
        }
    }

    fn batch_size(&self) -> usize {
        match self {
            Self::Bf16(e) => e.batch_size(),
            Self::Int4(e) => e.batch_size(),
        }
    }

    fn greedy_sample(logits: &[f32]) -> u32 {
        gemma4_engine::Gemma4Engine::greedy_sample(logits)
    }
}

#[derive(Clone, Copy)]
enum BackendChoice {
    Auto,
    Explicit(Backend),
}

impl BackendChoice {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "hip" => Some(Self::Explicit(Backend::Hip)),
            "cuda" => Some(Self::Explicit(Backend::Cuda)),
            "metal" => Some(Self::Explicit(Backend::Metal)),
            _ => None,
        }
    }
}

fn resolve_backend(choice: BackendChoice, ordinal: usize) -> Result<Backend> {
    match choice {
        BackendChoice::Explicit(backend) => {
            if !gpu_hal::is_backend_compiled(backend) {
                anyhow::bail!(
                    "Requested backend {backend} is not compiled into this build. Compiled backends: [{}]",
                    gpu_hal::compiled_backends()
                        .into_iter()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            Ok(backend)
        }
        BackendChoice::Auto => {
            if gpu_hal::is_backend_compiled(Backend::Cuda)
                && gpu_hal::query_device_info(Backend::Cuda, ordinal).is_ok()
            {
                return Ok(Backend::Cuda);
            }
            if gpu_hal::is_backend_compiled(Backend::Hip)
                && kernel_ffi::query_gpu_info(ordinal).is_ok()
            {
                return Ok(Backend::Hip);
            }
            if gpu_hal::is_backend_compiled(Backend::Metal)
                && gpu_hal::query_device_info(Backend::Metal, ordinal).is_ok()
            {
                return Ok(Backend::Metal);
            }
            anyhow::bail!(
                "No usable GPU backend available for device {ordinal}. Compiled backends: [{}]",
                gpu_hal::compiled_backends()
                    .into_iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

fn resolve_oracle_device(spec: &str, backend: Backend, ordinal: usize) -> String {
    match spec.trim().to_ascii_lowercase().as_str() {
        "auto" => match backend {
            Backend::Cuda => format!("cuda:{ordinal}"),
            Backend::Hip => "cpu".to_string(),
            Backend::Metal => "cpu".to_string(),
        },
        other => other.to_string(),
    }
}

#[derive(Parser)]
#[command(name = "supersonic", about = "SuperSonic — optimized LLM inference")]
pub(crate) struct Cli {
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

    /// Compute backend (`auto`, `hip`, `cuda`, or `metal`)
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Device ordinal on the selected backend
    #[arg(long, default_value = "0")]
    device: usize,

    /// Emit aggregated native decode stage timings at the end of the run.
    #[arg(long)]
    emit_stage_timings: bool,

    /// Run PyTorch oracle and compare logits
    #[arg(long)]
    validate: bool,

    /// Oracle dtype (bf16 or fp32)
    #[arg(long, default_value = "bf16")]
    oracle_dtype: String,

    /// Oracle device (`auto`, `cpu`, `cuda:0`, etc.)
    #[arg(long, default_value = "auto")]
    oracle_device: String,

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

    /// Validate decode against a replayed GPU prefill reference.
    /// Replays the full token history through native GPU prefill on each step and
    /// compares the resulting last-token logits against decode. Slower than decode,
    /// but avoids the stale incremental component-oracle path.
    #[arg(long)]
    gpu_validate: bool,

    /// Dump and compare per-layer prefill hidden states against the oracle.
    /// Debug-only path intended to localize long-context divergence.
    #[arg(long)]
    trace_prefill_layers: bool,

    /// Debug-only: when tracing Qwen prefill on Metal, also dump one selected
    /// linear-attention layer's internal tensors against the Python oracle.
    /// Defaults to a later linear layer to localize the current drift.
    #[arg(long, hide = true)]
    trace_prefill_linear_layer: Option<usize>,

    /// Batch size for decode (number of sequences decoded in parallel).
    /// Default 1. Supported on Qwen3.5 (requires 4B kernel: 2B/4B/9B models)
    /// and Gemma 4 BF16 + INT4 via per-family batched megakernels.
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// Debug-only: force single-sequence 4B decode to use the actual kernel path
    /// instead of the replayed prefill correctness path.
    /// (Historical — `replayed prefill` is no longer the default; the kernel
    /// path is used by default. This flag is kept as a no-op for callers
    /// that still pass it explicitly.)
    #[arg(long, hide = true)]
    force_kernel_decode: bool,

    /// Debug-only: force single-sequence 4B decode to use the component decode path
    /// instead of replayed prefill or the persistent kernel.
    #[arg(long, hide = true)]
    force_component_decode: bool,

    /// Debug-only: restore the legacy "replay prefill each decode step" path
    /// that was the default before 2026-04-20. Scales O(N) per token with
    /// context length and was ~7x slower than the persistent megakernel path,
    /// so retained only for parity validation. Mutually exclusive with
    /// --force-kernel-decode / --force-component-decode.
    #[arg(long, hide = true)]
    force_replay_decode: bool,

    /// Debug-only: allow unstable CUDA KV-FP8 experiments on the 4B kernel path.
    /// This is intentionally hidden until the path is validated.
    #[arg(long, hide = true)]
    allow_unstable_cuda_kv_fp8: bool,

    /// Debug-only: compare decode-appended KV-FP8 cache contents against a replayed
    /// prefill KV-FP8 reference after each decode step.
    #[arg(long, hide = true)]
    trace_kv_fp8_cache: bool,

    /// Debug-only: compare decode-appended KV cache contents against a replayed
    /// prefill reference after each decode step. Works for BF16 and FP8 KV.
    #[arg(long, hide = true)]
    trace_kv_cache: bool,

    /// Debug-only: on the component decode path, capture the BF16 hidden state
    /// immediately before this layer and compare it to replayed prefill.
    #[arg(long, hide = true)]
    trace_component_input_layer: Option<usize>,

    /// Debug-only: on the component decode path, compare one layer's stage outputs
    /// against replayed prefill (token mixer output, post-attn norm, mlp out, final hidden).
    #[arg(long, hide = true)]
    trace_component_layer: Option<usize>,

    /// Debug-only: on the component decode path, compare one linear-attention layer's
    /// internal tensors (qkv, z, attn, gated, proj_out) against replayed prefill.
    #[arg(long, hide = true)]
    trace_component_linear_layer: Option<usize>,

    /// Debug-only: compare one linear-attention layer's conv/recurrent state against
    /// replayed prefill before the decode step runs.
    #[arg(long, hide = true)]
    trace_component_linear_state_layer: Option<usize>,

    /// Debug-only: run the real persistent 4B kernel for the first N layers and compare
    /// the resulting hidden state against replayed prefill's input to that layer.
    #[arg(long, hide = true)]
    trace_persistent_input_layer: Option<usize>,

    /// Debug-only: run the real persistent 4B kernel through one selected linear layer
    /// and compare that layer's conv/recurrent state against replayed prefill.
    #[arg(long, hide = true)]
    trace_persistent_linear_state_layer: Option<usize>,

    /// Debug-only: compare one full-attention layer's K/V production on the real
    /// persistent path against the component full-attention path using the same
    /// hidden-state input.
    #[arg(long, hide = true)]
    trace_persistent_full_attn_layer: Option<usize>,

    /// Debug-only: compare one linear-attention layer's production on the real
    /// persistent path against the component linear path using the same
    /// hidden-state input and pre-step state.
    #[arg(long, hide = true)]
    trace_persistent_linear_layer: Option<usize>,

    /// Run on an arch without a registry entry by reusing another arch's kernel.
    /// Pass the arch name whose kernel you want to reuse (e.g. "gfx1150"). Emits
    /// a loud warning — correctness is not guaranteed. Intended for archs that
    /// are binary-compatible (same wavefront size, similar CU/LDS) but haven't
    /// been explicitly tuned.
    #[arg(long)]
    allow_untested_gpu: Option<String>,

    /// Disable downloading pre-baked weights from GitHub releases when the
    /// local bake is missing. Prints the manual bake guidance instead.
    #[arg(long)]
    no_download: bool,

    /// Force downloading a pre-baked package even if a valid local bake exists.
    #[arg(long)]
    download_bake: bool,

    /// Override the GitHub release/tag used for bake downloads.
    #[arg(long)]
    bake_release: Option<String>,

    /// Enable DFlash speculative decoding. Requires `--model qwen3.5-9b`,
    /// `--int4`, and `--dflash-draft-dir`. Target is the Qwen3.5-9B INT4
    /// bake; draft is the DFlash 5-layer checkpoint shared via Arc.
    #[arg(long)]
    dflash: bool,

    /// Path to the DFlash draft checkpoint directory (e.g.
    /// `z-lab/Qwen3.5-9B-DFlash` extracted locally). Must contain
    /// `config.json` and `model.safetensors`.
    #[arg(long)]
    dflash_draft_dir: Option<PathBuf>,

    /// Override the DFlash block size (draft candidates per round). Must
    /// be 1..=draft_config.block_size. Default is 3 — the fused verify
    /// megakernel on Qwen3.5-9B is LDS-bound and caps B at 3 on gfx1150
    /// (block_size + B*hidden + fp8_lut must fit in 64 KiB shared mem).
    /// Launches with B >= 4 fail fast with a shared-memory diagnostic.
    #[arg(long)]
    dflash_block: Option<usize>,

    /// Override the DFlash tap layers as a comma-separated list of
    /// target-model layer indices (e.g. `1,8,15,22,29`). Must match the
    /// count implied by the draft's `fc.in_features`. Defaults to the
    /// checkpoint's `dflash_config.target_layer_ids`.
    #[arg(long)]
    dflash_tap_layers: Option<String>,

}

fn resolve_release_source(cli: &Cli) -> Result<model_store::fetch::ReleaseSource> {
    let raw = cli
        .bake_release
        .clone()
        .or_else(|| env::var("SUPERSONIC_BAKE_RELEASE").ok());
    match raw {
        Some(s) if !s.is_empty() => model_store::fetch::ReleaseSource::from_override(&s)
            .map_err(|e| anyhow::anyhow!("invalid --bake-release: {e}")),
        _ => Ok(model_store::fetch::ReleaseSource::default_for_format_version()),
    }
}

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
        target_model_dir: &cli.model_dir,
        progress: &progress,
    };
    model_store::fetch::fetch_bake(req).map_err(|e| anyhow::anyhow!("fetch bake: {e}"))?;
    Ok(true)
}

/// Pick the variant the CLI flags imply, using the same INT4 > FP8 > BF16
/// priority order as the rest of the runner.
fn cli_variant(cli: &Cli) -> model_store::fetch::BakeVariant {
    if cli.int4 {
        model_store::fetch::BakeVariant::Int4Gptq
    } else if cli.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        model_store::fetch::BakeVariant::Bf16
    }
}

/// When `--model-dir` has no `config.json`, we can't load the tokenizer or
/// the model config — so fetch the bake first. The tarball bundles HF
/// metadata under `hf/`, which the downloader extracts into `--model-dir`
/// before anything else reads from it. This is the "fresh empty model dir"
/// path that makes release-hosted bakes self-sufficient.
fn ensure_hf_metadata_present(cli: &Cli, model_variant: &ModelVariant) -> Result<()> {
    if cli.no_bake || cli.no_download {
        return Ok(());
    }
    if cli.model_dir.join("config.json").exists() {
        return Ok(());
    }
    let variant = cli_variant(cli);
    let bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
    // Race: another process might have populated config between our check
    // above and the lock acquisition.
    if cli.model_dir.join("config.json").exists() {
        return Ok(());
    }
    let canonical_model = model_variant.to_string();
    eprintln!(
        "[fetch] --model-dir has no config.json; downloading bake to populate \
         HF metadata and weights in one pass"
    );
    try_download_bake(cli, variant, &canonical_model, &bake_dir)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ordinal = cli.device;
    let backend_choice = BackendChoice::parse(&cli.backend).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown backend '{}'. Expected one of: auto, hip, cuda, metal",
            cli.backend
        )
    })?;
    let backend = resolve_backend(backend_choice, ordinal)?;
    gpu_hal::set_backend(backend);

    // 1. Parse model variant
    let model_variant = ModelVariant::from_cli_str(&cli.model).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown model '{}'. Supported models: {}",
            cli.model,
            registry::supported_models_list().join(", ")
        )
    })?;

    // 2. Detect GPU
    let (arch_name, total_vram, warp_size) = match backend {
        Backend::Hip => {
            let (arch_name, total_vram) = kernel_ffi::query_gpu_info(ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            (arch_name, total_vram, 32)
        }
        Backend::Cuda => {
            let info = gpu_hal::query_device_info(backend, ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            (info.arch_name, info.total_vram_bytes, info.warp_size)
        }
        Backend::Metal => {
            let info = gpu_hal::query_device_info(backend, ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            (info.arch_name, info.total_vram_bytes, info.warp_size)
        }
    };
    let gpu_arch = GpuArch::from_backend_name(&backend, &arch_name);
    eprintln!(
        "[gpu] backend={backend} device={ordinal} arch={arch_name} warp={} vram={:.1}GiB",
        warp_size,
        total_vram as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // 3. Registry lookup
    let entry = match registry::lookup(&model_variant, &backend, &gpu_arch) {
        Some(e) => e,
        None => {
            if let Some(override_arch) = cli.allow_untested_gpu.as_deref() {
                let reuse_arch = GpuArch::from_backend_name(&backend, override_arch);
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

    // DFlash validation runs BEFORE the family dispatch so a misconfig on
    // non-Qwen families fails fast. The dispatch below returns for Gemma4/
    // Phi4 — if we deferred these checks until the Qwen branch, callers
    // would see --dflash / --dflash-draft-dir / etc. silently ignored on
    // other model families and think speculative decoding was enabled
    // when it wasn't.
    if cli.dflash && !matches!(model_variant.family(), ModelFamily::Qwen35) {
        anyhow::bail!(
            "--dflash is only supported on Qwen3.5 family models (got family={:?}, model={model_variant}).",
            model_variant.family(),
        );
    }
    if !cli.dflash
        && (cli.dflash_draft_dir.is_some()
            || cli.dflash_block.is_some()
            || cli.dflash_tap_layers.is_some())
    {
        anyhow::bail!("--dflash-* flags require --dflash");
    }

    match model_variant.family() {
        ModelFamily::Gemma4 => return run_gemma4(&cli, &model_variant, entry, ordinal, total_vram),
        ModelFamily::Phi4 => {
            return phi4_engine::run_phi4(&cli, &model_variant, entry, ordinal, total_vram);
        }
        ModelFamily::Qwen35 => {}
    }

    if cli.dflash {
        // DFlash needs the target's HF metadata (config.json + tokenizer.json)
        // and the INT4 bake. Reuse the same download hooks as the regular
        // Qwen35 path so the dflash dispatch is self-contained on a fresh
        // machine: ensure_hf_metadata_present fetches HF metadata from the
        // bake tarball if config.json is missing, then we verify or download
        // the INT4 bake itself.
        ensure_hf_metadata_present(&cli, &model_variant)?;
        if !cli.no_bake {
            let variant = model_store::fetch::BakeVariant::Int4Gptq;
            let bake_dir = variant.bake_dir(&cli.model_dir);
            let _lock = model_store::BakeLock::acquire(&cli.model_dir)
                .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
            if cli.download_bake || !model_store::version_ok(&bake_dir) {
                let canonical_model = model_variant.to_string();
                match try_download_bake(&cli, variant, &canonical_model, &bake_dir) {
                    Ok(true) => {
                        eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                    }
                    Ok(false) => {
                        anyhow::bail!(
                            "no INT4 bake at {} and --no-download set.\n\
                             Run:\n  python oracle/bake_int4.py --model-dir {}",
                            bake_dir.display(),
                            cli.model_dir.display(),
                        );
                    }
                    Err(e) => {
                        anyhow::bail!(
                            "could not obtain INT4 bake for --dflash: {e}\n\n\
                             INT4 baking requires a GPTQ calibration pass in Python. \
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                            cli.model_dir.display(),
                        );
                    }
                }
            }
        }
        return qwen35_dflash_engine::run_qwen35_dflash(
            &cli,
            &model_variant,
            entry,
            ordinal,
            total_vram,
        );
    }
    // --dflash-* guard already ran before the family dispatch above.

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => p,
        FamilyParams::Gemma4(_) => unreachable!("gemma4 handled above"),
        FamilyParams::Phi4(_) => unreachable!("phi4 handled above"),
    };

    if cli.trace_kv_fp8_cache && !cli.kv_fp8 {
        anyhow::bail!("--trace-kv-fp8-cache requires --kv-fp8");
    }
    let trace_kv_cache_enabled = cli.trace_kv_cache || cli.trace_kv_fp8_cache;
    if cli.force_kernel_decode && cli.force_component_decode {
        anyhow::bail!("Choose at most one of --force-kernel-decode or --force-component-decode");
    }
    if cli.trace_component_input_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-input-layer requires --force-component-decode");
    }
    if cli.trace_component_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_state_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-state-layer requires --force-component-decode");
    }
    if cli.trace_persistent_input_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!(
            "--trace-persistent-input-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_linear_state_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!(
            "--trace-persistent-linear-state-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_full_attn_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode))
    {
        anyhow::bail!(
            "--trace-persistent-full-attn-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_linear_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode)
            && !cli.kv_fp8)
    {
        anyhow::bail!(
            "--trace-persistent-linear-layer requires the real 4B persistent BF16 kernel path"
        );
    }

    if backend == Backend::Cuda {
        if cli.int4 {
            anyhow::bail!("CUDA v1 does not support --int4 yet");
        }
        if cli.fp8_runtime {
            anyhow::bail!("CUDA v1 does not support --fp8-runtime yet");
        }
        if cli.kv_fp8 {
            if !cli.allow_unstable_cuda_kv_fp8 {
                anyhow::bail!("CUDA v1 does not support --kv-fp8 yet");
            }
            if !params.use_4b_kernel {
                anyhow::bail!(
                    "--allow-unstable-cuda-kv-fp8 only supports the CUDA 4B kernel path"
                );
            }
            eprintln!(
                "[cuda] WARNING: enabling unstable CUDA KV-FP8 debug path; correctness is not guaranteed"
            );
        }
    }
    if backend == Backend::Metal {
        if model_variant != ModelVariant::Qwen3_5_0_8B {
            anyhow::bail!("Metal v1 only supports --model qwen3.5-0.8b");
        }
        if cli.int4 {
            anyhow::bail!("Metal v1 does not support --int4");
        }
        if cli.fp8_runtime {
            anyhow::bail!("Metal v1 does not support --fp8-runtime");
        }
        if cli.kv_fp8 {
            anyhow::bail!("Metal v1 does not support --kv-fp8");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("Metal v1 only supports --batch-size 1");
        }
        if cli.force_kernel_decode || cli.force_component_decode {
            anyhow::bail!("Metal v1 only supports replay-prefill decode");
        }
    }

    // If --model-dir is pristine (no config.json), fetch a bake first so the
    // downloader can populate HF metadata before we try to read it.
    ensure_hf_metadata_present(&cli, &model_variant)?;

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
            let local_bake_ok = matches!(
                variant,
                model_store::fetch::BakeVariant::Bf16 | model_store::fetch::BakeVariant::Fp8Native
            );
            let canonical_model = model_variant.to_string();
            match try_download_bake(&cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => {
                    eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                }
                Ok(false) => {
                    if !local_bake_ok {
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
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}\n\
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
                eprintln!("[bake] no baked package found — baking weights{mode_str} (one-time)...");
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
        }
        if model_store::version_ok(&bake_dir) {
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

    // attn_scratch_floats must cover saved_q+gate+pre_gate+scores for the largest
    // kv_max_t the run will hit. Registry value is the floor; scale up with context.
    let required_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        text_config.num_attention_heads,
        text_config.head_dim,
        context_tokens,
        params.kv_chunk_size,
    );
    let attn_scratch_floats = params.attn_scratch_floats.max(required_attn_scratch);
    if attn_scratch_floats > params.attn_scratch_floats {
        eprintln!(
            "[scratch] context={} → attn_scratch_floats={} (registry floor {})",
            context_tokens, attn_scratch_floats, params.attn_scratch_floats
        );
    }

    // Create decode engine
    let mut engine = DecodeEngine::new(
        weights,
        ordinal,
        params.proj_buf_floats,
        attn_scratch_floats,
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
    let oracle_device = resolve_oracle_device(&cli.oracle_device, backend, ordinal);
    if cli.trace_prefill_layers && !cli.validate {
        anyhow::bail!("--trace-prefill-layers requires --validate");
    }
    let trace_prefill_linear_layer = if cli.trace_prefill_layers
        && model_variant.family() == ModelFamily::Qwen35
    {
        let layer = cli.trace_prefill_linear_layer.unwrap_or(20);
        let config = &engine.weights().config;
        if layer >= config.num_hidden_layers {
            anyhow::bail!(
                "--trace-prefill-linear-layer {} out of range for {} layers",
                layer,
                config.num_hidden_layers
            );
        }
        if config.is_full_attention(layer) {
            anyhow::bail!(
                "--trace-prefill-linear-layer {} selects a full-attention layer; choose a linear-attention layer",
                layer
            );
        }
        Some(layer)
    } else {
        None
    };

    // Run prefill (native GPU or oracle)
    let prefill_start = Instant::now();
    let (
        prefill_logits,
        native_prefill_trace,
        native_linear_debug_trace,
        native_layer3_full_attn_trace,
        mut next_token,
    ) = if cli.oracle_prefill {
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
            &cli.oracle_dtype, &oracle_device, true,
            fp8_oracle_dir.as_deref(),
        )?;
        engine.load_prefill_state(&output)?;
        let first = output.generated_token_ids[0];
        eprintln!("[prefill] oracle prefill done in {:.0}ms", prefill_start.elapsed().as_millis());
        (output.prefill_logits, None, None, None, first)
    } else {
        let prefill_result = if cli.trace_prefill_layers {
            engine.prefill_native_with_trace(&prompt_ids, trace_prefill_linear_layer)?
        } else {
            prefill_engine::PrefillResult {
                logits: engine.prefill_native(&prompt_ids)?,
                final_norm_trace: None,
                layer_attn_trace: None,
                layer_post_attn_norm_trace: None,
                layer_mlp_swiglu_trace: None,
                layer_mlp_out_trace: None,
                layer_hidden_trace: None,
                tap_hiddens: None,
                linear_debug_trace: None,
                layer3_full_attn_trace: None,
            }
        };
        let first = DecodeEngine::greedy_sample(&prefill_result.logits);
        eprintln!("[prefill] native GPU prefill done in {:.0}ms", prefill_start.elapsed().as_millis());
        (
            prefill_result.logits,
            Some((
                prefill_result.final_norm_trace,
                prefill_result.layer_attn_trace,
                prefill_result.layer_post_attn_norm_trace,
                prefill_result.layer_mlp_out_trace,
                prefill_result.layer_hidden_trace,
            )),
            prefill_result.linear_debug_trace,
            prefill_result.layer3_full_attn_trace,
            first,
        )
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
            &oracle_device,
            cli.trace_prefill_layers,
            fp8_oracle_dir.as_deref(),
        )?;

        // Compare prefill logits
        let prefill_delta = validate::max_abs_delta(&prefill_logits, &output.prefill_logits);
        eprintln!("[validate] prefill logit delta={prefill_delta:.4}");

        let qwen35_trace_output = if cli.trace_prefill_layers
            && model_variant.family() == ModelFamily::Qwen35
        {
            let qwen35_trace_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .unwrap()
                .join("oracle/qwen35_oracle.py");
            Some(oracle::run_qwen35_trace_oracle(
                &qwen35_trace_script,
                &model_id,
                &prompt_ids,
                cli.max_new_tokens,
                &cli.oracle_dtype,
                &oracle_device,
                trace_prefill_linear_layer,
            )?)
        } else {
            None
        };

        if let (Some((native_final_norm_trace, ..)), Some(oracle_final_norm_b64)) = (
            native_prefill_trace.as_ref(),
            output.prefill_hidden.as_ref(),
        ) {
            if let Some(native_final_norm_trace) = native_final_norm_trace.as_ref() {
                let b64 = base64::engine::general_purpose::STANDARD;
                let oracle_final_norm_bytes = b64
                    .decode(oracle_final_norm_b64)
                    .map_err(|e| anyhow::anyhow!("decode oracle prefill_hidden: {e}"))?;
                let decode_bf16 = |bytes: &[u8]| -> Vec<f32> {
                    bytes.chunks_exact(2)
                        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                        .collect()
                };
                let native_final_norm = decode_bf16(native_final_norm_trace);
                let oracle_final_norm = decode_bf16(&oracle_final_norm_bytes);
                let final_norm_delta =
                    validate::max_abs_delta(&native_final_norm, &oracle_final_norm);
                eprintln!("[trace-prefill] final_norm_delta={final_norm_delta:.4}");
            }
        }

        // Check if oracle and native agree on first token
        let oracle_first = output.generated_token_ids[0];
        if oracle_first != next_token {
            eprintln!(
                "[validate] WARNING: prefill token mismatch! native={next_token} oracle={oracle_first}"
            );
        }

        if cli.trace_prefill_layers {
            if let (
                Some((_, native_attn_trace, native_post_norm_trace, native_mlp_out_trace, native_layer_trace)),
                Some(oracle_attn_trace),
                Some(oracle_post_norm_trace),
                Some(oracle_mlp_out_trace),
                Some(oracle_layer_trace),
            ) =
                (
                    native_prefill_trace.as_ref(),
                    output.layer_attn_residual_states.as_ref(),
                    output.layer_post_attn_norm_states.as_ref(),
                    output.layer_mlp_outputs.as_ref(),
                    output.layer_hidden_states.as_ref(),
                )
            {
                if let (
                    Some(native_attn_trace),
                    Some(native_post_norm_trace),
                    Some(native_mlp_out_trace),
                    Some(native_layer_trace),
                ) = (
                    native_attn_trace.as_ref(),
                    native_post_norm_trace.as_ref(),
                    native_mlp_out_trace.as_ref(),
                    native_layer_trace.as_ref(),
                )
                {
                let b64 = base64::engine::general_purpose::STANDARD;
                let mut first_bad = None;
                for layer in 0..native_layer_trace.len().min(oracle_layer_trace.len()) {
                    let decode_bf16 = |bytes: &[u8]| -> Vec<f32> {
                        bytes.chunks_exact(2)
                            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                            .collect()
                    };
                    let oracle_attn_bytes = b64
                        .decode(&oracle_attn_trace[layer])
                        .map_err(|e| anyhow::anyhow!("decode oracle layer_attn_residual_states[{layer}]: {e}"))?;
                    let oracle_post_norm_bytes = b64
                        .decode(&oracle_post_norm_trace[layer])
                        .map_err(|e| anyhow::anyhow!("decode oracle layer_post_attn_norm_states[{layer}]: {e}"))?;
                    let oracle_mlp_out_bytes = b64
                        .decode(&oracle_mlp_out_trace[layer])
                        .map_err(|e| anyhow::anyhow!("decode oracle layer_mlp_outputs[{layer}]: {e}"))?;
                    let oracle_layer_bytes = b64
                        .decode(&oracle_layer_trace[layer])
                        .map_err(|e| anyhow::anyhow!("decode oracle layer_hidden_states[{layer}]: {e}"))?;
                    let native_attn_f32 = decode_bf16(&native_attn_trace[layer]);
                    let native_post_norm_f32 = decode_bf16(&native_post_norm_trace[layer]);
                    let native_mlp_out_f32 = decode_bf16(&native_mlp_out_trace[layer]);
                    let native_layer_f32 = decode_bf16(&native_layer_trace[layer]);
                    let oracle_attn_f32 = decode_bf16(&oracle_attn_bytes);
                    let oracle_post_norm_f32 = decode_bf16(&oracle_post_norm_bytes);
                    let oracle_mlp_out_f32 = decode_bf16(&oracle_mlp_out_bytes);
                    let oracle_layer_f32 = decode_bf16(&oracle_layer_bytes);
                    let attn_delta = validate::max_abs_delta(&native_attn_f32, &oracle_attn_f32);
                    let post_norm_delta = validate::max_abs_delta(&native_post_norm_f32, &oracle_post_norm_f32);
                    let mlp_out_delta = validate::max_abs_delta(&native_mlp_out_f32, &oracle_mlp_out_f32);
                    let layer_delta = validate::max_abs_delta(&native_layer_f32, &oracle_layer_f32);
                    if first_bad.is_none() && layer_delta > 0.5 {
                        first_bad = Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta));
                    }
                    eprintln!(
                        "[trace-prefill] layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}"
                    );
                }
                if let Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta)) = first_bad {
                    eprintln!(
                        "[trace-prefill] first_bad_layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}"
                    );
                } else {
                    eprintln!("[trace-prefill] no layer exceeded delta threshold");
                }
                } else {
                    eprintln!("[trace-prefill] missing native attention, post-norm, mlp-out, or layer trace");
                }
            } else {
                eprintln!("[trace-prefill] missing native or oracle layer trace data");
            }

            if let (Some(native), Some(trace)) = (
                native_linear_debug_trace.as_ref(),
                qwen35_trace_output.as_ref(),
            ) {
                let trace_linear_layer = trace
                    .trace_linear_layer
                    .or(trace_prefill_linear_layer)
                    .unwrap_or(0);
                if let (
                    Some(oracle_normed),
                    Some(oracle_qkv),
                    Some(oracle_z),
                    Some(oracle_post_conv),
                    Some(oracle_q),
                    Some(oracle_k),
                    Some(oracle_v),
                    Some(oracle_beta),
                    Some(oracle_g),
                    Some(oracle_attn),
                    Some(oracle_gated),
                    Some(oracle_proj_out),
                ) = (
                    trace.trace_linear_input_layernorm_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_qkv_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_z_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_post_conv_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_prepared_query_output
                        .as_ref()
                        .and_then(flatten_last_token_bshd),
                    trace.trace_linear_prepared_key_output
                        .as_ref()
                        .and_then(flatten_last_token_bshd),
                    trace.trace_linear_prepared_value_output
                        .as_ref()
                        .and_then(flatten_last_token_bshd),
                    trace.trace_linear_prepared_beta_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_prepared_g_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_direct_recurrent_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_norm_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                    trace.trace_linear_token_mixer_output
                        .as_ref()
                        .and_then(flatten_last_token_bsd),
                ) {
                    let native_normed = decode_bf16_le(&native.normed);
                    let native_qkv = decode_bf16_le(&native.qkv);
                    let native_z = decode_bf16_le(&native.z);
                    let native_post_conv = decode_bf16_le(&native.post_conv);
                    let native_packed = decode_f32_le(&native.packed);
                    let native_attn = decode_bf16_le(&native.attn);
                    let native_gated = decode_bf16_le(&native.gated);
                    let native_proj_out = decode_bf16_le(&native.proj_out);

                    let cfg = &engine.weights().config;
                    let nv = cfg.linear_num_value_heads;
                    let khd = cfg.linear_key_head_dim;
                    let vhd = cfg.linear_value_head_dim;
                    let key_dim = cfg.linear_num_key_heads * khd;
                    let val_dim = nv * vhd;
                    let packed_width = 2 * khd + vhd + 2;
                    let q_scale = 1.0f32 / (khd as f32).sqrt();
                    let mut oracle_packed = vec![0.0f32; nv * packed_width];
                    for head in 0..nv {
                        let out_base = head * packed_width;
                        let q_base = head * khd;
                        let k_base = head * khd;
                        let v_base = head * vhd;
                        for i in 0..khd {
                            oracle_packed[out_base + i] = oracle_q[q_base + i] * q_scale;
                            oracle_packed[out_base + khd + i] = oracle_k[k_base + i];
                        }
                        for i in 0..vhd {
                            oracle_packed[out_base + 2 * khd + i] = oracle_v[v_base + i];
                        }
                        oracle_packed[out_base + 2 * khd + vhd] = oracle_beta[head];
                        oracle_packed[out_base + 2 * khd + vhd + 1] = oracle_g[head].exp();
                    }
                    let mut q_delta = 0.0f32;
                    let mut k_delta = 0.0f32;
                    let mut v_delta = 0.0f32;
                    let conv_q_delta = validate::max_abs_delta(
                        &native_post_conv[..key_dim],
                        &oracle_post_conv[..key_dim],
                    );
                    let conv_k_delta = validate::max_abs_delta(
                        &native_post_conv[key_dim..key_dim * 2],
                        &oracle_post_conv[key_dim..key_dim * 2],
                    );
                    let conv_v_delta = validate::max_abs_delta(
                        &native_post_conv[key_dim * 2..key_dim * 2 + val_dim],
                        &oracle_post_conv[key_dim * 2..key_dim * 2 + val_dim],
                    );
                    let mut beta_delta = 0.0f32;
                    let mut gexp_delta = 0.0f32;
                    for head in 0..nv {
                        let base = head * packed_width;
                        q_delta = q_delta.max(validate::max_abs_delta(
                            &native_packed[base..base + khd],
                            &oracle_packed[base..base + khd],
                        ));
                        k_delta = k_delta.max(validate::max_abs_delta(
                            &native_packed[base + khd..base + 2 * khd],
                            &oracle_packed[base + khd..base + 2 * khd],
                        ));
                        v_delta = v_delta.max(validate::max_abs_delta(
                            &native_packed[base + 2 * khd..base + 2 * khd + vhd],
                            &oracle_packed[base + 2 * khd..base + 2 * khd + vhd],
                        ));
                        beta_delta = beta_delta.max(
                            (native_packed[base + 2 * khd + vhd]
                                - oracle_packed[base + 2 * khd + vhd])
                                .abs(),
                        );
                        gexp_delta = gexp_delta.max(
                            (native_packed[base + 2 * khd + vhd + 1]
                                - oracle_packed[base + 2 * khd + vhd + 1])
                                .abs(),
                        );
                    }

                    eprintln!(
                        "[trace-prefill-linear] layer={} normed_delta={:.4} qkv_delta={:.4} z_delta={:.4} post_conv_delta={:.4} conv_q_delta={:.4} conv_k_delta={:.4} conv_v_delta={:.4} packed_delta={:.4} q_delta={:.4} k_delta={:.4} v_delta={:.4} beta_delta={:.4} gexp_delta={:.4} attn_delta={:.4} gated_delta={:.4} proj_out_delta={:.4}",
                        trace_linear_layer,
                        validate::max_abs_delta(&native_normed, &oracle_normed),
                        validate::max_abs_delta(&native_qkv, &oracle_qkv),
                        validate::max_abs_delta(&native_z, &oracle_z),
                        validate::max_abs_delta(&native_post_conv, &oracle_post_conv),
                        conv_q_delta,
                        conv_k_delta,
                        conv_v_delta,
                        validate::max_abs_delta(&native_packed, &oracle_packed),
                        q_delta,
                        k_delta,
                        v_delta,
                        beta_delta,
                        gexp_delta,
                        validate::max_abs_delta(&native_attn, &oracle_attn),
                        validate::max_abs_delta(&native_gated, &oracle_gated),
                        validate::max_abs_delta(&native_proj_out, &oracle_proj_out),
                    );
                } else {
                    eprintln!(
                        "[trace-prefill-linear] layer={} missing flattenable qwen35 trace tensors",
                        trace_linear_layer
                    );
                }
            } else if model_variant.family() == ModelFamily::Qwen35 {
                let layer = trace_prefill_linear_layer.unwrap_or(0);
                eprintln!(
                    "[trace-prefill-linear] layer={} missing native or qwen35 trace data",
                    layer
                );
            }

            if let (Some(native), Some(trace)) = (
                native_layer3_full_attn_trace.as_ref(),
                qwen35_trace_output.as_ref(),
            ) {
                if let (
                    Some(oracle_q_and_gate),
                    Some(oracle_gate_proj),
                    Some(oracle_k_proj),
                    Some(oracle_v_proj),
                    Some(oracle_q_prepared),
                    Some(oracle_k_prepared),
                    Some(oracle_v_prepared),
                    Some(oracle_attn),
                ) = (
                    flatten_last_token_bsd(&trace.layer3_q_and_gate_output),
                    flatten_last_token_bsd(&trace.layer3_gate_output),
                    flatten_last_token_bsd(&trace.layer3_k_proj_output),
                    flatten_last_token_bsd(&trace.layer3_v_proj_output),
                    flatten_last_token_bhsd(&trace.layer3_prepared_query_output),
                    flatten_last_token_bhsd(&trace.layer3_prepared_key_output),
                    flatten_last_token_bhsd(&trace.layer3_prepared_value_output),
                    flatten_last_token_bsd(&trace.layer3_attention_output),
                ) {
                    let native_q_proj = decode_bf16_le(&native.q_proj);
                    let native_gate_proj = decode_bf16_le(&native.gate_proj);
                    let native_k_proj = decode_bf16_le(&native.k_proj);
                    let native_v_proj = decode_bf16_le(&native.v_proj);
                    let native_q_prepared = decode_bf16_le(&native.q_prepared);
                    let native_k_prepared = decode_bf16_le(&native.k_prepared);
                    let native_v_prepared = decode_bf16_le(&native.v_prepared);
                    let native_attn = decode_bf16_le(&native.attn_output);
                    let cfg = &engine.weights().config;
                    let num_heads = cfg.num_attention_heads;
                    let head_dim = cfg.head_dim;
                    let q_dim = num_heads * head_dim;
                    let mut oracle_q_proj = vec![0.0f32; q_dim];
                    for head in 0..num_heads {
                        let src_base = head * head_dim * 2;
                        let dst_base = head * head_dim;
                        oracle_q_proj[dst_base..dst_base + head_dim]
                            .copy_from_slice(&oracle_q_and_gate[src_base..src_base + head_dim]);
                    }
                    eprintln!(
                        "[trace-prefill-layer3-full] q_proj_delta={:.4} gate_proj_delta={:.4} k_proj_delta={:.4} v_proj_delta={:.4} q_prepared_delta={:.4} k_prepared_delta={:.4} v_prepared_delta={:.4} attn_output_delta={:.4}",
                        validate::max_abs_delta(&native_q_proj, &oracle_q_proj),
                        validate::max_abs_delta(&native_gate_proj, &oracle_gate_proj),
                        validate::max_abs_delta(&native_k_proj, &oracle_k_proj),
                        validate::max_abs_delta(&native_v_proj, &oracle_v_proj),
                        validate::max_abs_delta(&native_q_prepared, &oracle_q_prepared),
                        validate::max_abs_delta(&native_k_prepared, &oracle_k_prepared),
                        validate::max_abs_delta(&native_v_prepared, &oracle_v_prepared),
                        validate::max_abs_delta(&native_attn, &oracle_attn),
                    );
                } else {
                    eprintln!("[trace-prefill-layer3-full] missing flattenable qwen35 trace tensors");
                }
            } else if model_variant.family() == ModelFamily::Qwen35 {
                eprintln!("[trace-prefill-layer3-full] missing native or qwen35 trace data");
            }
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

    let gpu_validate_enabled = if cli.gpu_validate && cli.batch_size == 1 {
        eprintln!("[gpu-validate] replaying full token history through GPU prefill for reference...");
        true
    } else {
        if cli.gpu_validate && cli.batch_size > 1 {
            eprintln!("[gpu-validate] GPU oracle disabled for batch_size > 1");
        }
        false
    };
    // Replay-prefill path used to be the default for 4B single-seq decode
    // (safety net for numerical-parity work during the CUDA sm86 bring-up)
    // but it scales O(N) per step with context length and was ~7x slower
    // than the persistent megakernel at 64-token generations. Default is now
    // the megakernel; --force-replay-decode re-enables the replay path for
    // the rare case where someone genuinely wants to reproduce the older
    // numeric semantics.
    let replay_decode_enabled = cli.batch_size == 1
        && !cli.force_kernel_decode
        && !cli.force_component_decode
        && !cli.kv_fp8
        && (backend == Backend::Metal
            || (params.use_4b_kernel && cli.force_replay_decode));
    let replay_kv_fp8_enabled = params.use_4b_kernel
        && cli.kv_fp8
        && cli.allow_unstable_cuda_kv_fp8
        && !cli.force_kernel_decode;
    let component_single_decode_enabled =
        cli.batch_size == 1 && params.use_4b_kernel && cli.force_component_decode;
    // Use the batched persistent megakernel path (decode_step_batch with b=1)
    // for 4B single-seq decode by default — measured ~300 ms/tok on gfx1150
    // vs ~500 ms/tok for decode_step() and ~2500 ms/tok for the legacy
    // replay path. Opt-out via --force-replay-decode (legacy parity) or
    // --force-component-decode (primitive-chain correctness).
    let kernel_single_decode_enabled = cli.batch_size == 1
        && params.use_4b_kernel
        && !cli.force_replay_decode
        && !cli.force_component_decode;
    let cuda_08b_hero_disabled = env::var_os("SUPERSONIC_DISABLE_CUDA_08B_HERO").is_some();
    let cuda_08b_hero_enabled = backend == Backend::Cuda
        && gpu_arch == GpuArch::Sm86
        && model_variant == ModelVariant::Qwen3_5_0_8B
        && !params.use_4b_kernel
        && cli.batch_size == 1
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && oracle_output.is_none()
        && !engine.weights().is_fp8
        && !engine.weights().is_int4
        && !cuda_08b_hero_disabled;
    let cuda_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_CUDA_FAST_GREEDY").is_some();
    let cuda_fast_greedy_enabled = backend == Backend::Cuda
        && !params.use_4b_kernel
        && cli.batch_size == 1
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && oracle_output.is_none()
        && !cuda_08b_hero_enabled
        && !cuda_fast_greedy_disabled;
    if replay_decode_enabled {
        if backend == Backend::Metal {
            eprintln!("[decode] Metal v1 replays native prefill for each decode step");
        } else {
            eprintln!("[decode] single-sequence 4B uses replayed GPU prefill for correctness");
        }
    } else if replay_kv_fp8_enabled && cli.batch_size == 1 {
        eprintln!("[decode] experimental single-sequence KV-FP8 uses replayed GPU prefill for correctness");
    } else if replay_kv_fp8_enabled && cli.batch_size > 1 {
        eprintln!("[decode] experimental batched KV-FP8 rebuilds state from replayed GPU prefill each step");
    } else if component_single_decode_enabled {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the component decode path");
    } else if cli.batch_size == 1 && params.use_4b_kernel && cli.force_kernel_decode {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the kernel decode path");
    } else if cli.batch_size == 1 && params.use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] WARNING: experimental single-sequence KV-FP8 uses the b=1 kernel path");
    } else if cuda_08b_hero_enabled {
        eprintln!("[decode] CUDA 0.8B sm86 hero path enabled (fused lm_head argmax)");
    } else if cuda_fast_greedy_enabled {
        eprintln!("[decode] CUDA fast greedy sampling enabled for the non-4B native decode path");
    }

    // Decode loop
    let seqlen_start = prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;
    let mut gpu_max_delta = 0.0f32;
    let mut native_decode_timings = DecodeStageTimings::default();
    let mut native_decode_timing_steps = 0usize;
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
            if let Some(trace_layer) = cli.trace_persistent_linear_state_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                let _ = engine.decode_step_batch_trace_hidden_after_layers(
                    &trace_tokens,
                    seqlen_offset,
                    trace_layer + 1,
                    0,
                )?;
                trace_persistent_linear_state_layer(
                    &engine,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            if let Some(trace_layer) = cli.trace_persistent_input_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
                    &trace_tokens,
                    seqlen_offset,
                    trace_layer,
                    0,
                )?;
                trace_persistent_input_layer(
                    &engine,
                    &native_hidden,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            if let Some(trace_layer) = cli.trace_persistent_full_attn_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                trace_persistent_full_attn_layer(
                    &mut engine,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    trace_tokens.as_slice(),
                    seqlen_offset,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            if let Some(trace_layer) = cli.trace_persistent_linear_layer {
                let trace_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let trace_tokens = vec![next_token; cli.batch_size];
                trace_persistent_linear_layer(
                    &mut engine,
                    trace_layer,
                    trace_token_ids.as_slice(),
                    trace_tokens.as_slice(),
                    seqlen_offset,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                )?;
                engine.rebuild_prefill_state(&trace_token_ids, true)?;
            }
            let (batch_logits, batch_timings) = if replay_kv_fp8_enabled {
                let token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let logits = engine.rebuild_prefill_state(&token_ids, true)?;
                (vec![logits; cli.batch_size], None)
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_batch_with_timings(&batch_next_tokens, seqlen_offset)?;
                (logits, Some(timings))
            } else {
                // Batched decode
                (engine.decode_step_batch(&batch_next_tokens, seqlen_offset)?, None)
            };
            if let Some(timings) = batch_timings {
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
            }

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
            let sampling_start = Instant::now();
            for (bi, seq_logits) in batch_logits.iter().enumerate() {
                batch_next_tokens[bi] = DecodeEngine::greedy_sample(seq_logits);
            }
            if batch_timings.is_some() {
                native_decode_timings.host_sampling_ms +=
                    sampling_start.elapsed().as_secs_f64() * 1000.0;
            }

            generated_ids.push(next_token);
            if trace_kv_cache_enabled {
                let cache_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .collect();
                trace_kv_cache(
                    &engine,
                    &cache_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                    cli.kv_fp8,
                    cli.batch_size,
                    step,
                )?;
            }
            next_token = batch_next_tokens[0];
        } else {
            // Single-sequence decode (original path)
            let mut maybe_fast_token = None;
            let logits = if cuda_fast_greedy_enabled {
                let (token, timings) = engine.decode_step_cuda_fast_greedy(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if cuda_08b_hero_enabled {
                let (token, timings) = engine.decode_step_cuda_08b_hero(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if replay_decode_enabled {
                let token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                prefill_engine::gpu_reference_replay_step(
                    &engine.weights(),
                    &engine.rotary(),
                    &token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                )?
            } else if replay_kv_fp8_enabled {
                let token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                engine.rebuild_prefill_state(&token_ids, false)?
            } else if component_single_decode_enabled {
                if let Some(trace_layer) = cli.trace_component_linear_state_layer {
                    trace_component_linear_state_layer(
                        &engine,
                        trace_layer,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                }
                if let Some(trace_layer) = cli.trace_component_input_layer {
                    let (logits, hidden_trace) =
                        engine.component_decode_step_4b_traced(next_token, seqlen_offset, trace_layer)?;
                    trace_component_input_layer(
                        &engine,
                        &hidden_trace,
                        trace_layer,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .chain(std::iter::once(next_token))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                    logits
                } else if let Some(trace_layer) = cli.trace_component_layer {
                    let (logits, layer_trace) =
                        engine.component_decode_step_4b_trace_layer(next_token, seqlen_offset, trace_layer)?;
                    trace_component_layer(
                        &engine,
                        trace_layer,
                        &layer_trace,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .chain(std::iter::once(next_token))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                    logits
                } else if let Some(trace_layer) = cli.trace_component_linear_layer {
                    let (logits, linear_trace) = engine
                        .component_decode_step_4b_trace_linear_layer(next_token, seqlen_offset, trace_layer)?;
                    trace_component_linear_layer(
                        &engine,
                        trace_layer,
                        &linear_trace,
                        prompt_ids
                            .iter()
                            .copied()
                            .chain(generated_ids.iter().copied())
                            .chain(std::iter::once(next_token))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                    logits
                } else {
                    engine.decode_step(next_token, seqlen_offset)?
                }
            } else if kernel_single_decode_enabled {
                if let Some(trace_layer) = cli.trace_persistent_linear_state_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    let _ = engine.decode_step_batch_trace_hidden_after_layers(
                        &[next_token],
                        seqlen_offset,
                        trace_layer + 1,
                        0,
                    )?;
                    trace_persistent_linear_state_layer(
                        &engine,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if let Some(trace_layer) = cli.trace_persistent_input_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
                        &[next_token],
                        seqlen_offset,
                        trace_layer,
                        0,
                    )?;
                    trace_persistent_input_layer(
                        &engine,
                        &native_hidden,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if cli.emit_stage_timings {
                    let (logits, timings) =
                        engine.decode_step_4b_single_kernel_with_timings(next_token, seqlen_offset)?;
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    logits
                } else {
                    engine.decode_step_batch(&[next_token], seqlen_offset)?.remove(0)
                }
            } else if cli.emit_stage_timings {
                let (logits, timings) = engine.decode_step_with_timings(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                logits
            } else {
                engine.decode_step(next_token, seqlen_offset)?
            };
            let native_token = maybe_fast_token.unwrap_or_else(|| DecodeEngine::greedy_sample(&logits));

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

            if gpu_validate_enabled {
                let gpu_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect();
                let gpu_logits = prefill_engine::gpu_reference_replay_step(
                    &engine.weights(),
                    &engine.rotary(),
                    &gpu_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                )?;
                let delta = validate::max_abs_delta(&logits, &gpu_logits);
                let gpu_token = DecodeEngine::greedy_sample(&gpu_logits);
                let token_match = if gpu_token == native_token { "" } else { " MISMATCH" };
                if delta > gpu_max_delta { gpu_max_delta = delta; }
                eprintln!(
                    "[gpu-validate] step={step} seq_off={seqlen_offset} delta={delta:.4} native_token={native_token} gpu_token={gpu_token}{token_match}"
                );
            }

            generated_ids.push(next_token);
            next_token = native_token;

            if trace_kv_cache_enabled {
                let cache_token_ids: Vec<u32> = prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .collect();
                trace_kv_cache(
                    &engine,
                    &cache_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    params.use_4b_kernel,
                    cli.kv_fp8,
                    cli.batch_size,
                    step,
                )?;
            }
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
    if cli.emit_stage_timings {
        if native_decode_timing_steps > 0 {
            eprintln!(
                "[stage-timings] steps={} persistent_ms={:.3} rms_norm_ms={:.3} lm_head_ms={:.3} logits_d2h_ms={:.3} host_sampling_ms={:.3} gpu_argmax_ms={:.3} token_d2h_ms={:.3} total_native_decode_ms={:.3} persistent_full_attn_ms={:.3} persistent_full_attn_proj_ms={:.3} persistent_full_attn_core_ms={:.3} persistent_full_attn_out_ms={:.3} persistent_linear_proj_ms={:.3} persistent_linear_core_ms={:.3} persistent_linear_core_conv_ms={:.3} persistent_linear_core_recurrent_ms={:.3} persistent_linear_core_post_ms={:.3} persistent_linear_out_ms={:.3} persistent_mlp_gate_up_ms={:.3} persistent_mlp_down_ms={:.3}",
                native_decode_timing_steps,
                native_decode_timings.persistent_ms,
                native_decode_timings.rms_norm_ms,
                native_decode_timings.lm_head_ms,
                native_decode_timings.logits_d2h_ms,
                native_decode_timings.host_sampling_ms,
                native_decode_timings.gpu_argmax_ms,
                native_decode_timings.token_d2h_ms,
                native_decode_timings.total_ms(),
                native_decode_timings.persistent_full_attn_ms,
                native_decode_timings.persistent_full_attn_proj_ms,
                native_decode_timings.persistent_full_attn_core_ms,
                native_decode_timings.persistent_full_attn_out_ms,
                native_decode_timings.persistent_linear_proj_ms,
                native_decode_timings.persistent_linear_core_ms,
                native_decode_timings.persistent_linear_core_conv_ms,
                native_decode_timings.persistent_linear_core_recurrent_ms,
                native_decode_timings.persistent_linear_core_post_ms,
                native_decode_timings.persistent_linear_out_ms,
                native_decode_timings.persistent_mlp_gate_up_ms,
                native_decode_timings.persistent_mlp_down_ms,
            );
        } else {
            eprintln!("[stage-timings] steps=0 note=no native decode stage timings collected for this path");
        }
    }

    Ok(())
}

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
        FamilyParams::Phi4(_) => unreachable!("dispatch filtered to Gemma4"),
    };

    if cli.fp8_runtime || cli.kv_fp8 {
        anyhow::bail!("Gemma 4 does not yet support --fp8-runtime / --kv-fp8");
    }
    if cli.batch_size < 1 || cli.batch_size > kernel_ffi::MAX_BATCH_SIZE {
        anyhow::bail!(
            "--batch-size must be 1..{}",
            kernel_ffi::MAX_BATCH_SIZE
        );
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

    // Fetch first if --model-dir is pristine so HF metadata lands before config load.
    ensure_hf_metadata_present(cli, model_variant)?;

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
    // Both BF16 and INT4 engines allocate `batch_size` parallel KV cache sets
    // (one per decode sequence); scale accordingly so `--batch-size > 1` can't
    // pass the preflight check and then OOM during engine load.
    let batch_size_u64 = cli.batch_size as u64;
    let kv_bytes_per_seq = kv_per_token * context_tokens as u64 * BF16_BYTES;
    let kv_bytes = kv_bytes_per_seq * batch_size_u64;
    let estimated_vram =
        ((entry.vram.fixed_bytes + kv_bytes) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights+scratch={:.2}GiB + kv_cache={:.2}GiB for {}tok x B={}) available={:.1}GiB",
        gib(estimated_vram),
        gib(entry.vram.fixed_bytes),
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
                        "No INT4 bake at {} and --no-download set.\nRun on a bigger machine:\n  python oracle/bake_int4_gemma4.py --model-dir {}",
                        target.display(),
                        cli.model_dir.display(),
                    );
                }
                Err(e) => {
                    anyhow::bail!(
                        "could not obtain Gemma 4 INT4 bake: {e}\n\nRun on a bigger machine:\n  python oracle/bake_int4_gemma4.py --model-dir {}\nthen `python oracle/upload_bake.py --model {} --int4 --model-dir {}` to publish.",
                        cli.model_dir.display(),
                        cli.model,
                        cli.model_dir.display(),
                    );
                }
            }
        }
        eprintln!("[gemma4] loading INT4 GPTQ bake (primitive-chain decode)");
        Gemma4Runtime::Int4(gemma4_int4_engine::Gemma4Int4Engine::load_with_batch(
            &cli.model_dir,
            params.weight_prefix,
            context_tokens,
            ordinal,
            cli.batch_size,
        )?)
    } else {
        Gemma4Runtime::Bf16(gemma4_engine::Gemma4Engine::load_with_batch(
            &cli.model_dir,
            params.weight_prefix,
            context_tokens,
            ordinal,
            cli.batch_size,
        )?)
    };
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

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
    let prefill_logits = engine.prefill(&prompt_ids)?;
    let prefill_token = Gemma4Runtime::greedy_sample(&prefill_logits);
    eprintln!(
        "[prefill] native GPU prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );

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
        // have stopped — the megakernel still has to handle the active ones.
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
        let logits_per_seq = engine.decode_step_batch(&next_tokens, &positions)?;

        if let Some(ref oracle) = oracle_output {
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

        // Sample per sequence and roll forward — but only record sampled
        // tokens for sequences that haven't already hit EOS.
        for b in 0..batch_size {
            if seq_done[b] {
                continue;
            }
            let sampled = Gemma4Runtime::greedy_sample(&logits_per_seq[b]);
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
        if batch_size == 1 {
            println!("{text}");
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

use decode_engine::decode_f32_le;

fn bf16_residual_sum(lhs_bf16: &[u8], rhs_bf16: &[u8]) -> Vec<f32> {
    lhs_bf16
        .chunks_exact(2)
        .zip(rhs_bf16.chunks_exact(2))
        .map(|(l, r)| {
            let sum = half::bf16::from_le_bytes([l[0], l[1]]).to_f32()
                + half::bf16::from_le_bytes([r[0], r[1]]).to_f32();
            half::bf16::from_f32(sum).to_f32()
        })
        .collect()
}

fn trace_kv_cache(
    engine: &DecodeEngine,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    kv_fp8: bool,
    batch_size: usize,
    step: usize,
) -> Result<()> {
    let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
        .map_err(|e| anyhow::anyhow!("kv-fp8 trace replay state init: {e}"))?;
    prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        false,
        None,
    )?;

    for batch_index in 0..batch_size {
        let native_state = engine.state_for_batch(batch_index);
        let mut first_bad = None;
        for (layer_idx, (native_layer, replay_layer)) in native_state
            .layers
            .iter()
            .zip(replay_state.layers.iter())
            .enumerate()
        {
            if !matches!(native_layer.kind, qwen35::weights::LayerKind::Full) {
                continue;
            }
            let diff = compare_kv_layer(native_layer, replay_layer)?;
            if first_bad.is_none()
                && (diff.k_mismatches > 0
                    || diff.v_mismatches > 0
                    || diff.max_scale_k_delta > 0.0
                    || diff.max_scale_v_delta > 0.0)
            {
                first_bad = Some((layer_idx, diff));
            }
        }
        if let Some((layer_idx, diff)) = first_bad {
            eprintln!(
                "[trace-kv-cache] step={step} batch={batch_index} first_bad_layer={layer_idx} filled={} dtype={} k_mismatches={} v_mismatches={} max_k_delta={:.6} max_v_delta={:.6} max_scale_k_delta={:.6} max_scale_v_delta={:.6}{}{}",
                diff.filled,
                diff.dtype,
                diff.k_mismatches,
                diff.v_mismatches,
                diff.max_k_delta,
                diff.max_v_delta,
                diff.max_scale_k_delta,
                diff.max_scale_v_delta,
                diff.first_k_mismatch
                    .map(|(h, t, d, native, replay)| format!(
                        " first_k_mismatch=(h={h},t={t},d={d},native={native},replay={replay})"
                    ))
                    .unwrap_or_default(),
                diff.first_v_mismatch
                    .map(|(h, t, d, native, replay)| format!(
                        " first_v_mismatch=(h={h},t={t},d={d},native={native},replay={replay})"
                    ))
                    .unwrap_or_default(),
            );
        } else {
            eprintln!(
                "[trace-kv-cache] step={step} batch={batch_index} all_full_attention_layers_match"
            );
        }
    }

    Ok(())
}

struct KvFp8LayerDiff {
    filled: usize,
    dtype: &'static str,
    k_mismatches: usize,
    v_mismatches: usize,
    max_k_delta: f32,
    max_v_delta: f32,
    max_scale_k_delta: f32,
    max_scale_v_delta: f32,
    first_k_mismatch: Option<(usize, usize, usize, u8, u8)>,
    first_v_mismatch: Option<(usize, usize, usize, u8, u8)>,
}

fn decode_bf16_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn flatten_json_numbers(value: &serde_json::Value, out: &mut Vec<f32>) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                flatten_json_numbers(value, out);
            }
        }
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                out.push(value as f32);
            }
        }
        _ => {}
    }
}

fn flatten_last_token_bsd(value: &serde_json::Value) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let token = batch.last()?;
    let mut out = Vec::new();
    flatten_json_numbers(token, &mut out);
    Some(out)
}

fn flatten_last_token_bhsd(value: &serde_json::Value) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let mut out = Vec::new();
    for head in batch {
        let token = head.as_array()?.last()?;
        flatten_json_numbers(token, &mut out);
    }
    Some(out)
}

fn flatten_last_token_bshd(value: &serde_json::Value) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let token = batch.last()?.as_array()?;
    let mut out = Vec::new();
    for head in token {
        flatten_json_numbers(head, &mut out);
    }
    Some(out)
}

fn fp8_e4m3_to_f32_host(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xF;
    let mantissa = byte & 0x7;
    if byte == 0x7F || byte == 0xFF {
        return 0.0;
    }
    let val = if exp == 0 {
        f32::from(mantissa) / 8.0 * 1.52587890625e-2
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };
    if sign != 0 { -val } else { val }
}

fn f32_to_bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
        .collect()
}

fn trace_component_input_layer(
    engine: &DecodeEngine,
    native_hidden: &[u8],
    trace_layer: usize,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("component input trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = if trace_layer == 0 {
        None
    } else {
        replay
            .layer_hidden_trace
            .as_ref()
            .and_then(|layers| layers.get(trace_layer - 1))
    };
    if let Some(replay_hidden) = replay_hidden {
        let native_f32 = decode_bf16_le(native_hidden);
        let replay_f32 = decode_bf16_le(replay_hidden);
        let delta = validate::max_abs_delta(&native_f32, &replay_f32);
        eprintln!(
            "[trace-component-input] layer={trace_layer} hidden_delta={delta:.6}"
        );
    } else {
        eprintln!(
            "[trace-component-input] layer={trace_layer} has no replay previous-layer hidden reference"
        );
    }
    Ok(())
}

fn trace_persistent_input_layer(
    engine: &DecodeEngine,
    native_hidden: &[u8],
    trace_layer: usize,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("persistent input trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = if trace_layer == 0 {
        None
    } else {
        replay
            .layer_hidden_trace
            .as_ref()
            .and_then(|layers| layers.get(trace_layer - 1))
    };
    if let Some(replay_hidden) = replay_hidden {
        let native_f32 = decode_bf16_le(native_hidden);
        let replay_f32 = decode_bf16_le(replay_hidden);
        let delta = validate::max_abs_delta(&native_f32, &replay_f32);
        eprintln!(
            "[trace-persistent-input] layer={trace_layer} hidden_delta={delta:.6}"
        );
    } else {
        eprintln!(
            "[trace-persistent-input] layer={trace_layer} has no replay previous-layer hidden reference"
        );
    }
    Ok(())
}

fn trace_persistent_linear_state_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent linear trace replay state init: {e}"))?;
    prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
    )?;

    let native_state = engine.state_for_batch(0);
    let native_layer = native_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("native layer {trace_layer} out of range"))?;
    let replay_layer = replay_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("replay layer {trace_layer} out of range"))?;

    let (conv_delta, first_conv_mismatch) = match (&native_layer.conv_state, &replay_layer.conv_state) {
        (Some(native), Some(replay)) => {
            let native_vals = decode_bf16_le(
                &native
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("native persistent conv trace D2H: {e}"))?,
            );
            let replay_vals = decode_bf16_le(
                &replay
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("replay persistent conv trace D2H: {e}"))?,
            );
            let delta = validate::max_abs_delta(&native_vals, &replay_vals);
            let first = native_vals
                .iter()
                .zip(replay_vals.iter())
                .enumerate()
                .find(|(_, (n, r))| (*n - *r).abs() > 0.0)
                .map(|(idx, (n, r))| (idx, *n, *r));
            (delta, first)
        }
        _ => (0.0, None),
    };
    let (rec_delta, first_rec_mismatch, max_rec_mismatch) = match (&native_layer.recurrent_state, &replay_layer.recurrent_state) {
        (Some(native), Some(replay)) => {
            let native_vals = decode_f32_le(
                &native
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("native persistent recurrent trace D2H: {e}"))?,
            );
            let replay_vals = decode_f32_le(
                &replay
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("replay persistent recurrent trace D2H: {e}"))?,
            );
            let delta = validate::max_abs_delta(&native_vals, &replay_vals);
            let first = native_vals
                .iter()
                .zip(replay_vals.iter())
                .enumerate()
                .find(|(_, (n, r))| (*n - *r).abs() > 0.0)
                .map(|(idx, (n, r))| (idx, *n, *r));
            let max_entry = native_vals
                .iter()
                .zip(replay_vals.iter())
                .enumerate()
                .max_by(|(_, (na, ra)), (_, (nb, rb))| {
                    (*na - *ra)
                        .abs()
                        .partial_cmp(&(*nb - *rb).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, (n, r))| (idx, *n, *r, (*n - *r).abs()));
            (delta, first, max_entry)
        }
        _ => (0.0, None, None),
    };
    eprintln!(
        "[trace-persistent-linear-state] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={rec_delta:.6}{}{}{}",
        first_conv_mismatch
            .map(|(idx, native, replay)| format!(
                " first_conv_mismatch=(idx={idx},native={native:.9},replay={replay:.9})"
            ))
            .unwrap_or_default(),
        first_rec_mismatch
            .map(|(idx, native, replay)| format!(
                " first_recurrent_mismatch=(idx={idx},native={native:.9},replay={replay:.9})"
            ))
            .unwrap_or_default(),
        max_rec_mismatch
            .map(|(idx, native, replay, delta)| format!(
                " max_recurrent_mismatch=(idx={idx},native={native:.9},replay={replay:.9},delta={delta:.9})"
            ))
            .unwrap_or_default()
    );
    Ok(())
}

fn trace_persistent_full_attn_layer(
    engine: &mut DecodeEngine,
    trace_layer: usize,
    token_ids: &[u32],
    trace_tokens: &[u32],
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let text_config = engine.weights().config.clone();
    anyhow::ensure!(
        text_config.is_full_attention(trace_layer),
        "layer {trace_layer} is not a full-attention layer"
    );
    anyhow::ensure!(trace_layer > 0, "trace layer must be > 0 for full-attention input tracing");

    let prefix_ids = token_ids
        .get(..token_ids.len().saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("missing prefix token ids for persistent full-attn trace"))?;
    engine.rebuild_prefill_state(prefix_ids, true)?;

    let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer,
        0,
    )?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let _ = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer + 1,
        0,
    )?;
    let native_gated = engine.trace_persistent_full_attention_gated_after_layers(0)?;
    let native_q = engine.trace_persistent_full_attention_q_after_layers(0)?;
    let native_saved_gate = engine.trace_persistent_full_attention_saved_gate_after_layers(0)?;
    let native_pre_gate = engine.trace_persistent_full_attention_pre_gate_after_layers(0)?;
    let native_scores = engine.trace_persistent_full_attention_scores_after_layers(0, seqlen_offset + 1)?;
    let (_, _, _, native_token_mixer) =
        engine.trace_persistent_mlp_stage_after_layers(0, text_config.intermediate_size)?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let native_component =
        engine.trace_full_attention_stages_from_hidden(trace_layer, &native_hidden, seqlen_offset)?;
    let native_component_layer = engine.trace_full_attention_layer_output_from_hidden_current_state(
        trace_layer,
        0,
        &native_hidden,
        seqlen_offset,
    )?;

    let mut replay_prefix_state = ModelState::new(&text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent full-attn replay prefix state init: {e}"))?;
    let _ = prefill_engine::prefill(
        engine.weights(),
        &mut replay_prefix_state,
        engine.rotary(),
        prefix_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let mut replay_state = ModelState::new(&text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent full-attn replay state init: {e}"))?;
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = replay
        .layer_hidden_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer - 1))
        .ok_or_else(|| anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}"))?;
    let replay_component =
        engine.trace_full_attention_stages_from_hidden(trace_layer, replay_hidden, seqlen_offset)?;
    let replay_cache_component_layer = engine.trace_full_attention_layer_output_from_hidden_state(
        &replay_prefix_state,
        trace_layer,
        &native_hidden,
        seqlen_offset,
    )?;

    engine.rebuild_prefill_state(prefix_ids, true)?;
    let _ = engine.decode_step_batch(trace_tokens, seqlen_offset)?;
    let native_hidden_f32 = decode_bf16_le(&native_hidden);
    let replay_hidden_f32 = decode_bf16_le(replay_hidden);
    let replay_attn_hidden = replay
        .layer_attn_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay attn trace for layer {trace_layer}"))?;
    let replay_attn_hidden_f32 = decode_bf16_le(replay_attn_hidden);
    let native_normed_f32 = decode_bf16_le(&native_component.normed);
    let replay_normed_f32 = decode_bf16_le(&replay_component.normed);
    let native_q_proj_f32 = decode_bf16_le(&native_component.q_proj);
    let replay_q_proj_f32 = decode_bf16_le(&replay_component.q_proj);
    let native_gate_proj_f32 = decode_bf16_le(&native_component.gate_proj);
    let replay_gate_proj_f32 = decode_bf16_le(&replay_component.gate_proj);
    let native_k_proj_f32 = decode_bf16_le(&native_component.k_proj);
    let replay_k_proj_f32 = decode_bf16_le(&replay_component.k_proj);
    let native_v_proj_f32 = decode_bf16_le(&native_component.v_proj);
    let replay_v_proj_f32 = decode_bf16_le(&replay_component.v_proj);
    let native_q_rope_f32 = decode_bf16_le(&native_component.q_rope);
    let replay_q_rope_f32 = decode_bf16_le(&replay_component.q_rope);
    let native_q_f32 = decode_f32_le(&native_q);
    let native_comp_k_f32 = decode_bf16_le(&native_component.k_rope);
    let native_comp_v_f32 = decode_bf16_le(&native_component.v_proj);
    let replay_comp_k_f32 = decode_bf16_le(&replay_component.k_rope);
    let replay_comp_v_f32 = decode_bf16_le(&replay_component.v_proj);
    let hidden_delta = validate::max_abs_delta(&native_hidden_f32, &replay_hidden_f32);
    let normed_delta = validate::max_abs_delta(&native_normed_f32, &replay_normed_f32);
    let q_proj_delta = validate::max_abs_delta(&native_q_proj_f32, &replay_q_proj_f32);
    let gate_proj_delta = validate::max_abs_delta(&native_gate_proj_f32, &replay_gate_proj_f32);
    let k_proj_delta = validate::max_abs_delta(&native_k_proj_f32, &replay_k_proj_f32);
    let v_proj_delta = validate::max_abs_delta(&native_v_proj_f32, &replay_v_proj_f32);
    let q_rope_delta = validate::max_abs_delta(&native_q_rope_f32, &replay_q_rope_f32);
    let native_vs_component_q = validate::max_abs_delta(&native_q_f32, &native_q_rope_f32);
    let native_vs_replay_k = validate::max_abs_delta(&native_comp_k_f32, &replay_comp_k_f32);
    let native_vs_replay_v = validate::max_abs_delta(&native_comp_v_f32, &replay_comp_v_f32);
    let native_gated_f32 = decode_f32_le(&native_gated);
    let full_weights = engine.weights().layers[trace_layer]
        .full
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} missing full-attention weights"))?;
    let q_dim = native_gated_f32.len();
    let native_gated_gpu = gpu_hal::GpuBuffer::from_host_bytes(
        ordinal,
        gpu_hal::ScalarType::BF16,
        &[1, q_dim],
        &f32_to_bf16_bytes(native_gated_f32.iter().copied()),
    )
    .map_err(|e| anyhow::anyhow!("trace native gated H2D: {e}"))?;
    let mut native_o_proj_gpu = gpu_hal::GpuBuffer::zeros(
        ordinal,
        gpu_hal::ScalarType::BF16,
        &[1, text_config.hidden_size],
    )
    .map_err(|e| anyhow::anyhow!("trace native o_proj alloc: {e}"))?;
    kernel_ffi::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        gpu_hal::ScalarType::BF16,
        1,
        1,
        text_config.hidden_size,
        q_dim,
        &native_gated_gpu,
        &full_weights.o_proj_w,
        &mut native_o_proj_gpu,
    )
    .map_err(|e| anyhow::anyhow!("trace native o_proj matmul: {e}"))?;
    let native_host_o_proj_f32 = decode_bf16_le(
        &native_o_proj_gpu
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace native o_proj D2H: {e}"))?,
    );
    let native_saved_gate_f32 = decode_f32_le(&native_saved_gate);
    let native_pre_gate_f32 = decode_f32_le(&native_pre_gate);
    let native_scores_f32 = decode_f32_le(&native_scores);
    let native_comp_gated_f32 = decode_bf16_le(&native_component_layer.gated);
    let native_comp_pre_gate_f32 = decode_bf16_le(&native_component_layer.pre_gate);
    let native_token_mixer_f32 = decode_f32_le(&native_token_mixer);
    let native_comp_token_mixer_f32 = decode_bf16_le(&native_component_layer.attn_hidden);
    let replay_cache_token_mixer_f32 = decode_bf16_le(&replay_cache_component_layer.attn_hidden);
    let mut kv_vs_bf16_pre_gate = None;
    let mut kv_vs_bf16_gated = None;
    let mut kv_vs_bf16_attn_hidden = None;
    let mut kv_vs_bf16_scores = None;
    let mut kv_vs_bf16_scores_heads = None;
    let mut kv_vs_bf16_hidden = None;
    let mut kv_vs_bf16_q = None;
    let mut kv_vs_bf16_saved_gate = None;
    let mut kv_vs_bf16_cache_k = None;
    let mut kv_vs_bf16_cache_v = None;
    if engine.kv_fp8_enabled() {
        let (native_cache_k_bf16, native_cache_v_bf16, _) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        engine.set_kv_fp8_for_trace(false);
        engine.rebuild_prefill_state(prefix_ids, true)?;
        let (bf16_cache_k_bf16, bf16_cache_v_bf16, _) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        let bf16_hidden = decode_bf16_le(&engine.decode_step_batch_trace_hidden_after_layers(
            trace_tokens,
            seqlen_offset,
            trace_layer,
            0,
        )?);
        let _ = engine.decode_step_batch_trace_hidden_after_layers(
            trace_tokens,
            seqlen_offset,
            trace_layer + 1,
            0,
        )?;
        let bf16_q = decode_f32_le(&engine.trace_persistent_full_attention_q_after_layers(0)?);
        let bf16_saved_gate =
            decode_f32_le(&engine.trace_persistent_full_attention_saved_gate_after_layers(0)?);
        let bf16_gated = decode_f32_le(&engine.trace_persistent_full_attention_gated_after_layers(0)?);
        let bf16_pre_gate =
            decode_f32_le(&engine.trace_persistent_full_attention_pre_gate_after_layers(0)?);
        let bf16_scores =
            decode_f32_le(&engine.trace_persistent_full_attention_scores_after_layers(0, seqlen_offset + 1)?);
        let (_, _, _, bf16_token_mixer) =
            engine.trace_persistent_mlp_stage_after_layers(0, text_config.intermediate_size)?;
        let bf16_token_mixer_f32 = decode_f32_le(&bf16_token_mixer);
        kv_vs_bf16_pre_gate = Some(validate::max_abs_delta(&native_pre_gate_f32, &bf16_pre_gate));
        kv_vs_bf16_gated = Some(validate::max_abs_delta(&native_gated_f32, &bf16_gated));
        kv_vs_bf16_attn_hidden =
            Some(validate::max_abs_delta(&native_token_mixer_f32, &bf16_token_mixer_f32));
        kv_vs_bf16_scores = Some(validate::max_abs_delta(&native_scores_f32, &bf16_scores));
        kv_vs_bf16_hidden = Some(validate::max_abs_delta(&native_hidden_f32, &bf16_hidden));
        kv_vs_bf16_q = Some(validate::max_abs_delta(&native_q_f32, &bf16_q));
        kv_vs_bf16_saved_gate =
            Some(validate::max_abs_delta(&native_saved_gate_f32, &bf16_saved_gate));
        kv_vs_bf16_cache_k = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_cache_k_bf16),
            &decode_bf16_le(&bf16_cache_k_bf16),
        ));
        kv_vs_bf16_cache_v = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_cache_v_bf16),
            &decode_bf16_le(&bf16_cache_v_bf16),
        ));
        let score_cols = seqlen_offset + 1;
        kv_vs_bf16_scores_heads = Some(
            (0..text_config.num_attention_heads)
                .map(|h| {
                    let start = h * score_cols;
                    let end = start + score_cols;
                    validate::max_abs_delta(&native_scores_f32[start..end], &bf16_scores[start..end])
                })
                .collect::<Vec<_>>()
        );
        engine.set_kv_fp8_for_trace(true);
        engine.rebuild_prefill_state(prefix_ids, true)?;
    }
    let native_state = engine.state_for_batch(0);
    let native_layer = native_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native layer {trace_layer}"))?;
    let native_vs_component_attn_hidden =
        validate::max_abs_delta(&native_token_mixer_f32, &native_comp_token_mixer_f32);
    let native_vs_host_o_proj =
        validate::max_abs_delta(&native_token_mixer_f32, &native_host_o_proj_f32);
    let native_vs_component_gated =
        validate::max_abs_delta(&native_gated_f32, &native_comp_gated_f32);
    let native_vs_component_saved_gate =
        validate::max_abs_delta(&native_saved_gate_f32, &native_gate_proj_f32);
    let native_vs_component_pre_gate =
        validate::max_abs_delta(&native_pre_gate_f32, &native_comp_pre_gate_f32);
    let head_dim = engine.weights().config.head_dim;
    let num_q_heads = engine.weights().config.num_attention_heads;
    let per_head_pre_gate = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            validate::max_abs_delta(&native_pre_gate_f32[start..end], &native_comp_pre_gate_f32[start..end])
        })
        .collect::<Vec<_>>();
    let per_head_pre_gate_str = per_head_pre_gate
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let per_head_q = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            validate::max_abs_delta(&native_q_f32[start..end], &native_q_rope_f32[start..end])
        })
        .collect::<Vec<_>>();
    let per_head_q_str = per_head_q
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let (score_row_delta, per_head_score_str) = if let (Some(scale_k), Some(k_cache)) = (
        native_layer.kv_scale_k.as_ref(),
        native_layer.kv_cache_k.as_ref(),
    ) {
        let hd = engine.weights().config.head_dim;
        let num_q_heads = engine.weights().config.num_attention_heads;
        let num_kv_heads = engine.weights().config.num_key_value_heads;
        let max_t = k_cache.shape()[2];
        let k_bytes = k_cache
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace native K cache D2H: {e}"))?;
        let k_scales = decode_f32_le(
            &scale_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("trace native K scale D2H: {e}"))?,
        );
        let kv_groups = num_q_heads / num_kv_heads;
        let mut host_scores = Vec::with_capacity(num_q_heads * (seqlen_offset + 1));
        let mut per_head_score = Vec::with_capacity(num_q_heads);
        for qh in 0..num_q_heads {
            let kvh = qh / kv_groups;
            let q_head = &native_q_f32[qh * hd..(qh + 1) * hd];
            let row_start = host_scores.len();
            for t in 0..=seqlen_offset {
                let scale_val = k_scales[kvh * max_t + t];
                let base = (kvh * max_t + t) * hd;
                let mut acc = 0.0f32;
                for d in 0..hd {
                    let k_val = half::bf16::from_f32(
                        fp8_e4m3_to_f32_host(k_bytes[base + d]) * scale_val
                    )
                    .to_f32();
                    acc += q_head[d] * k_val;
                }
                host_scores.push(acc / (hd as f32).sqrt());
            }
            let row_end = host_scores.len();
            per_head_score.push(validate::max_abs_delta(
                &native_scores_f32[row_start..row_end],
                &host_scores[row_start..row_end],
            ));
        }
        (
            validate::max_abs_delta(&native_scores_f32, &host_scores),
            per_head_score
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(","),
        )
    } else {
        (0.0, String::new())
    };
    let native_vs_replay_attn_hidden =
        validate::max_abs_delta(&native_token_mixer_f32, &replay_attn_hidden_f32);
    let native_cache_vs_replay_cache_attn_hidden =
        validate::max_abs_delta(&native_comp_token_mixer_f32, &replay_cache_token_mixer_f32);
    let component_vs_replay_attn_hidden =
        validate::max_abs_delta(&native_comp_token_mixer_f32, &replay_attn_hidden_f32);

    if let (Some(scale_k), Some(scale_v), Some(k_cache), Some(_v_cache)) = (
        native_layer.kv_scale_k.as_ref(),
        native_layer.kv_scale_v.as_ref(),
        native_layer.kv_cache_k.as_ref(),
        native_layer.kv_cache_v.as_ref(),
    ) {
        let nkv = engine.weights().config.num_key_value_heads;
        let hd = engine.weights().config.head_dim;
        let max_t = k_cache.shape()[2];

        let src_k = gpu_hal::GpuBuffer::from_host_bytes(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &[nkv, 1, hd],
            &native_component.k_rope,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp K H2D: {e}"))?;
        let src_v = gpu_hal::GpuBuffer::from_host_bytes(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &[nkv, 1, hd],
            &native_component.v_proj,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp V H2D: {e}"))?;
        let mut tmp_k_fp8 = gpu_hal::GpuBuffer::zeros(
            ordinal,
            gpu_hal::ScalarType::U8,
            &[nkv, max_t, hd],
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp K cache alloc: {e}"))?;
        let mut tmp_v_fp8 = gpu_hal::GpuBuffer::zeros(
            ordinal,
            gpu_hal::ScalarType::U8,
            &[nkv, max_t, hd],
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp V cache alloc: {e}"))?;
        let mut tmp_k_scale = gpu_hal::GpuBuffer::zeros(
            ordinal,
            gpu_hal::ScalarType::F32,
            &[nkv, max_t],
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp K scale alloc: {e}"))?;
        let mut tmp_v_scale = gpu_hal::GpuBuffer::zeros(
            ordinal,
            gpu_hal::ScalarType::F32,
            &[nkv, max_t],
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 temp V scale alloc: {e}"))?;
        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &src_k,
            &mut tmp_k_fp8,
            &mut tmp_k_scale,
            nkv,
            1,
            hd,
            max_t,
            seqlen_offset,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 quantize K: {e}"))?;
        kernel_ffi::prefill_ffi::quantize_kv_to_fp8(
            ordinal,
            gpu_hal::ScalarType::BF16,
            &src_v,
            &mut tmp_v_fp8,
            &mut tmp_v_scale,
            nkv,
            1,
            hd,
            max_t,
            seqlen_offset,
        )
        .map_err(|e| anyhow::anyhow!("trace fp8 quantize V: {e}"))?;

        let tmp_k_bytes = tmp_k_fp8
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp K D2H: {e}"))?;
        let tmp_v_bytes = tmp_v_fp8
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp V D2H: {e}"))?;
        let tmp_k_scale_bytes = tmp_k_scale
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp K scale D2H: {e}"))?;
        let tmp_v_scale_bytes = tmp_v_scale
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 temp V scale D2H: {e}"))?;
        let native_k_cache_bytes = k_cache
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native K cache D2H: {e}"))?;
        let native_v_cache_bytes = native_layer
            .kv_cache_v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing native V cache layer {trace_layer}"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native V cache D2H: {e}"))?;
        let native_k_scale_bytes = scale_k
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native K scale D2H: {e}"))?;
        let native_v_scale_bytes = scale_v
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace fp8 native V scale D2H: {e}"))?;

        let head_span = max_t * hd;
        let kv_groups = num_q_heads / nkv;
        let mut native_k_step = Vec::with_capacity(nkv * hd);
        let mut native_v_step = Vec::with_capacity(nkv * hd);
        let mut quant_k_step = Vec::with_capacity(nkv * hd);
        let mut quant_v_step = Vec::with_capacity(nkv * hd);
        for h in 0..nkv {
            let base = h * head_span + seqlen_offset * hd;
            native_k_step.extend_from_slice(&native_k_cache_bytes[base..base + hd]);
            native_v_step.extend_from_slice(&native_v_cache_bytes[base..base + hd]);
            quant_k_step.extend_from_slice(&tmp_k_bytes[base..base + hd]);
            quant_v_step.extend_from_slice(&tmp_v_bytes[base..base + hd]);
        }
        let native_k_scales = decode_f32_le(&native_k_scale_bytes);
        let native_v_scales = decode_f32_le(&native_v_scale_bytes);
        let quant_k_scales = decode_f32_le(&tmp_k_scale_bytes);
        let quant_v_scales = decode_f32_le(&tmp_v_scale_bytes);
        let mut native_k_scale_step = Vec::with_capacity(nkv);
        let mut native_v_scale_step = Vec::with_capacity(nkv);
        let mut quant_k_scale_step = Vec::with_capacity(nkv);
        let mut quant_v_scale_step = Vec::with_capacity(nkv);
        for h in 0..nkv {
            native_k_scale_step.push(native_k_scales[h * max_t + seqlen_offset]);
            native_v_scale_step.push(native_v_scales[h * max_t + seqlen_offset]);
            quant_k_scale_step.push(quant_k_scales[h * max_t + seqlen_offset]);
            quant_v_scale_step.push(quant_v_scales[h * max_t + seqlen_offset]);
        }
        let cache_vs_quant_k = native_k_step
            .iter()
            .zip(quant_k_step.iter())
            .filter(|(n, q)| n != q)
            .count();
        let cache_vs_quant_v = native_v_step
            .iter()
            .zip(quant_v_step.iter())
            .filter(|(n, q)| n != q)
            .count();
        let scale_vs_quant_k = validate::max_abs_delta(&native_k_scale_step, &quant_k_scale_step);
        let scale_vs_quant_v = validate::max_abs_delta(&native_v_scale_step, &quant_v_scale_step);
        let mut host_pre_gate = vec![0.0f32; num_q_heads * hd];
        for qh in 0..num_q_heads {
            let kvh = qh / kv_groups;
            let row = &native_scores_f32[qh * (seqlen_offset + 1)..(qh + 1) * (seqlen_offset + 1)];
            let row_max = row
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut weights = vec![0.0f32; row.len()];
            for (idx, score) in row.iter().copied().enumerate() {
                let w = (score - row_max).exp();
                weights[idx] = w;
                denom += w;
            }
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    let scale_val = native_v_scales[kvh * max_t + t];
                    let base = (kvh * max_t + t) * hd + d;
                    let v_val = half::bf16::from_f32(
                        fp8_e4m3_to_f32_host(native_v_cache_bytes[base]) * scale_val
                    )
                    .to_f32();
                    acc += w * v_val;
                }
                host_pre_gate[qh * hd + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let native_vs_host_pre_gate =
            validate::max_abs_delta(&native_pre_gate_f32, &host_pre_gate);
        let per_head_host_pre_gate = (0..num_q_heads)
            .map(|h| {
                let start = h * hd;
                let end = start + hd;
                validate::max_abs_delta(&native_pre_gate_f32[start..end], &host_pre_gate[start..end])
            })
            .collect::<Vec<_>>();
        let per_head_host_pre_gate_str = per_head_host_pre_gate
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        let host_gated = host_pre_gate
            .iter()
            .zip(native_saved_gate_f32.iter())
            .map(|(x, g)| x / (1.0 + (-g).exp()))
            .collect::<Vec<_>>();
        let native_vs_host_gated = validate::max_abs_delta(&native_gated_f32, &host_gated);
        let kv_vs_bf16_pre_gate = kv_vs_bf16_pre_gate.unwrap_or(0.0);
        let kv_vs_bf16_gated = kv_vs_bf16_gated.unwrap_or(0.0);
        let kv_vs_bf16_attn_hidden = kv_vs_bf16_attn_hidden.unwrap_or(0.0);
        let kv_vs_bf16_scores = kv_vs_bf16_scores.unwrap_or(0.0);
        let kv_vs_bf16_hidden = kv_vs_bf16_hidden.unwrap_or(0.0);
        let kv_vs_bf16_q = kv_vs_bf16_q.unwrap_or(0.0);
        let kv_vs_bf16_saved_gate = kv_vs_bf16_saved_gate.unwrap_or(0.0);
        let kv_vs_bf16_cache_k = kv_vs_bf16_cache_k.unwrap_or(0.0);
        let kv_vs_bf16_cache_v = kv_vs_bf16_cache_v.unwrap_or(0.0);
        let kv_vs_bf16_scores_heads_str = kv_vs_bf16_scores_heads
            .as_ref()
            .map(|vals| vals.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(","))
            .unwrap_or_default();
        eprintln!(
            "[trace-persistent-full-attn] layer={trace_layer} hidden_delta={hidden_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} native_vs_component_q={native_vs_component_q:.6} per_head_q=[{per_head_q_str}] native_comp_vs_replay_k={native_vs_replay_k:.6} native_comp_vs_replay_v={native_vs_replay_v:.6} native_vs_component_saved_gate={native_vs_component_saved_gate:.6} native_vs_component_pre_gate={native_vs_component_pre_gate:.6} native_vs_host_pre_gate={native_vs_host_pre_gate:.6} kv_vs_bf16_hidden={kv_vs_bf16_hidden:.6} kv_vs_bf16_cache_k={kv_vs_bf16_cache_k:.6} kv_vs_bf16_cache_v={kv_vs_bf16_cache_v:.6} kv_vs_bf16_q={kv_vs_bf16_q:.6} kv_vs_bf16_saved_gate={kv_vs_bf16_saved_gate:.6} kv_vs_bf16_scores={kv_vs_bf16_scores:.6} kv_vs_bf16_scores_heads=[{kv_vs_bf16_scores_heads_str}] kv_vs_bf16_pre_gate={kv_vs_bf16_pre_gate:.6} per_head_host_pre_gate=[{per_head_host_pre_gate_str}] native_score_row_delta={score_row_delta:.6} per_head_score=[{per_head_score_str}] native_vs_component_gated={native_vs_component_gated:.6} native_vs_host_gated={native_vs_host_gated:.6} kv_vs_bf16_gated={kv_vs_bf16_gated:.6} native_vs_component_attn_hidden={native_vs_component_attn_hidden:.6} native_vs_host_o_proj={native_vs_host_o_proj:.6} kv_vs_bf16_attn_hidden={kv_vs_bf16_attn_hidden:.6} native_vs_replay_attn_hidden={native_vs_replay_attn_hidden:.6} native_cache_vs_replay_cache_attn_hidden={native_cache_vs_replay_cache_attn_hidden:.6} component_vs_replay_attn_hidden={component_vs_replay_attn_hidden:.6} per_head_pre_gate=[{per_head_pre_gate_str}] cache_vs_quant_k_mismatches={cache_vs_quant_k} cache_vs_quant_v_mismatches={cache_vs_quant_v} cache_vs_quant_k_scale_delta={scale_vs_quant_k:.6} cache_vs_quant_v_scale_delta={scale_vs_quant_v:.6}"
        );
    } else {
        let native_cache = engine.full_attention_cache_step_bytes(trace_layer, 0, seqlen_offset)?;
        let native_cache_k_f32 = decode_bf16_le(&native_cache.0);
        let native_cache_v_f32 = decode_bf16_le(&native_cache.1);
        let cache_vs_component_k = validate::max_abs_delta(&native_cache_k_f32, &native_comp_k_f32);
        let cache_vs_component_v = validate::max_abs_delta(&native_cache_v_f32, &native_comp_v_f32);
        let cache_vs_replay_k = validate::max_abs_delta(&native_cache_k_f32, &replay_comp_k_f32);
        let cache_vs_replay_v = validate::max_abs_delta(&native_cache_v_f32, &replay_comp_v_f32);
        eprintln!(
            "[trace-persistent-full-attn] layer={trace_layer} hidden_delta={hidden_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} native_vs_component_q={native_vs_component_q:.6} per_head_q=[{per_head_q_str}] native_comp_vs_replay_k={native_vs_replay_k:.6} native_comp_vs_replay_v={native_vs_replay_v:.6} native_vs_component_saved_gate={native_vs_component_saved_gate:.6} native_vs_component_pre_gate={native_vs_component_pre_gate:.6} native_score_row_delta={score_row_delta:.6} per_head_score=[{per_head_score_str}] native_vs_component_gated={native_vs_component_gated:.6} native_vs_component_attn_hidden={native_vs_component_attn_hidden:.6} native_vs_host_o_proj={native_vs_host_o_proj:.6} native_vs_replay_attn_hidden={native_vs_replay_attn_hidden:.6} native_cache_vs_replay_cache_attn_hidden={native_cache_vs_replay_cache_attn_hidden:.6} component_vs_replay_attn_hidden={component_vs_replay_attn_hidden:.6} per_head_pre_gate=[{per_head_pre_gate_str}] cache_vs_component_k={cache_vs_component_k:.6} cache_vs_component_v={cache_vs_component_v:.6} cache_vs_replay_k={cache_vs_replay_k:.6} cache_vs_replay_v={cache_vs_replay_v:.6}"
        );
    }
    Ok(())
}

fn trace_persistent_linear_layer(
    engine: &mut DecodeEngine,
    trace_layer: usize,
    token_ids: &[u32],
    trace_tokens: &[u32],
    seqlen_offset: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let text_config = engine.weights().config.clone();
    anyhow::ensure!(
        !text_config.is_full_attention(trace_layer),
        "layer {trace_layer} is not a linear-attention layer"
    );

    let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer,
        0,
    )?;

    let mut replay_state = ModelState::new(&text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("persistent linear replay state init: {e}"))?;
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let replay_hidden = if trace_layer == 0 {
        None
    } else {
        Some(
            replay
                .layer_hidden_trace
                .as_ref()
                .and_then(|layers| layers.get(trace_layer - 1))
                .ok_or_else(|| anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}"))?,
        )
    };
    let replay_layer = replay_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing replay layer {trace_layer}"))?;
    let replay_conv = replay_layer
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing replay conv state for layer {trace_layer}"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay conv D2H layer {trace_layer}: {e}"))?;
    let replay_recurrent = replay_layer
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing replay recurrent state for layer {trace_layer}"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay recurrent D2H layer {trace_layer}: {e}"))?;
    let replay_hidden_out = replay
        .layer_hidden_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay output hidden trace for layer {trace_layer}"))?;
    let replay_attn = replay
        .layer_attn_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay attn trace for layer {trace_layer}"))?;
    let replay_post = replay
        .layer_post_attn_norm_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay post-attn trace for layer {trace_layer}"))?;
    let replay_swiglu = replay
        .layer_mlp_swiglu_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay swiglu trace for layer {trace_layer}"))?;
    let replay_mlp_out = replay
        .layer_mlp_out_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay mlp-out trace for layer {trace_layer}"))?;

    let prefix_ids = token_ids
        .get(..token_ids.len().saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("missing prefix token ids for persistent linear trace"))?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    engine.set_hidden_from_bytes(&native_hidden)?;
    let (native_comp_trace, native_comp_conv, native_comp_recurrent, native_comp_hidden) =
        engine.component_trace_linear_layer_from_current_hidden(trace_layer)?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    engine.set_hidden_from_bytes(&native_hidden)?;
    let native_comp_layer = engine.component_trace_full_layer_from_current_hidden(trace_layer)?;

    engine.rebuild_prefill_state(prefix_ids, true)?;
    let native_hidden_out = engine
        .decode_step_batch_trace_hidden_after_layers(trace_tokens, seqlen_offset, trace_layer + 1, 0)?;
    let cfg = engine.weights().config.clone();
    let qkv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let z_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let val_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let nv = cfg.linear_num_value_heads;
    let intermediate = cfg.intermediate_size;
    let (native_qkv_proj, native_z_proj, native_b_proj, native_a_proj) =
        engine.trace_persistent_linear_proj_buf_after_layers(0, qkv_dim, z_dim, nv)?;
    let native_gated = engine.trace_persistent_linear_gated_after_layers(0, val_dim)?;
    let (native_post_norm, native_swiglu, native_mlp_down, native_token_mixer) =
        engine.trace_persistent_mlp_stage_after_layers(0, intermediate)?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let _ = engine.decode_step_batch(trace_tokens, seqlen_offset)?;
    let native_layer = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native layer {trace_layer} after decode"))?;
    let native_conv = native_layer
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing native conv state for layer {trace_layer}"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native conv D2H layer {trace_layer}: {e}"))?;
    let native_recurrent = native_layer
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing native recurrent state for layer {trace_layer}"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native recurrent D2H layer {trace_layer}: {e}"))?;
    let hidden_delta = replay_hidden
        .map(|replay_hidden| {
            validate::max_abs_delta(&decode_bf16_le(&native_hidden), &decode_bf16_le(replay_hidden))
        })
        .unwrap_or(0.0);
    let replay_comp_trace = if let Some(replay_hidden) = replay_hidden {
        engine.rebuild_prefill_state(prefix_ids, true)?;
        engine.set_hidden_from_bytes(replay_hidden)?;
        Some(engine.component_trace_linear_layer_from_current_hidden(trace_layer)?)
    } else {
        None
    };
    let comp_vs_replay_conv = validate::max_abs_delta(&decode_bf16_le(&native_comp_conv), &decode_bf16_le(&replay_conv));
    let comp_vs_replay_recurrent =
        validate::max_abs_delta(&decode_f32_le(&native_comp_recurrent), &decode_f32_le(&replay_recurrent));
    let comp_vs_replay_hidden = validate::max_abs_delta(&decode_bf16_le(&native_comp_hidden), &decode_bf16_le(replay_hidden_out));
    let native_vs_comp_conv = validate::max_abs_delta(&decode_bf16_le(&native_conv), &decode_bf16_le(&native_comp_conv));
    let native_vs_comp_recurrent =
        validate::max_abs_delta(&decode_f32_le(&native_recurrent), &decode_f32_le(&native_comp_recurrent));
    let native_vs_comp_proj_residual = validate::max_abs_delta(
        &decode_bf16_le(&native_hidden_out),
        &bf16_residual_sum(&native_hidden, &native_comp_trace.proj_out),
    );
    let native_vs_comp_qkv_proj =
        validate::max_abs_delta(&decode_f32_le(&native_qkv_proj), &decode_bf16_le(&native_comp_trace.qkv));
    let native_vs_comp_z_proj =
        validate::max_abs_delta(&decode_f32_le(&native_z_proj), &decode_bf16_le(&native_comp_trace.z));
    let native_vs_comp_b_proj =
        validate::max_abs_delta(&decode_f32_le(&native_b_proj), &decode_bf16_le(&native_comp_trace.b));
    let native_vs_comp_a_proj =
        validate::max_abs_delta(&decode_f32_le(&native_a_proj), &decode_bf16_le(&native_comp_trace.a));
    let native_vs_comp_post_norm =
        validate::max_abs_delta(&decode_f32_le(&native_post_norm), &decode_bf16_le(&native_comp_layer.post_attn_norm));
    let native_vs_comp_gated =
        validate::max_abs_delta(&decode_f32_le(&native_gated), &decode_bf16_le(&native_comp_trace.gated));
    let native_vs_comp_swiglu =
        validate::max_abs_delta(&decode_f32_le(&native_swiglu), &decode_bf16_le(&native_comp_layer.mlp_swiglu));
    let native_vs_comp_token_mixer =
        validate::max_abs_delta(&decode_f32_le(&native_token_mixer), &decode_bf16_le(&native_comp_layer.attn_hidden));
    let native_vs_comp_mlp_down =
        validate::max_abs_delta(&decode_f32_le(&native_mlp_down), &decode_bf16_le(&native_comp_layer.mlp_out));
    let native_vs_replay_post_norm =
        validate::max_abs_delta(&decode_f32_le(&native_post_norm), &decode_bf16_le(replay_post));
    let native_vs_replay_gated =
        replay_comp_trace
            .as_ref()
            .map(|trace| validate::max_abs_delta(&decode_f32_le(&native_gated), &decode_bf16_le(&trace.0.gated)))
            .unwrap_or(0.0);
    let native_vs_replay_swiglu =
        validate::max_abs_delta(&decode_f32_le(&native_swiglu), &decode_bf16_le(replay_swiglu));
    let native_vs_replay_token_mixer =
        validate::max_abs_delta(&decode_f32_le(&native_token_mixer), &decode_bf16_le(replay_attn));
    let native_vs_replay_mlp_down =
        validate::max_abs_delta(&decode_f32_le(&native_mlp_down), &decode_bf16_le(replay_mlp_out));
    let native_vs_replay_qkv_proj = replay_comp_trace
        .as_ref()
        .map(|trace| validate::max_abs_delta(&decode_f32_le(&native_qkv_proj), &decode_bf16_le(&trace.0.qkv)))
        .unwrap_or(0.0);
    let native_vs_replay_z_proj = replay_comp_trace
        .as_ref()
        .map(|trace| validate::max_abs_delta(&decode_f32_le(&native_z_proj), &decode_bf16_le(&trace.0.z)))
        .unwrap_or(0.0);
    let native_vs_replay_b_proj = replay_comp_trace
        .as_ref()
        .map(|trace| validate::max_abs_delta(&decode_f32_le(&native_b_proj), &decode_bf16_le(&trace.0.b)))
        .unwrap_or(0.0);
    let native_vs_replay_a_proj = replay_comp_trace
        .as_ref()
        .map(|trace| validate::max_abs_delta(&decode_f32_le(&native_a_proj), &decode_bf16_le(&trace.0.a)))
        .unwrap_or(0.0);
    let comp_layer_vs_replay_hidden =
        validate::max_abs_delta(&decode_bf16_le(&native_comp_layer.layer_hidden), &decode_bf16_le(replay_hidden_out));
    let native_vs_comp_layer_hidden =
        validate::max_abs_delta(&decode_bf16_le(&native_hidden_out), &decode_bf16_le(&native_comp_layer.layer_hidden));
    let native_vs_replay_hidden =
        validate::max_abs_delta(&decode_bf16_le(&native_hidden_out), &decode_bf16_le(replay_hidden_out));
    let native_qkv_proj_f32 = decode_f32_le(&native_qkv_proj);
    let native_z_proj_f32 = decode_f32_le(&native_z_proj);
    let comp_qkv_proj_f32 = decode_bf16_le(&native_comp_trace.qkv);
    let comp_z_proj_f32 = decode_bf16_le(&native_comp_trace.z);
    let sample_qkv_native = native_qkv_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    let sample_qkv_comp = comp_qkv_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    let sample_z_native = native_z_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    let sample_z_comp = comp_z_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");

    eprintln!(
        "[trace-persistent-linear] layer={trace_layer} hidden_delta={hidden_delta:.6} comp_vs_replay_conv={comp_vs_replay_conv:.6} comp_vs_replay_recurrent={comp_vs_replay_recurrent:.6} comp_linear_hidden_vs_replay={comp_vs_replay_hidden:.6} native_vs_comp_qkv_proj={native_vs_comp_qkv_proj:.6} native_vs_replay_qkv_proj={native_vs_replay_qkv_proj:.6} native_vs_comp_z_proj={native_vs_comp_z_proj:.6} native_vs_replay_z_proj={native_vs_replay_z_proj:.6} native_vs_comp_b_proj={native_vs_comp_b_proj:.6} native_vs_replay_b_proj={native_vs_replay_b_proj:.6} native_vs_comp_a_proj={native_vs_comp_a_proj:.6} native_vs_replay_a_proj={native_vs_replay_a_proj:.6} native_vs_comp_conv={native_vs_comp_conv:.6} native_vs_comp_recurrent={native_vs_comp_recurrent:.6} native_vs_comp_token_mixer={native_vs_comp_token_mixer:.6} native_vs_replay_token_mixer={native_vs_replay_token_mixer:.6} native_vs_comp_post_norm={native_vs_comp_post_norm:.6} native_vs_replay_post_norm={native_vs_replay_post_norm:.6} native_vs_comp_gated={native_vs_comp_gated:.6} native_vs_replay_gated={native_vs_replay_gated:.6} native_vs_comp_swiglu={native_vs_comp_swiglu:.6} native_vs_replay_swiglu={native_vs_replay_swiglu:.6} native_vs_comp_mlp_down={native_vs_comp_mlp_down:.6} native_vs_replay_mlp_down={native_vs_replay_mlp_down:.6} native_vs_comp_proj_residual={native_vs_comp_proj_residual:.6} comp_layer_hidden_vs_replay={comp_layer_vs_replay_hidden:.6} native_vs_comp_layer_hidden={native_vs_comp_layer_hidden:.6} native_vs_replay_hidden={native_vs_replay_hidden:.6} sample_qkv_native=[{sample_qkv_native}] sample_qkv_comp=[{sample_qkv_comp}] sample_z_native=[{sample_z_native}] sample_z_comp=[{sample_z_comp}]"
    );
    Ok(())
}

fn trace_component_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    native: &decode_engine::ComponentLayerTrace,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("component layer trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        true,
        None,
    )?;
    let attn = replay
        .layer_attn_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay attn trace for layer {trace_layer}"))?;
    let post = replay
        .layer_post_attn_norm_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay post-attn trace for layer {trace_layer}"))?;
    let mlp = replay
        .layer_mlp_out_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay mlp trace for layer {trace_layer}"))?;
    let hidden = replay
        .layer_hidden_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer))
        .ok_or_else(|| anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}"))?;
    let attn_delta = validate::max_abs_delta(&decode_bf16_le(&native.attn_hidden), &decode_bf16_le(attn));
    let post_delta = validate::max_abs_delta(&decode_bf16_le(&native.post_attn_norm), &decode_bf16_le(post));
    let mlp_delta = validate::max_abs_delta(&decode_bf16_le(&native.mlp_out), &decode_bf16_le(mlp));
    let hidden_delta = validate::max_abs_delta(&decode_bf16_le(&native.layer_hidden), &decode_bf16_le(hidden));
    eprintln!(
        "[trace-component-layer] layer={trace_layer} attn_delta={attn_delta:.6} post_norm_delta={post_delta:.6} mlp_delta={mlp_delta:.6} hidden_delta={hidden_delta:.6}"
    );
    Ok(())
}

fn trace_component_linear_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    native: &decode_engine::ComponentLinearTrace,
    token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let replay = prefill_engine::prefill(
        engine.weights(),
        &mut ModelState::new(&engine.weights().config, ordinal)
            .map_err(|e| anyhow::anyhow!("component linear trace replay state init: {e}"))?,
        engine.rotary(),
        token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        Some(trace_layer),
    )?;
    let replay = replay
        .linear_debug_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing replay linear trace for layer {trace_layer}"))?;
    let qkv_delta = validate::max_abs_delta(&decode_bf16_le(&native.qkv), &decode_bf16_le(&replay.qkv));
    let z_delta = validate::max_abs_delta(&decode_bf16_le(&native.z), &decode_bf16_le(&replay.z));
    let packed_native = decode_f32_le(&native.packed);
    let packed_replay = decode_f32_le(&replay.packed);
    let packed_delta = validate::max_abs_delta(&packed_native, &packed_replay);
    let cfg = &engine.weights().config;
    let nv = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;
    let packed_width = 2 * khd + vhd + 2;
    let mut q_delta = 0.0f32;
    let mut k_delta = 0.0f32;
    let mut v_delta = 0.0f32;
    let mut beta_delta = 0.0f32;
    let mut gexp_delta = 0.0f32;
    let v_ref = build_linear_decode_v_reference(engine, trace_layer, &native.qkv)?;
    let mut v_ref_native_delta = 0.0f32;
    let mut v_ref_replay_delta = 0.0f32;
    let mut state_vs_tail_delta = 0.0f32;
    if !replay.qkv_tail.is_empty() {
        let state = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing state for layer {trace_layer}"))?;
        let conv_state = decode_bf16_le(
            &state
                .conv_state
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} missing conv_state"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("trace conv_state D2H: {e}"))?,
        );
        let qkv_tail = decode_bf16_le(&replay.qkv_tail);
        let qkv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let state_len = cfg.linear_conv_kernel_dim - 1;
        let mut expected = vec![0.0f32; qkv_dim * state_len];
        for t in 0..state_len {
            for c in 0..qkv_dim {
                expected[c * state_len + t] = qkv_tail[t * qkv_dim + c];
            }
        }
        state_vs_tail_delta = validate::max_abs_delta(&conv_state, &expected);
    }
    for h in 0..nv {
        let base = h * packed_width;
        q_delta = q_delta.max(validate::max_abs_delta(
            &packed_native[base..base + khd],
            &packed_replay[base..base + khd],
        ));
        k_delta = k_delta.max(validate::max_abs_delta(
            &packed_native[base + khd..base + 2 * khd],
            &packed_replay[base + khd..base + 2 * khd],
        ));
        v_delta = v_delta.max(validate::max_abs_delta(
            &packed_native[base + 2 * khd..base + 2 * khd + vhd],
            &packed_replay[base + 2 * khd..base + 2 * khd + vhd],
        ));
        let v_ref_base = h * vhd;
        v_ref_native_delta = v_ref_native_delta.max(validate::max_abs_delta(
            &packed_native[base + 2 * khd..base + 2 * khd + vhd],
            &v_ref[v_ref_base..v_ref_base + vhd],
        ));
        v_ref_replay_delta = v_ref_replay_delta.max(validate::max_abs_delta(
            &packed_replay[base + 2 * khd..base + 2 * khd + vhd],
            &v_ref[v_ref_base..v_ref_base + vhd],
        ));
        beta_delta = beta_delta.max(
            (packed_native[base + 2 * khd + vhd] - packed_replay[base + 2 * khd + vhd]).abs(),
        );
        gexp_delta = gexp_delta.max(
            (packed_native[base + 2 * khd + vhd + 1] - packed_replay[base + 2 * khd + vhd + 1]).abs(),
        );
    }
    let rec_apply_delta =
        validate::max_abs_delta(&decode_f32_le(&native.rec_apply), &decode_f32_le(&replay.rec_apply));
    let attn_delta = validate::max_abs_delta(&decode_bf16_le(&native.attn), &decode_bf16_le(&replay.attn));
    let gated_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.gated), &decode_bf16_le(&replay.gated));
    let proj_out_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.proj_out),
        &decode_bf16_le(&replay.proj_out),
    );
    eprintln!(
        "[trace-component-linear] layer={trace_layer} qkv_delta={qkv_delta:.6} z_delta={z_delta:.6} packed_delta={packed_delta:.6} q_delta={q_delta:.6} k_delta={k_delta:.6} v_delta={v_delta:.6} state_vs_tail_delta={state_vs_tail_delta:.6} v_ref_native_delta={v_ref_native_delta:.6} v_ref_replay_delta={v_ref_replay_delta:.6} beta_delta={beta_delta:.6} gexp_delta={gexp_delta:.6} rec_apply_delta={rec_apply_delta:.6} attn_delta={attn_delta:.6} gated_delta={gated_delta:.6} proj_out_delta={proj_out_delta:.6}"
    );
    Ok(())
}

fn build_linear_decode_v_reference(
    engine: &DecodeEngine,
    trace_layer: usize,
    qkv_bytes: &[u8],
) -> Result<Vec<f32>> {
    let cfg = &engine.weights().config;
    let layer = engine
        .weights()
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing weights for layer {trace_layer}"))?
        .linear
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} is not linear"))?;
    let state = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing state for layer {trace_layer}"))?;

    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let vhd = cfg.linear_value_head_dim;
    let state_len = cfg.linear_conv_kernel_dim - 1;
    let kernel_size = cfg.linear_conv_kernel_dim;
    let key_dim = nk * cfg.linear_key_head_dim;

    let qkv = decode_bf16_le(qkv_bytes);
    let conv_state = decode_bf16_le(
        &state
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} missing conv_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("conv_state D2H: {e}"))?,
    );
    let conv_w = decode_bf16_le(
        &layer
            .conv1d_w
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("conv1d_w D2H: {e}"))?,
    );
    let conv_channel = |channel: usize| -> f32 {
        let weight_base = channel * kernel_size;
        let state_base = channel * state_len;
        let mut acc = 0.0f32;
        for tap in 0..kernel_size {
            let x = if tap + 1 == kernel_size {
                qkv[channel]
            } else if tap < state_len {
                conv_state[state_base + tap]
            } else {
                0.0
            };
            acc += x * conv_w[weight_base + tap];
        }
        bf16_round(acc * sigmoid_fast(acc))
    };

    let mut v = vec![0.0f32; nv * vhd];
    for v_head in 0..nv {
        let v_base = key_dim * 2 + v_head * vhd;
        for i in 0..vhd {
            v[v_head * vhd + i] = conv_channel(v_base + i);
        }
    }
    Ok(v)
}

fn sigmoid_fast(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn bf16_round(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn trace_component_linear_state_layer(
    engine: &DecodeEngine,
    trace_layer: usize,
    history_token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let native_layer = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native layer {trace_layer}"))?;
    let native_conv = native_layer
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} has no conv_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native conv_state D2H: {e}"))?;
    let native_rec = native_layer
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} has no recurrent_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native recurrent_state D2H: {e}"))?;

    let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
        .map_err(|e| anyhow::anyhow!("component linear state replay init: {e}"))?;
    prefill_engine::prefill(
        engine.weights(),
        &mut replay_state,
        engine.rotary(),
        history_token_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
    )?;
    let replay_layer = replay_state
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing replay layer {trace_layer}"))?;
    let replay_conv = replay_layer
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay layer {trace_layer} has no conv_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay conv_state D2H: {e}"))?;
    let replay_rec = replay_layer
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay layer {trace_layer} has no recurrent_state"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay recurrent_state D2H: {e}"))?;

    let conv_delta = validate::max_abs_delta(&decode_bf16_le(&native_conv), &decode_bf16_le(&replay_conv));
    let rec_delta = validate::max_abs_delta(&decode_f32_le(&native_rec), &decode_f32_le(&replay_rec));
    eprintln!(
        "[trace-component-linear-state] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={rec_delta:.6}"
    );
    Ok(())
}

fn compare_kv_layer(native: &LayerState, replay: &LayerState) -> Result<KvFp8LayerDiff> {
    let filled = native.kv_filled.min(replay.kv_filled);
    let kv_dtype = native
        .kv_cache_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native kv_cache_k missing"))?
        .dtype();
    let mut diff = KvFp8LayerDiff {
        filled,
        dtype: if matches!(kv_dtype, gpu_hal::ScalarType::U8) { "fp8" } else { "bf16" },
        k_mismatches: 0,
        v_mismatches: 0,
        max_k_delta: 0.0,
        max_v_delta: 0.0,
        max_scale_k_delta: 0.0,
        max_scale_v_delta: 0.0,
        first_k_mismatch: None,
        first_v_mismatch: None,
    };
    if filled == 0 {
        return Ok(diff);
    }

    let native_k = native
        .kv_cache_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native kv_cache_k missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native kv_cache_k D2H: {e}"))?;
    let replay_k = replay
        .kv_cache_k
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay kv_cache_k missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay kv_cache_k D2H: {e}"))?;
    let native_v = native
        .kv_cache_v
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("native kv_cache_v missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native kv_cache_v D2H: {e}"))?;
    let replay_v = replay
        .kv_cache_v
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("replay kv_cache_v missing"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("replay kv_cache_v D2H: {e}"))?;

    let native_k_shape = native.kv_cache_k.as_ref().unwrap().shape();
    let replay_k_shape = replay.kv_cache_k.as_ref().unwrap().shape();
    let nkv = native_k_shape[1].min(replay_k_shape[1]);
    let hd = native_k_shape[3].min(replay_k_shape[3]);
    let native_cap = native_k_shape[2];
    let replay_cap = replay_k_shape[2];

    if matches!(kv_dtype, gpu_hal::ScalarType::U8) {
        let native_scale_shape = native.kv_scale_k.as_ref().unwrap().shape();
        let replay_scale_shape = replay.kv_scale_k.as_ref().unwrap().shape();
        for h in 0..nkv {
            for t in 0..filled {
                for d in 0..hd {
                    let native_idx = (h * native_cap + t) * hd + d;
                    let replay_idx = (h * replay_cap + t) * hd + d;
                    let nk = native_k[native_idx];
                    let rk = replay_k[replay_idx];
                    if nk != rk {
                        diff.k_mismatches += 1;
                        if diff.first_k_mismatch.is_none() {
                            diff.first_k_mismatch = Some((h, t, d, nk, rk));
                        }
                    }
                    let nv = native_v[native_idx];
                    let rv = replay_v[replay_idx];
                    if nv != rv {
                        diff.v_mismatches += 1;
                        if diff.first_v_mismatch.is_none() {
                            diff.first_v_mismatch = Some((h, t, d, nv, rv));
                        }
                    }
                }
            }
        }

        let native_scale_k = decode_f32_le(
            &native
                .kv_scale_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("native kv_scale_k missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("native kv_scale_k D2H: {e}"))?,
        );
        let replay_scale_k = decode_f32_le(
            &replay
                .kv_scale_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("replay kv_scale_k missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("replay kv_scale_k D2H: {e}"))?,
        );
        let native_scale_v = decode_f32_le(
            &native
                .kv_scale_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("native kv_scale_v missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("native kv_scale_v D2H: {e}"))?,
        );
        let replay_scale_v = decode_f32_le(
            &replay
                .kv_scale_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("replay kv_scale_v missing"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("replay kv_scale_v D2H: {e}"))?,
        );

        let native_scale_cap = native_scale_shape[1];
        let replay_scale_cap = replay_scale_shape[1];
        for h in 0..native_scale_shape[0].min(replay_scale_shape[0]) {
            for t in 0..filled {
                let nk = native_scale_k[h * native_scale_cap + t];
                let rk = replay_scale_k[h * replay_scale_cap + t];
                diff.max_scale_k_delta = diff.max_scale_k_delta.max((nk - rk).abs());
                let nv = native_scale_v[h * native_scale_cap + t];
                let rv = replay_scale_v[h * replay_scale_cap + t];
                diff.max_scale_v_delta = diff.max_scale_v_delta.max((nv - rv).abs());
            }
        }
    } else {
        let native_k_f32 = decode_bf16_le(&native_k);
        let replay_k_f32 = decode_bf16_le(&replay_k);
        let native_v_f32 = decode_bf16_le(&native_v);
        let replay_v_f32 = decode_bf16_le(&replay_v);
        for h in 0..nkv {
            for t in 0..filled {
                for d in 0..hd {
                    let native_idx = (h * native_cap + t) * hd + d;
                    let replay_idx = (h * replay_cap + t) * hd + d;
                    let nk = native_k_f32[native_idx];
                    let rk = replay_k_f32[replay_idx];
                    let kd = (nk - rk).abs();
                    diff.max_k_delta = diff.max_k_delta.max(kd);
                    if kd > 0.0 {
                        diff.k_mismatches += 1;
                        if diff.first_k_mismatch.is_none() {
                            diff.first_k_mismatch = Some((
                                h,
                                t,
                                d,
                                native_k[native_idx * 2],
                                replay_k[replay_idx * 2],
                            ));
                        }
                    }
                    let nv = native_v_f32[native_idx];
                    let rv = replay_v_f32[replay_idx];
                    let vd = (nv - rv).abs();
                    diff.max_v_delta = diff.max_v_delta.max(vd);
                    if vd > 0.0 {
                        diff.v_mismatches += 1;
                        if diff.first_v_mismatch.is_none() {
                            diff.first_v_mismatch = Some((
                                h,
                                t,
                                d,
                                native_v[native_idx * 2],
                                replay_v[replay_idx * 2],
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(diff)
}
