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
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use base64::Engine as _;
use clap::Parser;
use gpu_hal::{GpuBuffer, ScalarType};

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

type NativePrefillTraceBundle = (
    Option<Vec<u8>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
);

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

pub(crate) fn load_tokenizer(tokenizer_path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))
}

pub(crate) fn parse_prompt_ids_csv(spec: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for raw in spec.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            anyhow::bail!("--prompt-ids contains an empty entry");
        }
        let id = trimmed
            .parse::<u32>()
            .map_err(|e| anyhow::anyhow!("invalid token id '{trimmed}' in --prompt-ids: {e}"))?;
        ids.push(id);
    }
    if ids.is_empty() {
        anyhow::bail!("--prompt-ids must contain at least one token id");
    }
    Ok(ids)
}

pub(crate) fn resolve_prompt_token_ids(
    cli: &Cli,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<u32>> {
    let prompt_ids = if let Some(spec) = cli.prompt_ids.as_deref() {
        let ids = parse_prompt_ids_csv(spec)?;
        eprintln!(
            "[tokenizer] prompt_tokens={} (from --prompt-ids)",
            ids.len()
        );
        ids
    } else if let Some(prompt) = cli.prompt.as_deref() {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("[tokenizer] prompt_tokens={} (from --prompt)", ids.len());
        ids
    } else {
        anyhow::bail!("pass either --prompt or --prompt-ids");
    };

    if prompt_ids.is_empty() {
        anyhow::bail!("empty prompt after tokenization");
    }
    Ok(prompt_ids)
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
    #[arg(long, conflicts_with = "prompt_ids")]
    prompt: Option<String>,

    /// Exact prompt token IDs as a comma-separated list.
    #[arg(long, conflicts_with = "prompt")]
    prompt_ids: Option<String>,

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

    /// Debug-only: when tracing Qwen prefill on Metal, also dump one selected
    /// full-attention layer's internal tensors against the Python oracle.
    /// Defaults to a later full-attention layer where parity drift is larger.
    #[arg(long, hide = true)]
    trace_prefill_full_layer: Option<usize>,

    /// Debug-only: when tracing Qwen prefill on Metal, also dump one selected
    /// MLP block's internal tensors against the Python oracle.
    /// Defaults to the selected full-attention trace layer when present.
    #[arg(long, hide = true)]
    trace_prefill_mlp_layer: Option<usize>,

    /// Debug-only: seed prefill from the oracle decoder-block output of this
    /// layer, then replay the remaining tail natively. Used to distinguish
    /// upstream hidden-state drift from bugs in the later layers.
    #[arg(long, hide = true)]
    trace_prefill_restart_layer: Option<usize>,

    /// Debug-only: when tracing Qwen prefill, compare one selected prompt
    /// position instead of the last prompt token for the layer-hidden sweep
    /// and traced layer internals. Defaults to the last prompt token.
    #[arg(long, hide = true)]
    trace_prefill_position: Option<usize>,

    /// Debug-only: when replaying a traced Qwen prefill tail from an oracle
    /// layer output, sweep multiple prompt positions instead of one and emit
    /// a compact summary of the worst position per layer. Accepts `all` or a
    /// comma-separated list like `0,4,9,14`.
    #[arg(long, hide = true)]
    trace_prefill_restart_position_scan: Option<String>,

    /// Debug-only: on replay-prefill decode paths, re-run one selected decode
    /// step through the traced prefill engine and compare the last-token layer
    /// state against the Python oracle for the full token history at that step.
    #[arg(long, hide = true)]
    trace_replay_decode_step: Option<usize>,

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
    if cli.prompt.is_none() && cli.prompt_ids.is_none() {
        anyhow::bail!("pass either --prompt or --prompt-ids");
    }
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
                let e =
                    registry::lookup(&model_variant, &backend, &reuse_arch).ok_or_else(|| {
                        let supported_archs =
                            registry::supported_archs_for(&model_variant, &backend);
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
        anyhow::bail!("--trace-persistent-input-layer requires the real 4B persistent kernel path");
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
                anyhow::bail!("--allow-unstable-cuda-kv-fp8 only supports the CUDA 4B kernel path");
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
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let prompt_ids = resolve_prompt_token_ids(&cli, &tokenizer)?;

    // 4. VRAM check (needs config + prompt length for KV cache estimation)
    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    let kv_dtype_bytes = if cli.kv_fp8 {
        1usize
    } else {
        gpu_hal::ScalarType::BF16.size_in_bytes()
    };
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
    if (cli.trace_prefill_layers
        || cli.trace_replay_decode_step.is_some()
        || cli.trace_prefill_restart_layer.is_some())
        && !cli.validate
    {
        anyhow::bail!(
            "--trace-prefill-layers, --trace-prefill-restart-layer, and --trace-replay-decode-step require --validate"
        );
    }
    if cli.trace_prefill_restart_layer.is_some() && !cli.trace_prefill_layers {
        anyhow::bail!("--trace-prefill-restart-layer requires --trace-prefill-layers");
    }
    if cli.trace_prefill_restart_position_scan.is_some()
        && cli.trace_prefill_restart_layer.is_none()
    {
        anyhow::bail!(
            "--trace-prefill-restart-position-scan requires --trace-prefill-restart-layer"
        );
    }
    if let Some(step) = cli.trace_replay_decode_step {
        if model_variant.family() != ModelFamily::Qwen35 {
            anyhow::bail!("--trace-replay-decode-step is only supported on Qwen3.5");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("--trace-replay-decode-step only supports --batch-size 1");
        }
        if step >= cli.max_new_tokens {
            anyhow::bail!(
                "--trace-replay-decode-step {} must be less than --max-new-tokens {}",
                step,
                cli.max_new_tokens
            );
        }
    }
    let qwen_trace_enabled = (cli.trace_prefill_layers || cli.trace_replay_decode_step.is_some())
        && model_variant.family() == ModelFamily::Qwen35;
    let trace_prefill_linear_layer = if qwen_trace_enabled {
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
    let trace_prefill_full_layer = if qwen_trace_enabled {
        let layer = cli.trace_prefill_full_layer.unwrap_or(19);
        let config = &engine.weights().config;
        if layer >= config.num_hidden_layers {
            anyhow::bail!(
                "--trace-prefill-full-layer {} out of range for {} layers",
                layer,
                config.num_hidden_layers
            );
        }
        if !config.is_full_attention(layer) {
            anyhow::bail!(
                "--trace-prefill-full-layer {} selects a linear-attention layer; choose a full-attention layer",
                layer
            );
        }
        Some(layer)
    } else {
        None
    };
    let trace_prefill_mlp_layer = if qwen_trace_enabled {
        let layer = cli
            .trace_prefill_mlp_layer
            .or(trace_prefill_full_layer)
            .unwrap_or(19);
        let config = &engine.weights().config;
        if layer >= config.num_hidden_layers {
            anyhow::bail!(
                "--trace-prefill-mlp-layer {} out of range for {} layers",
                layer,
                config.num_hidden_layers
            );
        }
        Some(layer)
    } else {
        None
    };
    let trace_prefill_restart_layer = if cli.trace_prefill_restart_layer.is_some() {
        if model_variant.family() != ModelFamily::Qwen35 {
            anyhow::bail!("--trace-prefill-restart-layer is only supported on Qwen3.5");
        }
        let layer = cli.trace_prefill_restart_layer.unwrap();
        let config = &engine.weights().config;
        if layer >= config.num_hidden_layers {
            anyhow::bail!(
                "--trace-prefill-restart-layer {} out of range for {} layers",
                layer,
                config.num_hidden_layers
            );
        }
        Some(layer)
    } else {
        None
    };
    let trace_prefill_position = if cli.trace_prefill_layers {
        let position = cli
            .trace_prefill_position
            .unwrap_or_else(|| prompt_ids.len().saturating_sub(1));
        if position >= prompt_ids.len() {
            anyhow::bail!(
                "--trace-prefill-position {} out of range for prompt length {}",
                position,
                prompt_ids.len()
            );
        }
        Some(position)
    } else {
        None
    };
    let trace_prefill_restart_position_scan =
        if let Some(spec) = cli.trace_prefill_restart_position_scan.as_deref() {
            Some(parse_trace_position_scan(spec, prompt_ids.len())?)
        } else {
            None
        };
    let oracle_model_id = cli
        .model_id
        .clone()
        .unwrap_or_else(|| model_variant.hf_model_id().to_string());
    let oracle_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf();
    let oracle_script = oracle_root.join("oracle/run_oracle.py");
    let qwen35_trace_script = oracle_root.join("oracle/qwen35_oracle.py");
    let metal_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_METAL_FAST_GREEDY").is_some();
    let metal_prefill_fast_greedy_enabled = backend == Backend::Metal
        && model_variant == ModelVariant::Qwen3_5_0_8B
        && cli.batch_size == 1
        && !params.use_4b_kernel
        && !cli.validate
        && !cli.gpu_validate
        && !cli.trace_prefill_layers
        && !cli.oracle_prefill
        && !metal_fast_greedy_disabled;

    // Run prefill (native GPU or oracle)
    let prefill_start = Instant::now();
    let (
        prefill_logits,
        native_prefill_trace,
        native_linear_debug_trace,
        native_layer3_full_attn_trace,
        native_mlp_debug_trace,
        mut next_token,
    ) = if cli.oracle_prefill {
        let output = oracle::run_oracle(
            &oracle_script,
            &oracle_model_id,
            &prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            &oracle_device,
            true,
            fp8_oracle_dir.as_deref(),
            None,
        )?;
        engine.load_prefill_state(&output)?;
        let first = output
            .generated_token_ids
            .first()
            .copied()
            .unwrap_or_else(|| DecodeEngine::greedy_sample(&output.prefill_logits));
        eprintln!(
            "[prefill] oracle prefill done in {:.0}ms",
            prefill_start.elapsed().as_millis()
        );
        (output.prefill_logits, None, None, None, None, first)
    } else {
        let prefill_result = if cli.trace_prefill_layers {
            engine.prefill_native_with_trace(
                &prompt_ids,
                trace_prefill_linear_layer,
                trace_prefill_full_layer,
                trace_prefill_mlp_layer,
                trace_prefill_position,
            )?
        } else if metal_prefill_fast_greedy_enabled {
            let first = engine.prefill_native_greedy_token(&prompt_ids)?;
            prefill_engine::PrefillResult {
                logits: Vec::new(),
                sampled_token: Some(first),
                final_norm_trace: None,
                layer_attn_trace: None,
                layer_post_attn_norm_trace: None,
                layer_mlp_swiglu_trace: None,
                layer_mlp_out_trace: None,
                layer_hidden_trace: None,
                tap_hiddens: None,
                linear_debug_trace: None,
                layer3_full_attn_trace: None,
                mlp_debug_trace: None,
            }
        } else {
            prefill_engine::PrefillResult {
                logits: engine.prefill_native(&prompt_ids)?,
                sampled_token: None,
                final_norm_trace: None,
                layer_attn_trace: None,
                layer_post_attn_norm_trace: None,
                layer_mlp_swiglu_trace: None,
                layer_mlp_out_trace: None,
                layer_hidden_trace: None,
                tap_hiddens: None,
                linear_debug_trace: None,
                layer3_full_attn_trace: None,
                mlp_debug_trace: None,
            }
        };
        let first = prefill_result
            .sampled_token
            .unwrap_or_else(|| DecodeEngine::greedy_sample(&prefill_result.logits));
        eprintln!(
            "[prefill] native GPU prefill done in {:.0}ms",
            prefill_start.elapsed().as_millis()
        );
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
            prefill_result.mlp_debug_trace,
            first,
        )
    };

    // Optionally run oracle for validation
    let oracle_output = if cli.validate {
        let output = oracle::run_oracle(
            &oracle_script,
            &oracle_model_id,
            &prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            &oracle_device,
            cli.trace_prefill_layers,
            fp8_oracle_dir.as_deref(),
            None,
        )?;

        // Compare prefill logits
        let prefill_delta = validate::max_abs_delta(&prefill_logits, &output.prefill_logits);
        eprintln!("[validate] prefill logit delta={prefill_delta:.4}");

        let qwen35_trace_output =
            if cli.trace_prefill_layers && model_variant.family() == ModelFamily::Qwen35 {
                match oracle::run_qwen35_trace_oracle(
                    &qwen35_trace_script,
                    &oracle_model_id,
                    &prompt_ids,
                    cli.max_new_tokens,
                    &cli.oracle_dtype,
                    &oracle_device,
                    trace_prefill_linear_layer,
                    trace_prefill_full_layer,
                    trace_prefill_mlp_layer,
                    trace_prefill_position,
                ) {
                    Ok(trace) => Some(trace),
                    Err(err) => {
                        eprintln!("[trace-prefill] qwen35_trace_unavailable: {err}");
                        None
                    }
                }
            } else {
                None
            };

        // Check if oracle and native agree on first token
        if let Some(&oracle_first) = output.generated_token_ids.first() {
            if oracle_first != next_token {
                eprintln!(
                    "[validate] WARNING: prefill token mismatch! native={next_token} oracle={oracle_first}"
                );
            }
        }

        if cli.trace_prefill_layers {
            emit_qwen35_trace_report(
                &engine,
                "trace-prefill",
                trace_prefill_linear_layer,
                trace_prefill_full_layer,
                trace_prefill_mlp_layer,
                &prefill_logits,
                native_prefill_trace.as_ref(),
                native_linear_debug_trace.as_ref(),
                native_layer3_full_attn_trace.as_ref(),
                native_mlp_debug_trace.as_ref(),
                &output,
                qwen35_trace_output.as_ref(),
                trace_prefill_position,
            )?;
        }

        if let (Some(restart_layer), Some(trace)) =
            (trace_prefill_restart_layer, qwen35_trace_output.as_ref())
        {
            let hidden_value = trace
                .decoder_layer_outputs
                .get(restart_layer)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "qwen35 trace oracle missing decoder_layer_outputs[{restart_layer}]"
                    )
                })?;
            let hidden_f32 = flatten_bsh(hidden_value).ok_or_else(|| {
                anyhow::anyhow!(
                    "qwen35 trace oracle decoder_layer_outputs[{restart_layer}] was not flattenable"
                )
            })?;
            let expected_hidden = prompt_ids.len() * engine.weights().config.hidden_size;
            if hidden_f32.len() != expected_hidden {
                anyhow::bail!(
                    "qwen35 trace oracle decoder_layer_outputs[{restart_layer}] had {} floats, expected {} for prompt_len={} hidden_size={}",
                    hidden_f32.len(),
                    expected_hidden,
                    prompt_ids.len(),
                    engine.weights().config.hidden_size
                );
            }
            if let Some(position) = trace_prefill_position {
                emit_qwen35_restart_source_tail_report(
                    &engine,
                    "trace-prefill-restart-source",
                    restart_layer,
                    &hidden_f32,
                    trace,
                    position,
                )?;
            }
            let hidden_bf16 = encode_bf16_le(&hidden_f32);
            let start_layer = restart_layer + 1;
            let restart_prefill = engine.prefill_tail_from_hidden_with_trace(
                &hidden_bf16,
                start_layer,
                trace_prefill_linear_layer.filter(|layer| *layer >= start_layer),
                trace_prefill_full_layer.filter(|layer| *layer >= start_layer),
                trace_prefill_mlp_layer.filter(|layer| *layer >= start_layer),
                trace_prefill_position,
            )?;
            emit_qwen35_restart_report(
                &engine.weights().config,
                "trace-prefill-restart",
                restart_layer,
                start_layer,
                &restart_prefill,
                &output,
                qwen35_trace_output.as_ref(),
                trace_prefill_position,
            )?;
            if let Some(positions) = trace_prefill_restart_position_scan.as_ref() {
                emit_qwen35_restart_position_scan_report(
                    &engine,
                    "trace-prefill-restart-scan",
                    start_layer,
                    &hidden_bf16,
                    positions,
                    trace_prefill_linear_layer,
                    trace_prefill_full_layer,
                    trace_prefill_mlp_layer,
                    trace,
                )?;
            }
            let restart_trace_bundle: NativePrefillTraceBundle = (
                restart_prefill.final_norm_trace.clone(),
                restart_prefill.layer_attn_trace.clone(),
                restart_prefill.layer_post_attn_norm_trace.clone(),
                restart_prefill.layer_mlp_out_trace.clone(),
                restart_prefill.layer_hidden_trace.clone(),
            );
            emit_qwen35_trace_report_with_offset(
                &engine,
                "trace-prefill-restart",
                start_layer,
                trace_prefill_linear_layer.filter(|layer| *layer >= start_layer),
                trace_prefill_full_layer.filter(|layer| *layer >= start_layer),
                trace_prefill_mlp_layer.filter(|layer| *layer >= start_layer),
                &restart_prefill.logits,
                Some(&restart_trace_bundle),
                restart_prefill.linear_debug_trace.as_ref(),
                restart_prefill.layer3_full_attn_trace.as_ref(),
                restart_prefill.mlp_debug_trace.as_ref(),
                &output,
                qwen35_trace_output.as_ref(),
                trace_prefill_position,
            )?;
        }

        Some(output)
    } else {
        None
    };

    // Replicate prefill state to batch items if batch_size > 1
    if cli.batch_size > 1 {
        eprintln!(
            "[batch] replicating prefill state to {} sequences",
            cli.batch_size
        );
        engine.replicate_state_to_batch()?;
    }

    let gpu_validate_enabled = if cli.gpu_validate && cli.batch_size == 1 {
        eprintln!(
            "[gpu-validate] replaying full token history through GPU prefill for reference..."
        );
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
        && (backend == Backend::Metal || (params.use_4b_kernel && cli.force_replay_decode));
    if cli.trace_replay_decode_step.is_some() && !replay_decode_enabled {
        anyhow::bail!(
            "--trace-replay-decode-step requires the replay-prefill decode path \
             (Metal v1 or --force-replay-decode)"
        );
    }
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
    let metal_replay_fast_greedy_enabled = metal_prefill_fast_greedy_enabled
        && replay_decode_enabled
        && oracle_output.is_none()
        && !gpu_validate_enabled
        && !cli.emit_stage_timings
        && cli.trace_replay_decode_step.is_none();
    let metal_component_decode_enabled = metal_replay_fast_greedy_enabled
        && env::var_os("SUPERSONIC_METAL_ENABLE_COMPONENT_DECODE").is_some();
    if replay_decode_enabled {
        if metal_component_decode_enabled {
            eprintln!(
                "[decode] experimental Metal component decode path enabled \
                 (prototype; parity not guaranteed)"
            );
        } else if metal_replay_fast_greedy_enabled {
            eprintln!("[decode] Metal fast greedy replay-prefill path enabled");
        } else if backend == Backend::Metal {
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
                (
                    engine.decode_step_batch(&batch_next_tokens, seqlen_offset)?,
                    None,
                )
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
                    if delta > max_delta {
                        max_delta = delta;
                    }
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
            let replay_token_ids = replay_decode_enabled.then(|| {
                prompt_ids
                    .iter()
                    .copied()
                    .chain(generated_ids.iter().copied())
                    .chain(std::iter::once(next_token))
                    .collect::<Vec<_>>()
            });
            let logits = if cuda_fast_greedy_enabled {
                let (token, timings) =
                    engine.decode_step_cuda_fast_greedy(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if cuda_08b_hero_enabled {
                let (token, timings) =
                    engine.decode_step_cuda_08b_hero(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if replay_decode_enabled {
                let token_ids = replay_token_ids
                    .as_ref()
                    .expect("replay token ids are present when replay decode is enabled");
                if metal_component_decode_enabled {
                    let metal_linear_trace_layer =
                        if let Some(raw) = env::var_os("SUPERSONIC_METAL_TRACE_COMPONENT_LINEAR_LAYER")
                        {
                            Some(
                                raw.to_string_lossy().parse::<usize>().with_context(|| {
                                    format!(
                                        "invalid SUPERSONIC_METAL_TRACE_COMPONENT_LINEAR_LAYER '{}'",
                                        raw.to_string_lossy()
                                    )
                                })?,
                            )
                        } else {
                            None
                        };
                    let metal_input_trace_layer =
                        if let Some(raw) = env::var_os("SUPERSONIC_METAL_TRACE_COMPONENT_INPUT_LAYER")
                        {
                            Some(
                                raw.to_string_lossy().parse::<usize>().with_context(|| {
                                    format!(
                                        "invalid SUPERSONIC_METAL_TRACE_COMPONENT_INPUT_LAYER '{}'",
                                        raw.to_string_lossy()
                                    )
                                })?,
                            )
                        } else {
                            None
                        };
                    let (token, timings) = if let Some(trace_layer) = metal_input_trace_layer {
                        let (token, timings, hidden_trace) = engine
                            .decode_step_metal_component_greedy_trace_input_layer(
                                next_token,
                                seqlen_offset,
                                trace_layer,
                            )?;
                        trace_component_input_layer(
                            &engine,
                            &hidden_trace,
                            trace_layer,
                            token_ids,
                            ordinal,
                            params.kv_chunk_size,
                            cli.prefill_chunk_size,
                            params.use_4b_kernel,
                        )?;
                        (token, timings)
                    } else if let Some(trace_layer) = metal_linear_trace_layer {
                        let (token, timings, linear_trace) = engine
                            .decode_step_metal_component_greedy_trace_linear_layer(
                                next_token,
                                seqlen_offset,
                                trace_layer,
                            )?;
                        trace_component_linear_layer(
                            &engine,
                            trace_layer,
                            &linear_trace,
                            token_ids,
                            ordinal,
                            params.kv_chunk_size,
                            cli.prefill_chunk_size,
                            params.use_4b_kernel,
                        )?;
                        (token, timings)
                    } else {
                        engine.decode_step_metal_component_greedy(next_token, seqlen_offset)?
                    };
                    if let Some(trace_spec) =
                        env::var_os("SUPERSONIC_METAL_TRACE_COMPONENT_LINEAR_STATE_LAYER")
                    {
                        trace_metal_component_linear_state_layers(
                            &engine,
                            trace_spec.to_string_lossy().as_ref(),
                            token_ids,
                            ordinal,
                            params.kv_chunk_size,
                            cli.prefill_chunk_size,
                            params.use_4b_kernel,
                        )?;
                    }
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    maybe_fast_token = Some(token);
                    Vec::new()
                } else if metal_replay_fast_greedy_enabled {
                    maybe_fast_token = Some(engine.rebuild_prefill_state_greedy_token(token_ids)?);
                    Vec::new()
                } else {
                    prefill_engine::gpu_reference_replay_step(
                        &engine.weights(),
                        &engine.rotary(),
                        token_ids,
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?
                }
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
                    let (logits, hidden_trace) = engine.component_decode_step_4b_traced(
                        next_token,
                        seqlen_offset,
                        trace_layer,
                    )?;
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
                    let (logits, layer_trace) = engine.component_decode_step_4b_trace_layer(
                        next_token,
                        seqlen_offset,
                        trace_layer,
                    )?;
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
                        .component_decode_step_4b_trace_linear_layer(
                            next_token,
                            seqlen_offset,
                            trace_layer,
                        )?;
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
                    let (logits, timings) = engine
                        .decode_step_4b_single_kernel_with_timings(next_token, seqlen_offset)?;
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    logits
                } else {
                    engine
                        .decode_step_batch(&[next_token], seqlen_offset)?
                        .remove(0)
                }
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_with_timings(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                logits
            } else {
                engine.decode_step(next_token, seqlen_offset)?
            };
            let native_token =
                maybe_fast_token.unwrap_or_else(|| DecodeEngine::greedy_sample(&logits));

            if let Some(ref oracle) = oracle_output {
                if step < oracle.decode_logits.len() {
                    let oracle_logits = &oracle.decode_logits[step];
                    let delta = validate::max_abs_delta(&logits, oracle_logits);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    eprintln!(
                        "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
                    );
                }
            }

            if cli.trace_replay_decode_step == Some(step) {
                let trace_token_ids = replay_token_ids.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--trace-replay-decode-step requires replay-prefill decode for this run"
                    )
                })?;
                let mut replay_state = ModelState::new(&engine.weights().config, ordinal)
                    .map_err(|e| anyhow::anyhow!("replay trace model state init: {e}"))?;
                let replay_trace = prefill_engine::prefill(
                    engine.weights(),
                    &mut replay_state,
                    engine.rotary(),
                    trace_token_ids,
                    ordinal,
                    params.kv_chunk_size,
                    cli.prefill_chunk_size,
                    false,
                    params.use_4b_kernel,
                    true,
                    trace_prefill_linear_layer,
                    trace_prefill_full_layer,
                    trace_prefill_mlp_layer,
                )?;
                let replay_oracle = oracle::run_oracle(
                    &oracle_script,
                    &oracle_model_id,
                    trace_token_ids,
                    0,
                    &cli.oracle_dtype,
                    &oracle_device,
                    true,
                    fp8_oracle_dir.as_deref(),
                    None,
                )?;
                let replay_qwen_trace = match oracle::run_qwen35_trace_oracle(
                    &qwen35_trace_script,
                    &oracle_model_id,
                    trace_token_ids,
                    0,
                    &cli.oracle_dtype,
                    &oracle_device,
                    trace_prefill_linear_layer,
                    trace_prefill_full_layer,
                    trace_prefill_mlp_layer,
                    None,
                ) {
                    Ok(trace) => Some(trace),
                    Err(err) => {
                        eprintln!("[trace-replay] qwen35_trace_unavailable: {err}");
                        None
                    }
                };
                let replay_trace_bundle: NativePrefillTraceBundle = (
                    replay_trace.final_norm_trace,
                    replay_trace.layer_attn_trace,
                    replay_trace.layer_post_attn_norm_trace,
                    replay_trace.layer_mlp_out_trace,
                    replay_trace.layer_hidden_trace,
                );
                let replay_delta = validate::max_abs_delta(&logits, &replay_oracle.prefill_logits);
                eprintln!(
                    "[trace-replay] step={step} seq_off={seqlen_offset} prefill_logit_delta={replay_delta:.4}"
                );
                emit_qwen35_trace_report(
                    &engine,
                    "trace-replay",
                    trace_prefill_linear_layer,
                    trace_prefill_full_layer,
                    trace_prefill_mlp_layer,
                    &logits,
                    Some(&replay_trace_bundle),
                    replay_trace.linear_debug_trace.as_ref(),
                    replay_trace.layer3_full_attn_trace.as_ref(),
                    replay_trace.mlp_debug_trace.as_ref(),
                    &replay_oracle,
                    replay_qwen_trace.as_ref(),
                    None,
                )?;
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
                let token_match = if gpu_token == native_token {
                    ""
                } else {
                    " MISMATCH"
                };
                if delta > gpu_max_delta {
                    gpu_max_delta = delta;
                }
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
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let prompt_ids = resolve_prompt_token_ids(cli, &tokenizer)?;

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
                Ok(true) => eprintln!(
                    "[fetch] installed Gemma 4 INT4 bake at {}",
                    target.display()
                ),
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
        if cli.prompt_ids.is_some() {
            anyhow::bail!(
                "Gemma 4 validation does not support --prompt-ids yet; use --prompt or disable --validate"
            );
        }
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("oracle/gemma4_oracle.py");
        let oracle = oracle::run_gemma4_oracle(
            &oracle_script,
            &cli.model_dir,
            cli.prompt.as_deref().unwrap_or_default(),
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
        None,
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

fn emit_qwen35_trace_report(
    engine: &DecodeEngine,
    tag: &str,
    trace_linear_layer: Option<usize>,
    trace_full_layer: Option<usize>,
    trace_mlp_layer: Option<usize>,
    native_logits: &[f32],
    native_prefill_trace: Option<&NativePrefillTraceBundle>,
    native_linear_debug_trace: Option<&prefill_engine::LinearLayerDebugTrace>,
    native_layer3_full_attn_trace: Option<&prefill_engine::Layer3FullAttentionTrace>,
    native_mlp_debug_trace: Option<&prefill_engine::MlpLayerDebugTrace>,
    oracle_output: &oracle::OracleOutput,
    qwen35_trace_output: Option<&oracle::Qwen35TraceOutput>,
    trace_prefill_position: Option<usize>,
) -> Result<()> {
    emit_qwen35_trace_report_with_offset(
        engine,
        tag,
        0,
        trace_linear_layer,
        trace_full_layer,
        trace_mlp_layer,
        native_logits,
        native_prefill_trace,
        native_linear_debug_trace,
        native_layer3_full_attn_trace,
        native_mlp_debug_trace,
        oracle_output,
        qwen35_trace_output,
        trace_prefill_position,
    )
}

fn emit_qwen35_trace_report_with_offset(
    engine: &DecodeEngine,
    tag: &str,
    layer_offset: usize,
    trace_linear_layer: Option<usize>,
    trace_full_layer: Option<usize>,
    trace_mlp_layer: Option<usize>,
    native_logits: &[f32],
    native_prefill_trace: Option<&NativePrefillTraceBundle>,
    native_linear_debug_trace: Option<&prefill_engine::LinearLayerDebugTrace>,
    native_layer3_full_attn_trace: Option<&prefill_engine::Layer3FullAttentionTrace>,
    native_mlp_debug_trace: Option<&prefill_engine::MlpLayerDebugTrace>,
    oracle_output: &oracle::OracleOutput,
    qwen35_trace_output: Option<&oracle::Qwen35TraceOutput>,
    trace_prefill_position: Option<usize>,
) -> Result<()> {
    let linear_tag = format!("{tag}-linear");
    let full_tag = format!("{tag}-full");
    let mlp_tag = format!("{tag}-mlp");

    if let Some(position) = trace_prefill_position {
        eprintln!(
            "[{tag}] trace_position={position} layer_hidden/debug traces target this prompt position; unlabeled final-norm/logit lines still describe the last prompt token"
        );
    }

    if let Some(position) = trace_prefill_position {
        if let (Some((_, _, _, _, native_layer_trace)), Some(trace)) =
            (native_prefill_trace, qwen35_trace_output)
        {
            if let (Some(native_final_hidden), Some(oracle_final_hidden)) = (
                native_layer_trace.as_ref().and_then(|trace| trace.last()),
                trace
                    .decoder_layer_outputs
                    .last()
                    .and_then(|value| flatten_token_bsd(value, Some(position))),
            ) {
                let native_final_hidden = decode_bf16_le(native_final_hidden);
                let (
                    final_hidden_idx,
                    final_hidden_native,
                    final_hidden_oracle,
                    final_hidden_delta,
                ) = max_abs_delta_details(&native_final_hidden, &oracle_final_hidden);
                eprintln!(
                    "[{tag}] position={} final_hidden_delta={final_hidden_delta:.4}",
                    position
                );
                eprintln!(
                    "[{tag}] position={} final_hidden_max idx={final_hidden_idx} native={final_hidden_native:.4} oracle={final_hidden_oracle:.4}",
                    position
                );

                let native_final_norm =
                    compute_qwen_final_norm_from_hidden_row(engine, &native_final_hidden)?;
                let oracle_final_norm = if let Some(actual) = qwen35_trace_output
                    .and_then(|trace| trace.trace_position_prefill_final_norm_output.as_ref())
                    .and_then(flatten_json_vector)
                {
                    actual
                } else {
                    compute_qwen_final_norm_from_hidden_row(engine, &oracle_final_hidden)?
                };
                let (final_norm_idx, final_norm_native, final_norm_oracle, final_norm_delta) =
                    max_abs_delta_details(&native_final_norm, &oracle_final_norm);
                eprintln!(
                    "[{tag}] position={} final_norm_from_hidden_delta={final_norm_delta:.4}",
                    position
                );
                eprintln!(
                    "[{tag}] position={} final_norm_from_hidden_max idx={final_norm_idx} native={final_norm_native:.4} oracle={final_norm_oracle:.4}",
                    position
                );

                let native_hidden_logits =
                    compute_qwen_logits_from_hidden_row(engine, &native_final_hidden)?;
                let oracle_hidden_logits = if let Some(actual) = qwen35_trace_output
                    .and_then(|trace| trace.trace_position_prefill_logits.as_ref())
                    .and_then(flatten_json_vector)
                {
                    actual
                } else {
                    compute_qwen_logits_from_hidden_row(engine, &oracle_final_hidden)?
                };
                let (
                    hidden_logit_idx,
                    hidden_logit_native,
                    hidden_logit_oracle,
                    hidden_logit_delta,
                ) = max_abs_delta_details(&native_hidden_logits, &oracle_hidden_logits);
                eprintln!(
                    "[{tag}] position={} final_hidden_logit_delta={hidden_logit_delta:.4}",
                    position
                );
                eprintln!(
                    "[{tag}] position={} final_hidden_logit_max idx={hidden_logit_idx} native={hidden_logit_native:.4} oracle={hidden_logit_oracle:.4}",
                    position
                );
                eprintln!(
                    "[{tag}] position={} final_hidden_native_top_logits={}",
                    position,
                    format_top_logits(&top_logits(&native_hidden_logits, 5))
                );
                eprintln!(
                    "[{tag}] position={} final_hidden_oracle_top_logits={}",
                    position,
                    format_top_logits(&top_logits(&oracle_hidden_logits, 5))
                );
                let top_dims = logit_row_top_delta_dims(
                    &native_final_norm,
                    &oracle_final_norm,
                    &engine.weights().lm_head,
                    engine.ordinal(),
                    hidden_logit_idx,
                    6,
                )?;
                eprintln!(
                    "[{tag}] position={} final_hidden_logit_row_detail idx={hidden_logit_idx} top_dims={}",
                    position,
                    format_logit_row_dims(&top_dims)
                );
            }
        }
    } else if let (Some((_, _, _, _, native_layer_trace)), Some(oracle_layer_trace)) = (
        native_prefill_trace,
        oracle_output.layer_hidden_states.as_ref(),
    ) {
        if let Some(native_layer_trace) = native_layer_trace.as_ref() {
            if let (Some(native_final_hidden), Some(oracle_final_hidden_b64)) =
                (native_layer_trace.last(), oracle_layer_trace.last())
            {
                let b64 = base64::engine::general_purpose::STANDARD;
                let oracle_final_hidden_bytes = b64
                    .decode(oracle_final_hidden_b64)
                    .map_err(|e| anyhow::anyhow!("decode oracle final layer_hidden_states: {e}"))?;
                let native_final_hidden = decode_bf16_le(native_final_hidden);
                let oracle_final_hidden = decode_bf16_le(&oracle_final_hidden_bytes);
                let (
                    final_hidden_idx,
                    final_hidden_native,
                    final_hidden_oracle,
                    final_hidden_delta,
                ) = max_abs_delta_details(&native_final_hidden, &oracle_final_hidden);
                eprintln!("[{tag}] final_hidden_delta={final_hidden_delta:.4}");
                eprintln!(
                    "[{tag}] final_hidden_mae={:.9e}",
                    mean_abs_delta(&native_final_hidden, &oracle_final_hidden)
                );
                eprintln!(
                    "[{tag}] final_hidden_mse={:.9e}",
                    mean_square_delta(&native_final_hidden, &oracle_final_hidden)
                );
                eprintln!(
                    "[{tag}] final_hidden_max idx={final_hidden_idx} native={final_hidden_native:.4} oracle={final_hidden_oracle:.4}"
                );
            }
        }
    }

    if let (Some((native_final_norm_trace, ..)), Some(oracle_final_norm_b64)) =
        (native_prefill_trace, oracle_output.prefill_hidden.as_ref())
    {
        if let Some(native_final_norm_trace) = native_final_norm_trace.as_ref() {
            let b64 = base64::engine::general_purpose::STANDARD;
            let oracle_final_norm_bytes = b64
                .decode(oracle_final_norm_b64)
                .map_err(|e| anyhow::anyhow!("decode oracle prefill_hidden: {e}"))?;
            let native_final_norm = decode_bf16_le(native_final_norm_trace);
            let oracle_final_norm = decode_bf16_le(&oracle_final_norm_bytes);
            let (final_norm_idx, final_norm_native, final_norm_oracle, final_norm_delta) =
                max_abs_delta_details(&native_final_norm, &oracle_final_norm);
            eprintln!("[{tag}] final_norm_delta={final_norm_delta:.4}");
            eprintln!(
                "[{tag}] final_norm_mae={:.9e}",
                mean_abs_delta(&native_final_norm, &oracle_final_norm)
            );
            eprintln!(
                "[{tag}] final_norm_mse={:.9e}",
                mean_square_delta(&native_final_norm, &oracle_final_norm)
            );
            eprintln!(
                "[{tag}] final_norm_max idx={final_norm_idx} native={final_norm_native:.4} oracle={final_norm_oracle:.4}"
            );
            if let (Some((_, _, _, _, native_layer_trace)), Some(oracle_layer_trace)) = (
                native_prefill_trace,
                oracle_output.layer_hidden_states.as_ref(),
            ) {
                if let Some(native_layer_trace) = native_layer_trace.as_ref() {
                    if let (Some(native_final_hidden_bytes), Some(oracle_final_hidden_b64)) =
                        (native_layer_trace.last(), oracle_layer_trace.last())
                    {
                        let oracle_final_hidden_bytes =
                            b64.decode(oracle_final_hidden_b64).map_err(|e| {
                                anyhow::anyhow!(
                                    "decode oracle final layer hidden for rms detail: {e}"
                                )
                            })?;
                        let native_final_hidden = decode_bf16_le(native_final_hidden_bytes);
                        let oracle_final_hidden = decode_bf16_le(&oracle_final_hidden_bytes);
                        let native_mean_sq = mean_square(&native_final_hidden);
                        let oracle_mean_sq = mean_square(&oracle_final_hidden);
                        let native_inv_rms = 1.0f32
                            / (native_mean_sq + engine.weights().config.rms_norm_eps as f32).sqrt();
                        let oracle_inv_rms = 1.0f32
                            / (oracle_mean_sq + engine.weights().config.rms_norm_eps as f32).sqrt();
                        let scale =
                            read_weight_element_f32(&engine.weights().norm_weight, final_norm_idx)?
                                + 1.0;
                        eprintln!(
                            "[{tag}] final_norm_detail idx={final_norm_idx} hidden_native={:.4} hidden_oracle={:.4} inv_rms_native={native_inv_rms:.6} inv_rms_oracle={oracle_inv_rms:.6} scale={scale:.4}",
                            native_final_hidden[final_norm_idx],
                            oracle_final_hidden[final_norm_idx],
                        );
                    }
                }
            }
        }
    }

    let (logit_idx, native_logit, oracle_logit, logit_delta) =
        max_abs_delta_details(native_logits, &oracle_output.prefill_logits);
    eprintln!("[{tag}] logit_max_delta={logit_delta:.4}");
    eprintln!(
        "[{tag}] logit_mae={:.9e}",
        mean_abs_delta(native_logits, &oracle_output.prefill_logits)
    );
    eprintln!(
        "[{tag}] logit_mse={:.9e}",
        mean_square_delta(native_logits, &oracle_output.prefill_logits)
    );
    eprintln!(
        "[{tag}] logit_max idx={logit_idx} native={native_logit:.4} oracle={oracle_logit:.4}"
    );
    let native_top = top_logits(native_logits, 5);
    let oracle_top = top_logits(&oracle_output.prefill_logits, 5);
    eprintln!(
        "[{tag}] native_top_logits={}",
        format_top_logits(&native_top)
    );
    eprintln!(
        "[{tag}] oracle_top_logits={}",
        format_top_logits(&oracle_top)
    );
    let mut tracked_logit_dims: Option<Vec<usize>> = None;
    let top_logit_deltas = top_abs_delta_dims(native_logits, &oracle_output.prefill_logits, 3);
    eprintln!(
        "[{tag}] top_logit_deltas={}",
        format_top_delta_dims(&top_logit_deltas)
    );
    if let (Some((native_final_norm_trace, ..)), Some(oracle_final_norm_b64)) =
        (native_prefill_trace, oracle_output.prefill_hidden.as_ref())
    {
        if let Some(native_final_norm_trace) = native_final_norm_trace.as_ref() {
            let b64 = base64::engine::general_purpose::STANDARD;
            let oracle_final_norm_bytes = b64.decode(oracle_final_norm_b64).map_err(|e| {
                anyhow::anyhow!("decode oracle prefill_hidden for logit detail: {e}")
            })?;
            let native_final_norm = decode_bf16_le(native_final_norm_trace);
            let oracle_final_norm = decode_bf16_le(&oracle_final_norm_bytes);
            for (rank, (row_idx, _, _, _)) in top_logit_deltas.iter().enumerate() {
                let top_dims = logit_row_top_delta_dims(
                    &native_final_norm,
                    &oracle_final_norm,
                    &engine.weights().lm_head,
                    engine.ordinal(),
                    *row_idx,
                    6,
                )?;
                eprintln!(
                    "[{tag}] logit_row_detail rank={rank} idx={row_idx} top_dims={}",
                    format_logit_row_dims(&top_dims)
                );
            }
            let aggregate_dims = aggregate_logit_row_delta_dims(
                &native_final_norm,
                &oracle_final_norm,
                &engine.weights().lm_head,
                engine.ordinal(),
                &top_logit_deltas
                    .iter()
                    .map(|(row_idx, _, _, _)| *row_idx)
                    .collect::<Vec<_>>(),
                6,
            )?;
            eprintln!(
                "[{tag}] logit_dim_aggregate rows={} top_dims={}",
                top_logit_deltas
                    .iter()
                    .map(|(row_idx, _, _, delta)| format!("{row_idx}:{delta:.4}"))
                    .collect::<Vec<_>>()
                    .join(","),
                format_aggregate_logit_dims(&aggregate_dims)
            );
            tracked_logit_dims = Some(aggregate_dims.iter().map(|(dim, ..)| *dim).collect());
            eprintln!(
                "[{tag}] tracked_logit_dims={}",
                tracked_logit_dims
                    .as_ref()
                    .map(|dims| {
                        dims.iter()
                            .map(|dim| dim.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default()
            );
        }
    }

    if let Some(position) = trace_prefill_position {
        if let (Some((_, _, _, _, native_layer_trace)), Some(trace)) =
            (native_prefill_trace, qwen35_trace_output)
        {
            if let Some(native_layer_trace) = native_layer_trace.as_ref() {
                let mut first_bad = None;
                for rel_layer in 0..native_layer_trace.len() {
                    let layer = layer_offset + rel_layer;
                    let Some(oracle_layer_f32) = trace
                        .decoder_layer_outputs
                        .get(layer)
                        .and_then(|value| flatten_token_bsd(value, Some(position)))
                    else {
                        break;
                    };
                    let native_layer_f32 = decode_bf16_le(&native_layer_trace[rel_layer]);
                    let layer_delta = validate::max_abs_delta(&native_layer_f32, &oracle_layer_f32);
                    let layer_kind = qwen35_layer_kind_label(&engine.weights().config, layer);
                    if first_bad.is_none() && layer_delta > 0.5 {
                        first_bad = Some((layer, layer_delta));
                    }
                    eprintln!(
                        "[{tag}] position={} layer={} kind={} layer_delta={layer_delta:.4}",
                        position, layer, layer_kind
                    );
                }
                if let Some((layer, layer_delta)) = first_bad {
                    eprintln!(
                        "[{tag}] first_bad_layer position={} layer={} layer_delta={layer_delta:.4}",
                        position, layer
                    );
                } else {
                    eprintln!("[{tag}] no layer exceeded delta threshold at position={position}");
                }
            } else {
                eprintln!("[{tag}] missing native layer trace for position={position}");
            }
        } else {
            eprintln!("[{tag}] missing native or qwen35 trace data for position={position}");
        }
    } else if let (
        Some((
            _,
            native_attn_trace,
            native_post_norm_trace,
            native_mlp_out_trace,
            native_layer_trace,
        )),
        Some(oracle_attn_trace),
        Some(oracle_post_norm_trace),
        Some(oracle_mlp_out_trace),
        Some(oracle_layer_trace),
    ) = (
        native_prefill_trace,
        oracle_output.layer_attn_residual_states.as_ref(),
        oracle_output.layer_post_attn_norm_states.as_ref(),
        oracle_output.layer_mlp_outputs.as_ref(),
        oracle_output.layer_hidden_states.as_ref(),
    ) {
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
        ) {
            let b64 = base64::engine::general_purpose::STANDARD;
            let mut first_bad = None;
            let mut max_layer_detail: Option<(usize, f32, f32, f32, f32)> = None;
            let mut max_post_norm_detail: Option<(usize, f32, usize, f32, f32, f32, f32, f32)> =
                None;
            let tracked_dim_layer_start =
                layer_offset.max(engine.weights().config.num_hidden_layers.saturating_sub(6));
            for rel_layer in 0..native_layer_trace.len() {
                let layer = layer_offset + rel_layer;
                if layer >= oracle_layer_trace.len()
                    || layer >= oracle_attn_trace.len()
                    || layer >= oracle_post_norm_trace.len()
                    || layer >= oracle_mlp_out_trace.len()
                {
                    break;
                }
                let oracle_attn_bytes = b64.decode(&oracle_attn_trace[layer]).map_err(|e| {
                    anyhow::anyhow!("decode oracle layer_attn_residual_states[{layer}]: {e}")
                })?;
                let oracle_post_norm_bytes =
                    b64.decode(&oracle_post_norm_trace[layer]).map_err(|e| {
                        anyhow::anyhow!("decode oracle layer_post_attn_norm_states[{layer}]: {e}")
                    })?;
                let oracle_mlp_out_bytes =
                    b64.decode(&oracle_mlp_out_trace[layer]).map_err(|e| {
                        anyhow::anyhow!("decode oracle layer_mlp_outputs[{layer}]: {e}")
                    })?;
                let oracle_layer_bytes = b64.decode(&oracle_layer_trace[layer]).map_err(|e| {
                    anyhow::anyhow!("decode oracle layer_hidden_states[{layer}]: {e}")
                })?;
                let native_attn_f32 = decode_bf16_le(&native_attn_trace[rel_layer]);
                let native_post_norm_f32 = decode_bf16_le(&native_post_norm_trace[rel_layer]);
                let native_mlp_out_f32 = decode_bf16_le(&native_mlp_out_trace[rel_layer]);
                let native_layer_f32 = decode_bf16_le(&native_layer_trace[rel_layer]);
                let oracle_attn_f32 = decode_bf16_le(&oracle_attn_bytes);
                let oracle_post_norm_f32 = decode_bf16_le(&oracle_post_norm_bytes);
                let oracle_mlp_out_f32 = decode_bf16_le(&oracle_mlp_out_bytes);
                let oracle_layer_f32 = decode_bf16_le(&oracle_layer_bytes);
                let attn_delta = validate::max_abs_delta(&native_attn_f32, &oracle_attn_f32);
                let post_norm_delta =
                    validate::max_abs_delta(&native_post_norm_f32, &oracle_post_norm_f32);
                let mlp_out_delta =
                    validate::max_abs_delta(&native_mlp_out_f32, &oracle_mlp_out_f32);
                let layer_delta = validate::max_abs_delta(&native_layer_f32, &oracle_layer_f32);
                let attn_mae = mean_abs_delta(&native_attn_f32, &oracle_attn_f32);
                let post_norm_mae = mean_abs_delta(&native_post_norm_f32, &oracle_post_norm_f32);
                let mlp_out_mae = mean_abs_delta(&native_mlp_out_f32, &oracle_mlp_out_f32);
                let layer_mae = mean_abs_delta(&native_layer_f32, &oracle_layer_f32);
                let attn_mse = mean_square_delta(&native_attn_f32, &oracle_attn_f32);
                let post_norm_mse = mean_square_delta(&native_post_norm_f32, &oracle_post_norm_f32);
                let mlp_out_mse = mean_square_delta(&native_mlp_out_f32, &oracle_mlp_out_f32);
                let layer_mse = mean_square_delta(&native_layer_f32, &oracle_layer_f32);
                let (post_norm_idx, _, _, post_norm_max_delta) =
                    max_abs_delta_details(&native_post_norm_f32, &oracle_post_norm_f32);
                let native_mean_sq = mean_square(&native_attn_f32);
                let oracle_mean_sq = mean_square(&oracle_attn_f32);
                let native_inv_rms =
                    1.0f32 / (native_mean_sq + engine.weights().config.rms_norm_eps as f32).sqrt();
                let oracle_inv_rms =
                    1.0f32 / (oracle_mean_sq + engine.weights().config.rms_norm_eps as f32).sqrt();
                let scale = read_weight_element_f32(
                    &engine.weights().layers[layer].post_attn_norm_w,
                    post_norm_idx,
                )? + 1.0;
                if max_post_norm_detail
                    .as_ref()
                    .map(|(_, delta, _, _, _, _, _, _)| post_norm_max_delta > *delta)
                    .unwrap_or(true)
                {
                    max_post_norm_detail = Some((
                        layer,
                        post_norm_max_delta,
                        post_norm_idx,
                        native_attn_f32[post_norm_idx],
                        oracle_attn_f32[post_norm_idx],
                        native_inv_rms,
                        oracle_inv_rms,
                        scale,
                    ));
                }
                if max_layer_detail
                    .as_ref()
                    .map(|(_, _, _, _, max_layer_delta)| layer_delta > *max_layer_delta)
                    .unwrap_or(true)
                {
                    max_layer_detail = Some((
                        layer,
                        attn_delta,
                        post_norm_delta,
                        mlp_out_delta,
                        layer_delta,
                    ));
                }
                if first_bad.is_none() && layer_delta > 0.5 {
                    first_bad = Some((
                        layer,
                        attn_delta,
                        post_norm_delta,
                        mlp_out_delta,
                        layer_delta,
                    ));
                }
                eprintln!(
                    "[{tag}] layer={layer} attn_delta={attn_delta:.4} attn_mae={attn_mae:.9e} attn_mse={attn_mse:.9e} post_norm_delta={post_norm_delta:.4} post_norm_mae={post_norm_mae:.9e} post_norm_mse={post_norm_mse:.9e} mlp_out_delta={mlp_out_delta:.4} mlp_out_mae={mlp_out_mae:.9e} mlp_out_mse={mlp_out_mse:.9e} layer_delta={layer_delta:.4} layer_mae={layer_mae:.9e} layer_mse={layer_mse:.9e}"
                );
                if let Some(dims) = tracked_logit_dims.as_ref() {
                    if layer >= tracked_dim_layer_start {
                        eprintln!(
                            "[{tag}] tracked_dims layer={layer} dims={}",
                            format_tracked_stage_dim_deltas(
                                dims,
                                &native_attn_f32,
                                &oracle_attn_f32,
                                &native_post_norm_f32,
                                &oracle_post_norm_f32,
                                &native_mlp_out_f32,
                                &oracle_mlp_out_f32,
                                &native_layer_f32,
                                &oracle_layer_f32,
                            )
                        );
                    }
                }
            }
            if let Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta)) =
                first_bad
            {
                eprintln!(
                    "[{tag}] first_bad_layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}"
                );
            } else {
                eprintln!("[{tag}] no layer exceeded delta threshold");
            }
            if let Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta)) =
                max_layer_detail
            {
                eprintln!(
                    "[{tag}] max_layer_detail layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}"
                );
            }
            if let Some((
                layer,
                post_norm_delta,
                idx,
                hidden_native,
                hidden_oracle,
                native_inv_rms,
                oracle_inv_rms,
                scale,
            )) = max_post_norm_detail
            {
                eprintln!(
                    "[{tag}] max_post_norm_detail layer={layer} idx={idx} post_norm_delta={post_norm_delta:.4} hidden_native={hidden_native:.4} hidden_oracle={hidden_oracle:.4} inv_rms_native={native_inv_rms:.6} inv_rms_oracle={oracle_inv_rms:.6} scale={scale:.4}"
                );
            }
        } else {
            eprintln!("[{tag}] missing native attention, post-norm, mlp-out, or layer trace");
        }
    } else {
        eprintln!("[{tag}] missing native or oracle layer trace data");
    }

    if let (Some(native), Some(trace)) = (native_linear_debug_trace, qwen35_trace_output) {
        let trace_linear_layer = trace.trace_linear_layer.or(trace_linear_layer).unwrap_or(0);
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
            trace
                .trace_linear_input_layernorm_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_qkv_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_z_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_post_conv_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_prepared_query_output
                .as_ref()
                .and_then(|value| flatten_token_bshd(value, trace_prefill_position)),
            trace
                .trace_linear_prepared_key_output
                .as_ref()
                .and_then(|value| flatten_token_bshd(value, trace_prefill_position)),
            trace
                .trace_linear_prepared_value_output
                .as_ref()
                .and_then(|value| flatten_token_bshd(value, trace_prefill_position)),
            trace
                .trace_linear_prepared_beta_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_prepared_g_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_direct_recurrent_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_norm_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_linear_token_mixer_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
        ) {
            let native_normed = decode_bf16_le(&native.normed);
            let native_qkv = decode_bf16_le(&native.qkv);
            let native_z = decode_bf16_le(&native.z);
            let native_conv_window = decode_bf16_le(&native.conv_window);
            let native_post_conv = decode_bf16_le(&native.post_conv);
            let native_packed = decode_f32_le(&native.packed);
            let native_rec_apply = decode_f32_le(&native.rec_apply);
            let native_attn = decode_bf16_le(&native.attn);
            let native_gated = decode_bf16_le(&native.gated);
            let native_proj_out = decode_bf16_le(&native.proj_out);

            let cfg = &engine.weights().config;
            let hidden_dim = cfg.hidden_size;
            let nv = cfg.linear_num_value_heads;
            let khd = cfg.linear_key_head_dim;
            let vhd = cfg.linear_value_head_dim;
            let key_dim = cfg.linear_num_key_heads * khd;
            let val_dim = nv * vhd;
            let qkv_dim = key_dim * 2 + val_dim;
            let packed_width = 2 * khd + vhd + 2;
            let q_scale = 1.0f32 / (khd as f32).sqrt();
            let mut oracle_packed = vec![0.0f32; nv * packed_width];
            if trace_linear_layer > 0 {
                if let (Some((_, _, _, _, native_layer_trace)), Some(oracle_input_hidden)) = (
                    native_prefill_trace,
                    qwen35_trace_output
                        .and_then(|trace| trace.decoder_layer_outputs.get(trace_linear_layer - 1))
                        .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
                ) {
                    if let Some(native_input_hidden_bytes) = native_layer_trace
                        .as_ref()
                        .and_then(|layers| layers.get(trace_linear_layer - 1))
                    {
                        let native_input_hidden = decode_bf16_le(native_input_hidden_bytes);
                        let native_input_norm_ref = compute_qwen_rms_norm_from_hidden_row(
                            &native_input_hidden,
                            &engine.weights().layers[trace_linear_layer].input_norm_w,
                            cfg.rms_norm_eps as f32,
                        )?;
                        let oracle_input_norm_ref = compute_qwen_rms_norm_from_hidden_row(
                            &oracle_input_hidden,
                            &engine.weights().layers[trace_linear_layer].input_norm_w,
                            cfg.rms_norm_eps as f32,
                        )?;
                        let (
                            native_norm_idx,
                            native_norm_v,
                            native_norm_ref_v,
                            native_norm_selfcheck_delta,
                        ) = max_abs_delta_details(&native_normed, &native_input_norm_ref);
                        let (
                            oracle_norm_idx,
                            oracle_norm_v,
                            oracle_norm_ref_v,
                            oracle_norm_selfcheck_delta,
                        ) = max_abs_delta_details(&oracle_normed, &oracle_input_norm_ref);
                        let (normed_idx, normed_native, normed_oracle, normed_delta_detail) =
                            max_abs_delta_details(&native_normed, &oracle_normed);
                        let (hidden_idx, hidden_native, hidden_oracle, hidden_delta_detail) =
                            max_abs_delta_details(&native_input_hidden, &oracle_input_hidden);
                        eprintln!(
                            "[{linear_tag}] layer={} input_norm_selfcheck native_idx={} native={:.4} ref={:.4} delta={:.4} oracle_idx={} oracle={:.4} ref_oracle={:.4} oracle_delta={:.4} native_vs_oracle={:.4}",
                            trace_linear_layer,
                            native_norm_idx,
                            native_norm_v,
                            native_norm_ref_v,
                            native_norm_selfcheck_delta,
                            oracle_norm_idx,
                            oracle_norm_v,
                            oracle_norm_ref_v,
                            oracle_norm_selfcheck_delta,
                            normed_delta_detail,
                        );
                        eprintln!(
                            "[{linear_tag}] layer={} input_hidden_detail idx={} hidden_native={:.4} hidden_oracle={:.4} hidden_delta={:.4}",
                            trace_linear_layer,
                            hidden_idx,
                            hidden_native,
                            hidden_oracle,
                            hidden_delta_detail,
                        );
                        let native_inv_rms = 1.0f32
                            / (mean_square(&native_input_hidden) + cfg.rms_norm_eps as f32).sqrt();
                        let oracle_inv_rms = 1.0f32
                            / (mean_square(&oracle_input_hidden) + cfg.rms_norm_eps as f32).sqrt();
                        let scale = read_weight_element_f32(
                            &engine.weights().layers[trace_linear_layer].input_norm_w,
                            normed_idx,
                        )? + 1.0;
                        eprintln!(
                            "[{linear_tag}] layer={} input_norm_detail idx={} hidden_native={:.4} hidden_oracle={:.4} inv_rms_native={:.6} inv_rms_oracle={:.6} scale={:.4} normed_native={:.4} normed_oracle={:.4}",
                            trace_linear_layer,
                            normed_idx,
                            native_input_hidden[normed_idx],
                            oracle_input_hidden[normed_idx],
                            native_inv_rms,
                            oracle_inv_rms,
                            scale,
                            normed_native,
                            normed_oracle,
                        );
                    }
                }
            }
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
            let direct_recurrent_delta = validate::max_abs_delta(
                &native_rec_apply[..val_dim.min(native_rec_apply.len())],
                &oracle_attn,
            );
            let conv_q_delta =
                validate::max_abs_delta(&native_post_conv[..key_dim], &oracle_post_conv[..key_dim]);
            let conv_k_delta = validate::max_abs_delta(
                &native_post_conv[key_dim..key_dim * 2],
                &oracle_post_conv[key_dim..key_dim * 2],
            );
            let conv_v_delta = validate::max_abs_delta(
                &native_post_conv[key_dim * 2..key_dim * 2 + val_dim],
                &oracle_post_conv[key_dim * 2..key_dim * 2 + val_dim],
            );
            if let Some(linear) = engine.weights().layers[trace_linear_layer].linear.as_ref() {
                let qkv_w = decode_gpu_buffer_f32(&linear.qkv_proj_w)?;
                let qkv_ref = apply_matmul_rhs_transposed_reference(
                    &native_normed,
                    &qkv_w,
                    native_qkv.len(),
                    hidden_dim,
                )?;
                let oracle_qkv_ref = apply_matmul_rhs_transposed_reference(
                    &oracle_normed,
                    &qkv_w,
                    oracle_qkv.len(),
                    hidden_dim,
                )?;
                let qkv_matmul_selfcheck_delta = validate::max_abs_delta(&native_qkv, &qkv_ref);
                let oracle_qkv_matmul_selfcheck_delta =
                    validate::max_abs_delta(&oracle_qkv, &oracle_qkv_ref);
                let (idx, native_v, ref_v, delta) = max_abs_delta_details(&native_qkv, &qkv_ref);
                eprintln!(
                    "[{linear_tag}] layer={} qkv_matmul_selfcheck_delta={:.4} oracle_qkv_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                    trace_linear_layer,
                    qkv_matmul_selfcheck_delta,
                    oracle_qkv_matmul_selfcheck_delta,
                    idx,
                    native_v,
                    ref_v,
                    delta
                );
                let (qkv_idx, qkv_native, qkv_oracle, qkv_delta_detail) =
                    max_abs_delta_details(&native_qkv, &oracle_qkv);
                eprintln!(
                    "[{linear_tag}] layer={} qkv_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_linear_layer, qkv_idx, qkv_native, qkv_oracle, qkv_delta_detail
                );
                let qkv_top_dims = weight_row_top_delta_dims(
                    &native_normed,
                    &oracle_normed,
                    &linear.qkv_proj_w,
                    engine.ordinal(),
                    qkv_idx,
                    hidden_dim,
                    6,
                )?;
                eprintln!(
                    "[{linear_tag}] layer={} qkv_row_detail idx={} top_dims={}",
                    trace_linear_layer,
                    qkv_idx,
                    format_logit_row_dims(&qkv_top_dims)
                );

                let z_w = decode_gpu_buffer_f32(&linear.z_proj_w)?;
                let z_ref = apply_matmul_rhs_transposed_reference(
                    &native_normed,
                    &z_w,
                    native_z.len(),
                    hidden_dim,
                )?;
                let oracle_z_ref = apply_matmul_rhs_transposed_reference(
                    &oracle_normed,
                    &z_w,
                    oracle_z.len(),
                    hidden_dim,
                )?;
                let z_matmul_selfcheck_delta = validate::max_abs_delta(&native_z, &z_ref);
                let oracle_z_matmul_selfcheck_delta =
                    validate::max_abs_delta(&oracle_z, &oracle_z_ref);
                let (idx, native_v, ref_v, delta) = max_abs_delta_details(&native_z, &z_ref);
                eprintln!(
                    "[{linear_tag}] layer={} z_matmul_selfcheck_delta={:.4} oracle_z_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                    trace_linear_layer,
                    z_matmul_selfcheck_delta,
                    oracle_z_matmul_selfcheck_delta,
                    idx,
                    native_v,
                    ref_v,
                    delta
                );
                let (z_idx, z_native, z_oracle, z_delta_detail) =
                    max_abs_delta_details(&native_z, &oracle_z);
                eprintln!(
                    "[{linear_tag}] layer={} z_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_linear_layer, z_idx, z_native, z_oracle, z_delta_detail
                );
                let z_top_dims = weight_row_top_delta_dims(
                    &native_normed,
                    &oracle_normed,
                    &linear.z_proj_w,
                    engine.ordinal(),
                    z_idx,
                    hidden_dim,
                    6,
                )?;
                eprintln!(
                    "[{linear_tag}] layer={} z_row_detail idx={} top_dims={}",
                    trace_linear_layer,
                    z_idx,
                    format_logit_row_dims(&z_top_dims)
                );

                let conv_w = decode_gpu_buffer_f32(&linear.conv1d_w)?;
                let conv_ref = apply_linear_conv_pack_row_reference(
                    &native_conv_window,
                    &conv_w,
                    qkv_dim,
                    cfg.linear_conv_kernel_dim,
                )?;
                let conv_selfcheck_delta = validate::max_abs_delta(&native_post_conv, &conv_ref);
                let (idx, native_v, ref_v, delta) =
                    max_abs_delta_details(&native_post_conv, &conv_ref);
                eprintln!(
                    "[{linear_tag}] layer={} conv_matmul_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                    trace_linear_layer,
                    conv_selfcheck_delta,
                    idx,
                    native_v,
                    ref_v,
                    delta
                );
                if let (Some(_), Some(oracle_conv_window)) = (
                    trace_prefill_position,
                    trace.trace_linear_qkv_output.as_ref().and_then(|value| {
                        extract_causal_conv_window_bsd(
                            value,
                            trace_prefill_position?,
                            qkv_dim,
                            cfg.linear_conv_kernel_dim,
                        )
                    }),
                ) {
                    let oracle_conv_window_delta =
                        validate::max_abs_delta(&native_conv_window, &oracle_conv_window);
                    let oracle_conv_ref = apply_linear_conv_pack_row_reference(
                        &oracle_conv_window,
                        &conv_w,
                        qkv_dim,
                        cfg.linear_conv_kernel_dim,
                    )?;
                    let oracle_conv_selfcheck_delta =
                        validate::max_abs_delta(&oracle_post_conv, &oracle_conv_ref);
                    let (idx, native_v, oracle_v, delta) =
                        max_abs_delta_details(&native_conv_window, &oracle_conv_window);
                    eprintln!(
                        "[{linear_tag}] layer={} conv_window_delta={:.4} oracle_conv_selfcheck_delta={:.4} idx={} native={:.4} oracle={:.4} delta={:.4}",
                        trace_linear_layer,
                        oracle_conv_window_delta,
                        oracle_conv_selfcheck_delta,
                        idx,
                        native_v,
                        oracle_v,
                        delta
                    );
                }

                let norm_w = decode_gpu_buffer_f32(&linear.norm_w)?;
                let native_gated_ref = apply_rms_norm_gated_reference(
                    &native_attn,
                    &native_z,
                    &norm_w,
                    cfg.rms_norm_eps as f32,
                )?;
                let oracle_gated_ref = apply_rms_norm_gated_reference(
                    &oracle_attn,
                    &oracle_z,
                    &norm_w,
                    cfg.rms_norm_eps as f32,
                )?;
                let (
                    native_gated_idx,
                    native_gated_v,
                    native_gated_ref_v,
                    native_gated_selfcheck_delta,
                ) = max_abs_delta_details(&native_gated, &native_gated_ref);
                let (
                    oracle_gated_idx,
                    oracle_gated_v,
                    oracle_gated_ref_v,
                    oracle_gated_selfcheck_delta,
                ) = max_abs_delta_details(&oracle_gated, &oracle_gated_ref);
                eprintln!(
                    "[{linear_tag}] layer={} gated_selfcheck native_idx={} native={:.4} ref={:.4} delta={:.4} oracle_idx={} oracle={:.4} ref_oracle={:.4} oracle_delta={:.4} native_vs_oracle={:.4}",
                    trace_linear_layer,
                    native_gated_idx,
                    native_gated_v,
                    native_gated_ref_v,
                    native_gated_selfcheck_delta,
                    oracle_gated_idx,
                    oracle_gated_v,
                    oracle_gated_ref_v,
                    oracle_gated_selfcheck_delta,
                    validate::max_abs_delta(&native_gated, &oracle_gated),
                );
            }
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
                    (native_packed[base + 2 * khd + vhd] - oracle_packed[base + 2 * khd + vhd])
                        .abs(),
                );
                gexp_delta = gexp_delta.max(
                    (native_packed[base + 2 * khd + vhd + 1]
                        - oracle_packed[base + 2 * khd + vhd + 1])
                        .abs(),
                );
            }

            eprintln!(
                "[{linear_tag}] layer={} normed_delta={:.4} qkv_delta={:.4} z_delta={:.4} post_conv_delta={:.4} conv_q_delta={:.4} conv_k_delta={:.4} conv_v_delta={:.4} packed_delta={:.4} q_delta={:.4} k_delta={:.4} v_delta={:.4} beta_delta={:.4} gexp_delta={:.4} direct_recurrent_delta={:.4} attn_delta={:.4} gated_delta={:.4} proj_out_delta={:.4}",
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
                direct_recurrent_delta,
                validate::max_abs_delta(&native_attn, &oracle_attn),
                validate::max_abs_delta(&native_gated, &oracle_gated),
                validate::max_abs_delta(&native_proj_out, &oracle_proj_out),
            );
        } else {
            eprintln!(
                "[{linear_tag}] layer={} missing flattenable qwen35 trace tensors",
                trace_linear_layer
            );
        }
    } else {
        let layer = trace_linear_layer.unwrap_or(0);
        eprintln!(
            "[{linear_tag}] layer={} missing native or qwen35 trace data",
            layer
        );
    }

    if let (Some(native), Some(trace)) = (native_layer3_full_attn_trace, qwen35_trace_output) {
        let trace_full_layer = trace.trace_full_layer.or(trace_full_layer).unwrap_or(3);
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
            trace
                .trace_full_q_and_gate_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_full_gate_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_full_k_proj_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_full_v_proj_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_full_prepared_query_output
                .as_ref()
                .and_then(|value| flatten_token_bhsd(value, trace_prefill_position)),
            trace
                .trace_full_prepared_key_output
                .as_ref()
                .and_then(|value| flatten_token_bhsd(value, trace_prefill_position)),
            trace
                .trace_full_prepared_value_output
                .as_ref()
                .and_then(|value| flatten_token_bhsd(value, trace_prefill_position)),
            trace
                .trace_full_attention_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
        ) {
            let native_input_norm = decode_bf16_le(&native.input_norm);
            let native_q_and_gate = decode_bf16_le(&native.q_and_gate);
            let native_q_proj = decode_bf16_le(&native.q_proj);
            let native_gate_proj = decode_bf16_le(&native.gate_proj);
            let native_k_proj = decode_bf16_le(&native.k_proj);
            let native_v_proj = decode_bf16_le(&native.v_proj);
            let native_q_prepared = decode_bf16_le(&native.q_prepared);
            let native_k_prepared = decode_bf16_le(&native.k_prepared);
            let native_q_rotated = decode_bf16_le(&native.q_rotated);
            let native_k_rotated = decode_bf16_le(&native.k_rotated);
            let native_v_prepared = decode_bf16_le(&native.v_prepared);
            let native_attn_pregate = decode_bf16_le(&native.attn_raw);
            let native_attn = decode_bf16_le(&native.attn_output);
            let oracle_q_rotated = trace
                .trace_full_rotated_query_output
                .as_ref()
                .and_then(|value| flatten_token_bhsd(value, trace_prefill_position));
            let oracle_k_rotated = trace
                .trace_full_rotated_key_output
                .as_ref()
                .and_then(|value| flatten_token_bhsd(value, trace_prefill_position));
            let oracle_attn_pregate = trace
                .trace_full_raw_attention_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position));
            let cfg = &engine.weights().config;
            let hidden_dim = cfg.hidden_size;
            let num_heads = cfg.num_attention_heads;
            let head_dim = cfg.head_dim;
            let q_dim = num_heads * head_dim;
            let q_proj_dim = q_dim * 2;
            let q_and_gate_delta = validate::max_abs_delta(&native_q_and_gate, &oracle_q_and_gate);
            let mut oracle_q_proj = vec![0.0f32; q_dim];
            for head in 0..num_heads {
                let src_base = head * head_dim * 2;
                let dst_base = head * head_dim;
                oracle_q_proj[dst_base..dst_base + head_dim]
                    .copy_from_slice(&oracle_q_and_gate[src_base..src_base + head_dim]);
            }
            let q_rotated_delta = oracle_q_rotated
                .as_ref()
                .map(|oracle| format!("{:.4}", validate::max_abs_delta(&native_q_rotated, oracle)))
                .unwrap_or_else(|| "n/a".to_string());
            let k_rotated_delta = oracle_k_rotated
                .as_ref()
                .map(|oracle| format!("{:.4}", validate::max_abs_delta(&native_k_rotated, oracle)))
                .unwrap_or_else(|| "n/a".to_string());
            let attn_pregate_delta = oracle_attn_pregate
                .as_ref()
                .map(|oracle| {
                    format!(
                        "{:.4}",
                        validate::max_abs_delta(&native_attn_pregate, oracle)
                    )
                })
                .unwrap_or_else(|| "n/a".to_string());
            eprintln!(
                "[{full_tag}] layer={} q_and_gate_delta={:.4} q_proj_delta={:.4} gate_proj_delta={:.4} k_proj_delta={:.4} v_proj_delta={:.4} q_prepared_delta={:.4} k_prepared_delta={:.4} q_rotated_delta={} k_rotated_delta={} v_prepared_delta={:.4} attn_pregate_delta={} attn_output_delta={:.4}",
                trace_full_layer,
                q_and_gate_delta,
                validate::max_abs_delta(&native_q_proj, &oracle_q_proj),
                validate::max_abs_delta(&native_gate_proj, &oracle_gate_proj),
                validate::max_abs_delta(&native_k_proj, &oracle_k_proj),
                validate::max_abs_delta(&native_v_proj, &oracle_v_proj),
                validate::max_abs_delta(&native_q_prepared, &oracle_q_prepared),
                validate::max_abs_delta(&native_k_prepared, &oracle_k_prepared),
                q_rotated_delta,
                k_rotated_delta,
                validate::max_abs_delta(&native_v_prepared, &oracle_v_prepared),
                attn_pregate_delta,
                validate::max_abs_delta(&native_attn, &oracle_attn),
            );
            if let Some(full) = engine.weights().layers[trace_full_layer].full.as_ref() {
                let q_proj_w = decode_gpu_buffer_f32(&full.q_proj_w)?;
                let q_matmul_ref = apply_matmul_rhs_transposed_reference(
                    &native_input_norm,
                    &q_proj_w,
                    q_proj_dim,
                    hidden_dim,
                )?;
                let q_matmul_selfcheck_delta =
                    validate::max_abs_delta(&native_q_and_gate, &q_matmul_ref);
                let (idx, native_v, ref_v, delta) =
                    max_abs_delta_details(&native_q_and_gate, &q_matmul_ref);
                eprintln!(
                    "[{full_tag}] layer={} q_proj_matmul_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                    trace_full_layer, q_matmul_selfcheck_delta, idx, native_v, ref_v, delta
                );
            }
            {
                let (idx, native_v, oracle_v, delta) =
                    max_abs_delta_details(&native_q_and_gate, &oracle_q_and_gate);
                eprintln!(
                    "[{full_tag}] layer={} q_and_gate_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_full_layer, idx, native_v, oracle_v, delta
                );
            }
            if let Some(oracle) = oracle_q_rotated.as_ref() {
                let (idx, native_v, oracle_v, delta) =
                    max_abs_delta_details(&native_q_rotated, oracle);
                eprintln!(
                    "[{full_tag}] layer={} q_rotated_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_full_layer, idx, native_v, oracle_v, delta
                );
            }
            {
                let (idx, native_v, oracle_v, delta) =
                    max_abs_delta_details(&native_q_prepared, &oracle_q_prepared);
                eprintln!(
                    "[{full_tag}] layer={} q_prepared_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_full_layer, idx, native_v, oracle_v, delta
                );
                if let Some(full) = engine.weights().layers[trace_full_layer].full.as_ref() {
                    let q_norm_w = decode_gpu_buffer_f32(&full.q_norm_w)?;
                    let detail = rms_norm_head_detail(
                        &native_q_proj,
                        &oracle_q_proj,
                        &q_norm_w,
                        head_dim,
                        idx,
                    )?;
                    eprintln!(
                        "[{full_tag}] layer={} q_prepared_detail idx={} head={} dim={} hidden_native={:.4} hidden_oracle={:.4} inv_rms_native={:.6} inv_rms_oracle={:.6} scale={:.4}",
                        trace_full_layer,
                        idx,
                        detail.head,
                        detail.dim,
                        detail.hidden_native,
                        detail.hidden_oracle,
                        detail.inv_rms_native,
                        detail.inv_rms_oracle,
                        detail.scale,
                    );
                    let q_proj_row_idx = detail.head * head_dim * 2 + detail.dim;
                    if trace_full_layer > 0 {
                        if let Some(oracle_input_hidden) = qwen35_trace_output
                            .and_then(|trace| trace.decoder_layer_outputs.get(trace_full_layer - 1))
                            .and_then(|value| flatten_token_bsd(value, trace_prefill_position))
                        {
                            let oracle_input_norm = compute_qwen_rms_norm_from_hidden_row(
                                &oracle_input_hidden,
                                &engine.weights().layers[trace_full_layer].input_norm_w,
                                cfg.rms_norm_eps as f32,
                            )?;
                            let top_dims = weight_row_top_delta_dims(
                                &native_input_norm,
                                &oracle_input_norm,
                                &full.q_proj_w,
                                engine.ordinal(),
                                q_proj_row_idx,
                                hidden_dim,
                                6,
                            )?;
                            eprintln!(
                                "[{full_tag}] layer={} q_proj_row_detail idx={} row={} top_dims={}",
                                trace_full_layer,
                                idx,
                                q_proj_row_idx,
                                format_logit_row_dims(&top_dims)
                            );
                        } else {
                            eprintln!(
                                "[{full_tag}] layer={} q_proj_row_detail idx={} missing_oracle_input_hidden",
                                trace_full_layer,
                                idx,
                            );
                        }
                    }
                }
            }
            {
                let (idx, native_v, oracle_v, delta) =
                    max_abs_delta_details(&native_k_prepared, &oracle_k_prepared);
                eprintln!(
                    "[{full_tag}] layer={} k_prepared_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_full_layer, idx, native_v, oracle_v, delta
                );
                if let Some(full) = engine.weights().layers[trace_full_layer].full.as_ref() {
                    let k_norm_w = decode_gpu_buffer_f32(&full.k_norm_w)?;
                    let detail = rms_norm_head_detail(
                        &native_k_proj,
                        &oracle_k_proj,
                        &k_norm_w,
                        head_dim,
                        idx,
                    )?;
                    eprintln!(
                        "[{full_tag}] layer={} k_prepared_detail idx={} head={} dim={} hidden_native={:.4} hidden_oracle={:.4} inv_rms_native={:.6} inv_rms_oracle={:.6} scale={:.4}",
                        trace_full_layer,
                        idx,
                        detail.head,
                        detail.dim,
                        detail.hidden_native,
                        detail.hidden_oracle,
                        detail.inv_rms_native,
                        detail.inv_rms_oracle,
                        detail.scale,
                    );
                }
            }
            if let Some(position) = trace_prefill_position {
                if let Some(full) = engine.weights().layers[trace_full_layer].full.as_ref() {
                    let eps = cfg.rms_norm_eps as f32;
                    let q_norm_w = decode_gpu_buffer_f32(&full.q_norm_w)?;
                    let q_norm_ref = apply_rms_norm_reference(
                        &native_q_proj,
                        &q_norm_w,
                        num_heads,
                        head_dim,
                        eps,
                    )?;
                    let q_norm_selfcheck_delta =
                        validate::max_abs_delta(&native_q_prepared, &q_norm_ref);
                    let (idx, native_v, ref_v, delta) =
                        max_abs_delta_details(&native_q_prepared, &q_norm_ref);
                    eprintln!(
                        "[{full_tag}] layer={} q_norm_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                        trace_full_layer, q_norm_selfcheck_delta, idx, native_v, ref_v, delta
                    );
                    let num_kv_heads = cfg.num_key_value_heads;
                    let k_norm_w = decode_gpu_buffer_f32(&full.k_norm_w)?;
                    let k_norm_ref = apply_rms_norm_reference(
                        &native_k_proj,
                        &k_norm_w,
                        num_kv_heads,
                        head_dim,
                        eps,
                    )?;
                    let k_norm_selfcheck_delta =
                        validate::max_abs_delta(&native_k_prepared, &k_norm_ref);
                    let (idx, native_v, ref_v, delta) =
                        max_abs_delta_details(&native_k_prepared, &k_norm_ref);
                    eprintln!(
                        "[{full_tag}] layer={} k_norm_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                        trace_full_layer, k_norm_selfcheck_delta, idx, native_v, ref_v, delta
                    );
                }
                let q_rope_ref = apply_rope_reference_for_position(
                    &native_q_prepared,
                    engine.rotary(),
                    position,
                    num_heads,
                    head_dim,
                )?;
                let q_rope_ref_delta = validate::max_abs_delta(&native_q_rotated, &q_rope_ref);
                let (idx, native_v, ref_v, delta) =
                    max_abs_delta_details(&native_q_rotated, &q_rope_ref);
                eprintln!(
                    "[{full_tag}] layer={} q_rope_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                    trace_full_layer, q_rope_ref_delta, idx, native_v, ref_v, delta
                );
                let num_kv_heads = cfg.num_key_value_heads;
                let k_rope_ref = apply_rope_reference_for_position(
                    &native_k_prepared,
                    engine.rotary(),
                    position,
                    num_kv_heads,
                    head_dim,
                )?;
                let k_rope_ref_delta = validate::max_abs_delta(&native_k_rotated, &k_rope_ref);
                let (idx, native_v, ref_v, delta) =
                    max_abs_delta_details(&native_k_rotated, &k_rope_ref);
                eprintln!(
                    "[{full_tag}] layer={} k_rope_selfcheck_delta={:.4} idx={} native={:.4} ref={:.4} delta={:.4}",
                    trace_full_layer, k_rope_ref_delta, idx, native_v, ref_v, delta
                );
            }
            if let Some(oracle) = oracle_k_rotated.as_ref() {
                let (idx, native_v, oracle_v, delta) =
                    max_abs_delta_details(&native_k_rotated, oracle);
                eprintln!(
                    "[{full_tag}] layer={} k_rotated_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_full_layer, idx, native_v, oracle_v, delta
                );
            }
            if let Some(oracle) = oracle_attn_pregate.as_ref() {
                let (idx, native_v, oracle_v, delta) =
                    max_abs_delta_details(&native_attn_pregate, oracle);
                eprintln!(
                    "[{full_tag}] layer={} attn_pregate_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                    trace_full_layer, idx, native_v, oracle_v, delta
                );
            }
            let (idx, native_v, oracle_v, delta) =
                max_abs_delta_details(&native_attn, &oracle_attn);
            eprintln!(
                "[{full_tag}] layer={} attn_output_max idx={} native={:.4} oracle={:.4} delta={:.4}",
                trace_full_layer, idx, native_v, oracle_v, delta
            );
        } else {
            eprintln!(
                "[{full_tag}] layer={} missing flattenable qwen35 trace tensors",
                trace_full_layer
            );
        }
    } else {
        let layer = trace_full_layer.unwrap_or(3);
        eprintln!(
            "[{full_tag}] layer={} missing native or qwen35 trace data",
            layer
        );
    }

    if let (Some(native), Some(trace)) = (native_mlp_debug_trace, qwen35_trace_output) {
        let trace_mlp_layer = trace.trace_mlp_layer.or(trace_mlp_layer).unwrap_or(19);
        if let (
            Some(oracle_post_norm),
            Some(oracle_gate_proj),
            Some(oracle_up_proj),
            Some(oracle_swiglu),
            Some(oracle_down_proj),
        ) = (
            trace
                .trace_mlp_post_attention_layernorm_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_mlp_gate_proj_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_mlp_up_proj_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_mlp_activated_hidden
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
            trace
                .trace_mlp_down_proj_output
                .as_ref()
                .and_then(|value| flatten_token_bsd(value, trace_prefill_position)),
        ) {
            let native_post_norm = decode_bf16_le(&native.post_norm);
            let native_gate_proj = decode_bf16_le(&native.gate_proj);
            let native_up_proj = decode_bf16_le(&native.up_proj);
            let native_swiglu = decode_bf16_le(&native.swiglu);
            let native_down_proj = decode_bf16_le(&native.down_proj);
            let (post_norm_idx, post_norm_native, post_norm_oracle, post_norm_delta) =
                max_abs_delta_details(&native_post_norm, &oracle_post_norm);
            let (gate_proj_idx, _gate_proj_native, _gate_proj_oracle, gate_proj_delta) =
                max_abs_delta_details(&native_gate_proj, &oracle_gate_proj);
            let (up_proj_idx, _up_proj_native, _up_proj_oracle, up_proj_delta) =
                max_abs_delta_details(&native_up_proj, &oracle_up_proj);
            let (down_proj_idx, _down_proj_native, _down_proj_oracle, down_proj_delta) =
                max_abs_delta_details(&native_down_proj, &oracle_down_proj);
            eprintln!(
                "[{mlp_tag}] layer={} post_norm_delta={:.4} gate_proj_delta={:.4} up_proj_delta={:.4} swiglu_delta={:.4} down_proj_delta={:.4}",
                trace_mlp_layer,
                post_norm_delta,
                gate_proj_delta,
                up_proj_delta,
                validate::max_abs_delta(&native_swiglu, &oracle_swiglu),
                down_proj_delta,
            );
            let hidden = engine.weights().config.hidden_size;
            let (native_gate_proj_v, native_gate_proj_ref, native_gate_proj_selfcheck) =
                matmul_row_selfcheck_details(
                    &native_post_norm,
                    &engine.weights().layers[trace_mlp_layer].gate_proj_w,
                    engine.ordinal(),
                    gate_proj_idx,
                    hidden,
                    &native_gate_proj,
                )?;
            let (oracle_gate_proj_v, oracle_gate_proj_ref, oracle_gate_proj_selfcheck) =
                matmul_row_selfcheck_details(
                    &oracle_post_norm,
                    &engine.weights().layers[trace_mlp_layer].gate_proj_w,
                    engine.ordinal(),
                    gate_proj_idx,
                    hidden,
                    &oracle_gate_proj,
                )?;
            eprintln!(
                "[{mlp_tag}] layer={} gate_proj_matmul_selfcheck idx={} native={:.4} ref={:.4} delta={:.4} oracle={:.4} ref_oracle={:.4} oracle_delta={:.4} native_vs_oracle={:.4}",
                trace_mlp_layer,
                gate_proj_idx,
                native_gate_proj_v,
                native_gate_proj_ref,
                native_gate_proj_selfcheck,
                oracle_gate_proj_v,
                oracle_gate_proj_ref,
                oracle_gate_proj_selfcheck,
                gate_proj_delta,
            );
            eprintln!(
                "[{mlp_tag}] layer={} gate_proj_row_detail idx={} top_dims={}",
                trace_mlp_layer,
                gate_proj_idx,
                format_weight_row_top_delta_dims(&weight_row_top_delta_dims(
                    &native_post_norm,
                    &oracle_post_norm,
                    &engine.weights().layers[trace_mlp_layer].gate_proj_w,
                    engine.ordinal(),
                    gate_proj_idx,
                    hidden,
                    6,
                )?)
            );
            let (native_up_proj_v, native_up_proj_ref, native_up_proj_selfcheck) =
                matmul_row_selfcheck_details(
                    &native_post_norm,
                    &engine.weights().layers[trace_mlp_layer].up_proj_w,
                    engine.ordinal(),
                    up_proj_idx,
                    hidden,
                    &native_up_proj,
                )?;
            let (oracle_up_proj_v, oracle_up_proj_ref, oracle_up_proj_selfcheck) =
                matmul_row_selfcheck_details(
                    &oracle_post_norm,
                    &engine.weights().layers[trace_mlp_layer].up_proj_w,
                    engine.ordinal(),
                    up_proj_idx,
                    hidden,
                    &oracle_up_proj,
                )?;
            eprintln!(
                "[{mlp_tag}] layer={} up_proj_matmul_selfcheck idx={} native={:.4} ref={:.4} delta={:.4} oracle={:.4} ref_oracle={:.4} oracle_delta={:.4} native_vs_oracle={:.4}",
                trace_mlp_layer,
                up_proj_idx,
                native_up_proj_v,
                native_up_proj_ref,
                native_up_proj_selfcheck,
                oracle_up_proj_v,
                oracle_up_proj_ref,
                oracle_up_proj_selfcheck,
                up_proj_delta,
            );
            eprintln!(
                "[{mlp_tag}] layer={} up_proj_row_detail idx={} top_dims={}",
                trace_mlp_layer,
                up_proj_idx,
                format_weight_row_top_delta_dims(&weight_row_top_delta_dims(
                    &native_post_norm,
                    &oracle_post_norm,
                    &engine.weights().layers[trace_mlp_layer].up_proj_w,
                    engine.ordinal(),
                    up_proj_idx,
                    hidden,
                    6,
                )?)
            );
            eprintln!(
                "[{mlp_tag}] layer={} swiglu_selfcheck native={:.4} oracle={:.4}",
                trace_mlp_layer,
                swiglu_selfcheck_delta(&native_gate_proj, &native_up_proj, &native_swiglu),
                swiglu_selfcheck_delta(&oracle_gate_proj, &oracle_up_proj, &oracle_swiglu),
            );
            let intermediate = engine.weights().config.intermediate_size;
            let (native_down_proj_v, native_down_proj_ref, native_down_proj_selfcheck) =
                matmul_row_selfcheck_details(
                    &native_swiglu,
                    &engine.weights().layers[trace_mlp_layer].down_proj_w,
                    engine.ordinal(),
                    down_proj_idx,
                    intermediate,
                    &native_down_proj,
                )?;
            let (oracle_down_proj_v, oracle_down_proj_ref, oracle_down_proj_selfcheck) =
                matmul_row_selfcheck_details(
                    &oracle_swiglu,
                    &engine.weights().layers[trace_mlp_layer].down_proj_w,
                    engine.ordinal(),
                    down_proj_idx,
                    intermediate,
                    &oracle_down_proj,
                )?;
            eprintln!(
                "[{mlp_tag}] layer={} down_proj_matmul_selfcheck idx={} native={:.4} ref={:.4} delta={:.4} oracle={:.4} ref_oracle={:.4} oracle_delta={:.4} native_vs_oracle={:.4}",
                trace_mlp_layer,
                down_proj_idx,
                native_down_proj_v,
                native_down_proj_ref,
                native_down_proj_selfcheck,
                oracle_down_proj_v,
                oracle_down_proj_ref,
                oracle_down_proj_selfcheck,
                down_proj_delta,
            );
            eprintln!(
                "[{mlp_tag}] layer={} down_proj_row_detail idx={} top_dims={}",
                trace_mlp_layer,
                down_proj_idx,
                format_mlp_down_proj_contributors(
                    &weight_row_top_delta_dims(
                        &native_swiglu,
                        &oracle_swiglu,
                        &engine.weights().layers[trace_mlp_layer].down_proj_w,
                        engine.ordinal(),
                        down_proj_idx,
                        intermediate,
                        6,
                    )?,
                    &native_gate_proj,
                    &oracle_gate_proj,
                    &native_up_proj,
                    &oracle_up_proj,
                    &native_swiglu,
                    &oracle_swiglu,
                )
            );
            if let Some(dims) = tracked_logit_dims.as_ref() {
                let tracked_hidden_dims = dims.iter().copied().take(4).collect::<Vec<_>>();
                eprintln!(
                    "[{mlp_tag}] layer={} tracked_hidden_dims={}",
                    trace_mlp_layer,
                    tracked_hidden_dims
                        .iter()
                        .map(|dim| dim.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
                eprintln!(
                    "[{mlp_tag}] layer={} tracked_hidden_stage dims={}",
                    trace_mlp_layer,
                    tracked_hidden_dims
                        .iter()
                        .filter(|dim| {
                            **dim < native_post_norm.len()
                                && **dim < oracle_post_norm.len()
                                && **dim < native_down_proj.len()
                                && **dim < oracle_down_proj.len()
                        })
                        .map(|dim| {
                            format!(
                                "{dim}:pn=({:.4},{:.4}) dp=({:.4},{:.4})",
                                native_post_norm[*dim],
                                oracle_post_norm[*dim],
                                native_down_proj[*dim],
                                oracle_down_proj[*dim],
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                for &hidden_dim in &tracked_hidden_dims {
                    if hidden_dim >= native_down_proj.len() || hidden_dim >= oracle_down_proj.len()
                    {
                        continue;
                    }
                    let top_intermediate = weight_row_top_delta_dims(
                        &native_swiglu,
                        &oracle_swiglu,
                        &engine.weights().layers[trace_mlp_layer].down_proj_w,
                        engine.ordinal(),
                        hidden_dim,
                        intermediate,
                        4,
                    )?;
                    eprintln!(
                        "[{mlp_tag}] layer={} hidden_dim={} down_proj_contributors={}",
                        trace_mlp_layer,
                        hidden_dim,
                        format_mlp_down_proj_contributors(
                            &top_intermediate,
                            &native_gate_proj,
                            &oracle_gate_proj,
                            &native_up_proj,
                            &oracle_up_proj,
                            &native_swiglu,
                            &oracle_swiglu,
                        )
                    );
                }
                let aggregate_intermediate = aggregate_down_proj_contributors(
                    &native_swiglu,
                    &oracle_swiglu,
                    &engine.weights().layers[trace_mlp_layer].down_proj_w,
                    engine.ordinal(),
                    &tracked_hidden_dims,
                    intermediate,
                    6,
                )?;
                eprintln!(
                    "[{mlp_tag}] layer={} tracked_hidden_intermediate_aggregate={}",
                    trace_mlp_layer,
                    format_mlp_intermediate_aggregate(
                        &aggregate_intermediate,
                        &native_gate_proj,
                        &oracle_gate_proj,
                        &native_up_proj,
                        &oracle_up_proj,
                    )
                );
            }
            if trace_prefill_position.is_none() {
                if let (Some((_, native_attn_trace, _, _, _)), Some(oracle_attn_trace)) = (
                    native_prefill_trace,
                    oracle_output.layer_attn_residual_states.as_ref(),
                ) {
                    if let Some(native_attn_trace) = native_attn_trace.as_ref() {
                        if trace_mlp_layer >= layer_offset {
                            let rel_layer = trace_mlp_layer - layer_offset;
                            if rel_layer < native_attn_trace.len()
                                && trace_mlp_layer < oracle_attn_trace.len()
                            {
                                let b64 = base64::engine::general_purpose::STANDARD;
                                let oracle_attn_bytes = b64
                                .decode(&oracle_attn_trace[trace_mlp_layer])
                                .map_err(|e| anyhow::anyhow!(
                                    "decode oracle layer_attn_residual_states[{trace_mlp_layer}] for mlp detail: {e}"
                                ))?;
                                let native_attn_hidden =
                                    decode_bf16_le(&native_attn_trace[rel_layer]);
                                let oracle_attn_hidden = decode_bf16_le(&oracle_attn_bytes);
                                let native_mean_sq = mean_square(&native_attn_hidden);
                                let oracle_mean_sq = mean_square(&oracle_attn_hidden);
                                let native_inv_rms = 1.0f32
                                    / (native_mean_sq
                                        + engine.weights().config.rms_norm_eps as f32)
                                        .sqrt();
                                let oracle_inv_rms = 1.0f32
                                    / (oracle_mean_sq
                                        + engine.weights().config.rms_norm_eps as f32)
                                        .sqrt();
                                let scale = read_weight_element_f32(
                                    &engine.weights().layers[trace_mlp_layer].post_attn_norm_w,
                                    post_norm_idx,
                                )? + 1.0;
                                eprintln!(
                                "[{mlp_tag}] layer={} post_norm_detail idx={} hidden_native={:.4} hidden_oracle={:.4} inv_rms_native={native_inv_rms:.6} inv_rms_oracle={oracle_inv_rms:.6} scale={scale:.4} normed_native={:.4} normed_oracle={:.4}",
                                trace_mlp_layer,
                                post_norm_idx,
                                native_attn_hidden[post_norm_idx],
                                oracle_attn_hidden[post_norm_idx],
                                post_norm_native,
                                post_norm_oracle,
                            );
                            }
                        }
                    }
                }
            }
        } else {
            eprintln!(
                "[{mlp_tag}] layer={} missing flattenable qwen35 trace tensors",
                trace_mlp_layer
            );
        }
    } else {
        let layer = trace_mlp_layer.unwrap_or(19);
        eprintln!(
            "[{mlp_tag}] layer={} missing native or qwen35 trace data",
            layer
        );
    }

    Ok(())
}

fn qwen35_layer_kind_label(config: &qwen35::config::TextConfig, layer: usize) -> &'static str {
    if config.is_full_attention(layer) {
        "full"
    } else {
        "linear"
    }
}

fn emit_qwen35_restart_report(
    config: &qwen35::config::TextConfig,
    tag: &str,
    source_layer: usize,
    start_layer: usize,
    native_prefill: &prefill_engine::PrefillResult,
    oracle_output: &oracle::OracleOutput,
    qwen35_trace_output: Option<&oracle::Qwen35TraceOutput>,
    trace_prefill_position: Option<usize>,
) -> Result<()> {
    let logit_delta =
        validate::max_abs_delta(&native_prefill.logits, &oracle_output.prefill_logits);
    if trace_prefill_position.is_some() {
        eprintln!(
            "[{tag}] source_layer={source_layer} start_layer={start_layer} last_token_logit_delta={logit_delta:.4}"
        );
    } else {
        eprintln!(
            "[{tag}] source_layer={source_layer} start_layer={start_layer} logit_delta={logit_delta:.4}"
        );
    }
    let (logit_idx, native_logit, oracle_logit, logit_max_delta) =
        max_abs_delta_details(&native_prefill.logits, &oracle_output.prefill_logits);
    if trace_prefill_position.is_some() {
        eprintln!(
            "[{tag}] last_token_logit_max idx={logit_idx} native={native_logit:.4} oracle={oracle_logit:.4} delta={logit_max_delta:.4}"
        );
    } else {
        eprintln!(
            "[{tag}] logit_max idx={logit_idx} native={native_logit:.4} oracle={oracle_logit:.4} delta={logit_max_delta:.4}"
        );
    }

    if let Some(position) = trace_prefill_position {
        if let (Some(native_hidden_trace), Some(trace)) = (
            native_prefill.layer_hidden_trace.as_ref(),
            qwen35_trace_output,
        ) {
            for (offset, native_hidden_bytes) in native_hidden_trace.iter().enumerate() {
                let layer = start_layer + offset;
                let Some(oracle_hidden) = trace
                    .decoder_layer_outputs
                    .get(layer)
                    .and_then(|value| flatten_token_bsd(value, Some(position)))
                else {
                    break;
                };
                let native_hidden = decode_bf16_le(native_hidden_bytes);
                let layer_delta = validate::max_abs_delta(&native_hidden, &oracle_hidden);
                let layer_kind = qwen35_layer_kind_label(config, layer);
                eprintln!(
                    "[{tag}] position={} layer={} kind={} layer_delta={layer_delta:.4}",
                    position, layer, layer_kind
                );
            }
        }
    } else if let (Some(native_hidden_trace), Some(oracle_hidden_trace)) = (
        native_prefill.layer_hidden_trace.as_ref(),
        oracle_output.layer_hidden_states.as_ref(),
    ) {
        let b64 = base64::engine::general_purpose::STANDARD;
        for (offset, native_hidden_bytes) in native_hidden_trace.iter().enumerate() {
            let layer = start_layer + offset;
            if layer >= oracle_hidden_trace.len() {
                break;
            }
            let oracle_hidden_bytes = b64.decode(&oracle_hidden_trace[layer]).map_err(|e| {
                anyhow::anyhow!("decode restart oracle layer_hidden_states[{layer}]: {e}")
            })?;
            let oracle_hidden = decode_bf16_le(&oracle_hidden_bytes);
            let native_hidden = decode_bf16_le(native_hidden_bytes);
            let layer_delta = validate::max_abs_delta(&native_hidden, &oracle_hidden);
            let layer_kind = qwen35_layer_kind_label(config, layer);
            eprintln!("[{tag}] layer={layer} kind={layer_kind} layer_delta={layer_delta:.4}");
        }
    }

    if let (Some(native_final_norm_trace), Some(oracle_final_norm_b64)) = (
        native_prefill.final_norm_trace.as_ref(),
        oracle_output.prefill_hidden.as_ref(),
    ) {
        let b64 = base64::engine::general_purpose::STANDARD;
        let oracle_final_norm_bytes = b64
            .decode(oracle_final_norm_b64)
            .map_err(|e| anyhow::anyhow!("decode restart oracle prefill_hidden: {e}"))?;
        let native_final_norm = decode_bf16_le(native_final_norm_trace);
        let oracle_final_norm = decode_bf16_le(&oracle_final_norm_bytes);
        let (idx, native_v, oracle_v, delta) =
            max_abs_delta_details(&native_final_norm, &oracle_final_norm);
        if trace_prefill_position.is_some() {
            eprintln!(
                "[{tag}] last_token_final_norm_max idx={idx} native={native_v:.4} oracle={oracle_v:.4} delta={delta:.4}"
            );
        } else {
            eprintln!(
                "[{tag}] final_norm_max idx={idx} native={native_v:.4} oracle={oracle_v:.4} delta={delta:.4}"
            );
        }
    }

    Ok(())
}

fn emit_qwen35_restart_position_scan_report(
    engine: &DecodeEngine,
    tag: &str,
    start_layer: usize,
    hidden_bf16: &[u8],
    positions: &[usize],
    trace_linear_layer: Option<usize>,
    trace_full_layer: Option<usize>,
    trace_mlp_layer: Option<usize>,
    qwen35_trace_output: &oracle::Qwen35TraceOutput,
) -> Result<()> {
    let num_tail_layers = engine
        .weights()
        .config
        .num_hidden_layers
        .saturating_sub(start_layer);
    let mut worst_by_layer: Vec<Option<(usize, f32)>> = vec![None; num_tail_layers];
    let mut worst_final_hidden_logit: Option<(usize, f32)> = None;

    for &position in positions {
        let replay = engine.prefill_tail_from_hidden_with_trace(
            hidden_bf16,
            start_layer,
            trace_linear_layer.filter(|layer| *layer >= start_layer),
            trace_full_layer.filter(|layer| *layer >= start_layer),
            trace_mlp_layer.filter(|layer| *layer >= start_layer),
            Some(position),
        )?;

        let native_layer_trace = replay
            .layer_hidden_trace
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restart position scan missing native layer trace"))?;

        let mut max_layer_delta = -1.0f32;
        let mut max_layer = start_layer;
        for (rel_layer, native_layer_bytes) in native_layer_trace.iter().enumerate() {
            let layer = start_layer + rel_layer;
            let oracle_layer = qwen35_trace_output
                .decoder_layer_outputs
                .get(layer)
                .and_then(|value| flatten_token_bsd(value, Some(position)))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "restart position scan missing oracle decoder_layer_outputs[{layer}] for position {position}"
                    )
                })?;
            let native_layer = decode_bf16_le(native_layer_bytes);
            let layer_delta = validate::max_abs_delta(&native_layer, &oracle_layer);
            if layer_delta > max_layer_delta {
                max_layer_delta = layer_delta;
                max_layer = layer;
            }
            let slot = &mut worst_by_layer[rel_layer];
            if slot.map(|(_, delta)| layer_delta > delta).unwrap_or(true) {
                *slot = Some((position, layer_delta));
            }
        }

        let native_final_hidden = native_layer_trace
            .last()
            .map(|bytes| decode_bf16_le(bytes))
            .ok_or_else(|| anyhow::anyhow!("restart position scan missing final hidden trace"))?;
        let oracle_final_hidden = qwen35_trace_output
            .decoder_layer_outputs
            .last()
            .and_then(|value| flatten_token_bsd(value, Some(position)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "restart position scan missing oracle final hidden for position {position}"
                )
            })?;
        let native_hidden_logits =
            compute_qwen_logits_from_hidden_row(engine, &native_final_hidden)?;
        let oracle_hidden_logits =
            compute_qwen_logits_from_hidden_row(engine, &oracle_final_hidden)?;
        let final_hidden_logit_delta =
            validate::max_abs_delta(&native_hidden_logits, &oracle_hidden_logits);
        if worst_final_hidden_logit
            .map(|(_, delta)| final_hidden_logit_delta > delta)
            .unwrap_or(true)
        {
            worst_final_hidden_logit = Some((position, final_hidden_logit_delta));
        }

        eprintln!(
            "[{tag}] position={} final_hidden_logit_delta={:.4} max_layer={} max_layer_delta={:.4}",
            position, final_hidden_logit_delta, max_layer, max_layer_delta
        );
    }

    if let Some((position, delta)) = worst_final_hidden_logit {
        eprintln!(
            "[{tag}] worst_final_hidden_logit_position={} delta={:.4}",
            position, delta
        );
    }
    for (rel_layer, worst) in worst_by_layer.into_iter().enumerate() {
        if let Some((position, delta)) = worst {
            eprintln!(
                "[{tag}] layer={} worst_position={} layer_delta={:.4}",
                start_layer + rel_layer,
                position,
                delta
            );
        }
    }

    Ok(())
}

fn emit_qwen35_restart_source_tail_report(
    engine: &DecodeEngine,
    tag: &str,
    source_layer: usize,
    source_hidden_f32: &[f32],
    qwen35_trace_output: &oracle::Qwen35TraceOutput,
    trace_position: usize,
) -> Result<()> {
    let hidden_dim = engine.weights().config.hidden_size;
    let start = trace_position
        .checked_mul(hidden_dim)
        .ok_or_else(|| anyhow::anyhow!("trace position hidden offset overflow"))?;
    let end = start + hidden_dim;
    if end > source_hidden_f32.len() {
        anyhow::bail!(
            "trace position {} out of range for source hidden rows (len={} hidden_dim={})",
            trace_position,
            source_hidden_f32.len(),
            hidden_dim
        );
    }
    let source_hidden_row = &source_hidden_f32[start..end];
    if let Some(oracle_source_hidden) = qwen35_trace_output
        .decoder_layer_outputs
        .get(source_layer)
        .and_then(|value| flatten_token_bsd(value, Some(trace_position)))
    {
        let (idx, native_v, oracle_v, delta) =
            max_abs_delta_details(source_hidden_row, &oracle_source_hidden);
        let top_dims = top_abs_delta_dims(source_hidden_row, &oracle_source_hidden, 6);
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_delta={delta:.4}"
        );
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_max idx={idx} native={native_v:.4} oracle={oracle_v:.4}"
        );
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_detail top_dims={}",
            format_top_delta_dims(&top_dims)
        );
    }

    if let Some(oracle_final_norm) = qwen35_trace_output
        .trace_position_prefill_final_norm_output
        .as_ref()
        .and_then(flatten_json_vector)
    {
        let source_final_norm = compute_qwen_final_norm_from_hidden_row(engine, source_hidden_row)?;
        let (idx, native_v, oracle_v, delta) =
            max_abs_delta_details(&source_final_norm, &oracle_final_norm);
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_final_norm_delta={delta:.4}"
        );
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_final_norm_max idx={idx} native={native_v:.4} oracle={oracle_v:.4}"
        );
    }

    if let Some(oracle_logits) = qwen35_trace_output
        .trace_position_prefill_logits
        .as_ref()
        .and_then(flatten_json_vector)
    {
        let source_logits = compute_qwen_logits_from_hidden_row(engine, source_hidden_row)?;
        let (idx, native_v, oracle_v, delta) =
            max_abs_delta_details(&source_logits, &oracle_logits);
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_logit_delta={delta:.4}"
        );
        eprintln!(
            "[{tag}] source_layer={source_layer} position={trace_position} source_hidden_logit_max idx={idx} native={native_v:.4} oracle={oracle_v:.4}"
        );
    }

    Ok(())
}

fn top_logits(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k.min(indexed.len()));
    indexed
}

fn max_abs_delta_details(lhs: &[f32], rhs: &[f32]) -> (usize, f32, f32, f32) {
    let mut best = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (idx, (l, r)) in lhs.iter().copied().zip(rhs.iter().copied()).enumerate() {
        let delta = (l - r).abs();
        if delta > best.3 {
            best = (idx, l, r, delta);
        }
    }
    best
}

fn mean_square_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .take(len)
        .map(|(l, r)| {
            let delta = l - r;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn mean_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .take(len)
        .map(|(l, r)| (l - r).abs())
        .sum::<f32>()
        / len as f32
}

fn top_abs_delta_dims(lhs: &[f32], rhs: &[f32], top_k: usize) -> Vec<(usize, f32, f32, f32)> {
    let mut dims = lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .enumerate()
        .map(|(idx, (native_v, oracle_v))| (idx, native_v, oracle_v, (native_v - oracle_v).abs()))
        .collect::<Vec<_>>();
    dims.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
    dims.truncate(top_k.min(dims.len()));
    dims
}

fn format_top_delta_dims(entries: &[(usize, f32, f32, f32)]) -> String {
    entries
        .iter()
        .map(|(idx, native_v, oracle_v, delta)| {
            format!("{idx}:n={native_v:.4} o={oracle_v:.4} d={delta:.4}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn mean_square(values: &[f32]) -> f32 {
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    sum_sq / values.len() as f32
}

fn read_buffer_f32_range(
    buf: &GpuBuffer,
    ordinal: usize,
    start_elem: usize,
    elem_count: usize,
) -> Result<Vec<f32>> {
    match buf.dtype() {
        ScalarType::BF16 => {
            let mut bytes = vec![0u8; elem_count * 2];
            gpu_hal::copy_d2h(
                ordinal,
                bytes.as_mut_ptr() as *mut c_void,
                buf.offset_ptr(start_elem * 2),
                bytes.len(),
            )
            .map_err(|e| anyhow::anyhow!("buffer range D2H BF16: {e}"))?;
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect())
        }
        ScalarType::F32 => {
            let mut bytes = vec![0u8; elem_count * 4];
            gpu_hal::copy_d2h(
                ordinal,
                bytes.as_mut_ptr() as *mut c_void,
                buf.offset_ptr(start_elem * 4),
                bytes.len(),
            )
            .map_err(|e| anyhow::anyhow!("buffer range D2H F32: {e}"))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        other => Err(anyhow::anyhow!(
            "unsupported buffer dtype for trace range read: {other:?}"
        )),
    }
}

fn read_weight_element_f32(buf: &GpuBuffer, idx: usize) -> Result<f32> {
    let bytes = buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("norm weight D2H: {e}"))?;
    match buf.dtype() {
        ScalarType::BF16 => {
            let start = idx * 2;
            let bits = [bytes[start], bytes[start + 1]];
            Ok(half::bf16::from_le_bytes(bits).to_f32())
        }
        ScalarType::F32 => {
            let start = idx * 4;
            let bits = [
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ];
            Ok(f32::from_le_bytes(bits))
        }
        other => Err(anyhow::anyhow!(
            "unsupported norm weight dtype for trace: {other:?}"
        )),
    }
}

fn read_buffer_all_f32(buf: &GpuBuffer) -> Result<Vec<f32>> {
    let bytes = buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("buffer D2H: {e}"))?;
    match buf.dtype() {
        ScalarType::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect()),
        ScalarType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        other => Err(anyhow::anyhow!(
            "unsupported buffer dtype for full read: {other:?}"
        )),
    }
}

fn compute_qwen_rms_norm_from_hidden_row(
    hidden_row: &[f32],
    weight_buf: &GpuBuffer,
    eps: f32,
) -> Result<Vec<f32>> {
    let hidden_dim = hidden_row.len();
    let weights = read_buffer_all_f32(weight_buf)?;
    if weights.len() != hidden_dim {
        anyhow::bail!(
            "norm weight length {} did not match hidden size {}",
            weights.len(),
            hidden_dim
        );
    }
    let inv_rms = 1.0f32 / (mean_square(hidden_row) + eps).sqrt();
    Ok(hidden_row
        .iter()
        .zip(weights.iter())
        .map(|(hidden, weight)| hidden * inv_rms * (weight + 1.0))
        .collect())
}

fn compute_qwen_final_norm_from_hidden_row(
    engine: &DecodeEngine,
    hidden_row: &[f32],
) -> Result<Vec<f32>> {
    let hidden_dim = engine.weights().config.hidden_size;
    if hidden_row.len() != hidden_dim {
        anyhow::bail!(
            "hidden row length {} did not match hidden size {}",
            hidden_row.len(),
            hidden_dim
        );
    }
    compute_qwen_rms_norm_from_hidden_row(
        hidden_row,
        &engine.weights().norm_weight,
        engine.weights().config.rms_norm_eps as f32,
    )
}

fn compute_qwen_logits_from_hidden_row(
    engine: &DecodeEngine,
    hidden_row: &[f32],
) -> Result<Vec<f32>> {
    let hidden_dim = engine.weights().config.hidden_size;
    if hidden_row.len() != hidden_dim {
        anyhow::bail!(
            "hidden row length {} did not match hidden size {}",
            hidden_row.len(),
            hidden_dim
        );
    }
    let hidden_bf16 = encode_bf16_le(hidden_row);
    let hidden_gpu = GpuBuffer::from_host_bytes(
        engine.ordinal(),
        ScalarType::BF16,
        &[1, hidden_dim],
        &hidden_bf16,
    )
    .map_err(|e| anyhow::anyhow!("trace hidden row upload: {e}"))?;
    kernel_ffi::qwen_rms_norm_standalone_matvec_host_f32(
        engine.ordinal(),
        ScalarType::BF16,
        &hidden_gpu,
        &engine.weights().norm_weight,
        engine.weights().config.rms_norm_eps as f32,
        &*engine.weights().lm_head,
        hidden_dim,
        engine.weights().config.vocab_size,
    )
    .map_err(|e| anyhow::anyhow!("trace hidden row logits: {e}"))
}

fn format_top_logits(entries: &[(usize, f32)]) -> String {
    entries
        .iter()
        .map(|(token, logit)| format!("{token}:{logit:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn logit_row_top_delta_dims(
    native_normed: &[f32],
    oracle_normed: &[f32],
    lm_head: &GpuBuffer,
    ordinal: usize,
    row_idx: usize,
    top_k: usize,
) -> Result<Vec<(usize, f32, f32, f32, f32, f32)>> {
    let hidden_dim = native_normed.len();
    let row = read_buffer_f32_range(lm_head, ordinal, row_idx * hidden_dim, hidden_dim)?;
    let mut dims = Vec::with_capacity(hidden_dim);
    for dim in 0..hidden_dim {
        let weight = row[dim];
        let native_contrib = native_normed[dim] * weight;
        let oracle_contrib = oracle_normed[dim] * weight;
        dims.push((
            dim,
            weight,
            native_normed[dim],
            oracle_normed[dim],
            native_contrib,
            oracle_contrib,
        ));
    }
    dims.sort_by(|a, b| {
        let ad = (a.4 - a.5).abs();
        let bd = (b.4 - b.5).abs();
        bd.total_cmp(&ad).then_with(|| a.0.cmp(&b.0))
    });
    dims.truncate(top_k.min(dims.len()));
    Ok(dims)
}

fn aggregate_logit_row_delta_dims(
    native_normed: &[f32],
    oracle_normed: &[f32],
    lm_head: &GpuBuffer,
    ordinal: usize,
    row_indices: &[usize],
    top_k: usize,
) -> Result<Vec<(usize, f32, f32, f32, f32, f32)>> {
    if native_normed.len() != oracle_normed.len() {
        anyhow::bail!(
            "aggregate_logit_row_delta_dims len mismatch: native={} oracle={}",
            native_normed.len(),
            oracle_normed.len()
        );
    }
    let hidden_dim = native_normed.len();
    let hidden_delta = native_normed
        .iter()
        .copied()
        .zip(oracle_normed.iter().copied())
        .map(|(native_v, oracle_v)| native_v - oracle_v)
        .collect::<Vec<_>>();
    let mut sum_abs_contrib = vec![0.0f32; hidden_dim];
    let mut max_abs_contrib = vec![0.0f32; hidden_dim];
    for &row_idx in row_indices {
        let row = read_buffer_f32_range(lm_head, ordinal, row_idx * hidden_dim, hidden_dim)?;
        for dim in 0..hidden_dim {
            let abs_delta = (hidden_delta[dim] * row[dim]).abs();
            sum_abs_contrib[dim] += abs_delta;
            if abs_delta > max_abs_contrib[dim] {
                max_abs_contrib[dim] = abs_delta;
            }
        }
    }
    let mut dims = (0..hidden_dim)
        .map(|dim| {
            (
                dim,
                native_normed[dim],
                oracle_normed[dim],
                hidden_delta[dim],
                sum_abs_contrib[dim],
                max_abs_contrib[dim],
            )
        })
        .collect::<Vec<_>>();
    dims.sort_by(|a, b| {
        b.4.total_cmp(&a.4)
            .then_with(|| b.5.total_cmp(&a.5))
            .then_with(|| a.0.cmp(&b.0))
    });
    dims.truncate(top_k.min(dims.len()));
    Ok(dims)
}

fn format_logit_row_dims(entries: &[(usize, f32, f32, f32, f32, f32)]) -> String {
    entries
        .iter()
        .map(|(dim, weight, native_hidden, oracle_hidden, native_contrib, oracle_contrib)| {
            format!(
                "{dim}:w={weight:.4} n={native_hidden:.4} o={oracle_hidden:.4} dc={:.4} nc={native_contrib:.4} oc={oracle_contrib:.4}",
                native_contrib - oracle_contrib,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_aggregate_logit_dims(entries: &[(usize, f32, f32, f32, f32, f32)]) -> String {
    entries
        .iter()
        .map(
            |(dim, native_hidden, oracle_hidden, hidden_delta, sum_abs_contrib, max_abs_contrib)| {
                format!(
                    "{dim}:n={native_hidden:.4} o={oracle_hidden:.4} hd={hidden_delta:.4} sum={sum_abs_contrib:.4} max={max_abs_contrib:.4}"
                )
            },
        )
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_tracked_stage_dim_deltas(
    dims: &[usize],
    native_attn: &[f32],
    oracle_attn: &[f32],
    native_post_norm: &[f32],
    oracle_post_norm: &[f32],
    native_mlp_out: &[f32],
    oracle_mlp_out: &[f32],
    native_layer: &[f32],
    oracle_layer: &[f32],
) -> String {
    dims.iter()
        .copied()
        .filter(|dim| {
            *dim < native_attn.len()
                && *dim < oracle_attn.len()
                && *dim < native_post_norm.len()
                && *dim < oracle_post_norm.len()
                && *dim < native_mlp_out.len()
                && *dim < oracle_mlp_out.len()
                && *dim < native_layer.len()
                && *dim < oracle_layer.len()
        })
        .map(|dim| {
            let attn_delta = native_attn[dim] - oracle_attn[dim];
            let post_norm_delta = native_post_norm[dim] - oracle_post_norm[dim];
            let mlp_out_delta = native_mlp_out[dim] - oracle_mlp_out[dim];
            let layer_delta = native_layer[dim] - oracle_layer[dim];
            format!(
                "{dim}:a={attn_delta:+.4} pn={post_norm_delta:+.4} m={mlp_out_delta:+.4} l={layer_delta:+.4}"
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_mlp_down_proj_contributors(
    entries: &[(usize, f32, f32, f32, f32, f32)],
    native_gate: &[f32],
    oracle_gate: &[f32],
    native_up: &[f32],
    oracle_up: &[f32],
    native_swiglu: &[f32],
    oracle_swiglu: &[f32],
) -> String {
    entries
        .iter()
        .map(
            |(idx, coeff, _native_swiglu_row, _oracle_swiglu_row, native_contrib, oracle_contrib)| {
                let gate_native = native_gate.get(*idx).copied().unwrap_or(0.0);
                let gate_oracle = oracle_gate.get(*idx).copied().unwrap_or(0.0);
                let up_native = native_up.get(*idx).copied().unwrap_or(0.0);
                let up_oracle = oracle_up.get(*idx).copied().unwrap_or(0.0);
                let swiglu_native = native_swiglu.get(*idx).copied().unwrap_or(0.0);
                let swiglu_oracle = oracle_swiglu.get(*idx).copied().unwrap_or(0.0);
                format!(
                    "{idx}:w={coeff:.4} g=({gate_native:.4},{gate_oracle:.4}) u=({up_native:.4},{up_oracle:.4}) s=({swiglu_native:.4},{swiglu_oracle:.4}) dc={:.4} nc={native_contrib:.4} oc={oracle_contrib:.4}",
                    native_contrib - oracle_contrib,
                )
            },
        )
        .collect::<Vec<_>>()
        .join(", ")
}

fn aggregate_down_proj_contributors(
    native_swiglu: &[f32],
    oracle_swiglu: &[f32],
    down_proj_w: &GpuBuffer,
    ordinal: usize,
    hidden_dims: &[usize],
    row_width: usize,
    top_k: usize,
) -> Result<Vec<(usize, f32, f32, f32, f32, f32)>> {
    let len = native_swiglu.len().min(oracle_swiglu.len()).min(row_width);
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut sum_abs_contrib = vec![0.0f32; len];
    let mut max_abs_contrib = vec![0.0f32; len];
    for &hidden_dim in hidden_dims {
        let row = read_buffer_f32_range(down_proj_w, ordinal, hidden_dim * row_width, row_width)?;
        for idx in 0..len {
            let native_contrib = native_swiglu[idx] * row[idx];
            let oracle_contrib = oracle_swiglu[idx] * row[idx];
            let abs_delta = (native_contrib - oracle_contrib).abs();
            sum_abs_contrib[idx] += abs_delta;
            if abs_delta > max_abs_contrib[idx] {
                max_abs_contrib[idx] = abs_delta;
            }
        }
    }
    let mut entries = (0..len)
        .map(|idx| {
            (
                idx,
                native_swiglu[idx],
                oracle_swiglu[idx],
                native_swiglu[idx] - oracle_swiglu[idx],
                sum_abs_contrib[idx],
                max_abs_contrib[idx],
            )
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| {
        b.4.total_cmp(&a.4)
            .then_with(|| b.5.total_cmp(&a.5))
            .then_with(|| a.0.cmp(&b.0))
    });
    entries.truncate(top_k.min(entries.len()));
    Ok(entries)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn format_mlp_intermediate_aggregate(
    entries: &[(usize, f32, f32, f32, f32, f32)],
    native_gate: &[f32],
    oracle_gate: &[f32],
    native_up: &[f32],
    oracle_up: &[f32],
) -> String {
    entries
        .iter()
        .map(|(idx, native_swiglu, oracle_swiglu, swiglu_delta, sum_abs, max_abs)| {
            let gate_native = native_gate.get(*idx).copied().unwrap_or(0.0);
            let gate_oracle = oracle_gate.get(*idx).copied().unwrap_or(0.0);
            let up_native = native_up.get(*idx).copied().unwrap_or(0.0);
            let up_oracle = oracle_up.get(*idx).copied().unwrap_or(0.0);
            let silu_native = silu(gate_native);
            let silu_oracle = silu(gate_oracle);
            let gate_only = silu(gate_native) * up_oracle;
            let up_only = silu(gate_oracle) * up_native;
            format!(
                "{idx}:g=({gate_native:.4},{gate_oracle:.4}) gd={:+.4} sg=({silu_native:.4},{silu_oracle:.4}) sgd={:+.4} u=({up_native:.4},{up_oracle:.4}) s=({native_swiglu:.4},{oracle_swiglu:.4}) sd={swiglu_delta:+.4} gate_only={:+.4} up_only={:+.4} sum={sum_abs:.4} max={max_abs:.4}",
                gate_native - gate_oracle,
                silu_native - silu_oracle,
                gate_only - oracle_swiglu,
                up_only - oracle_swiglu,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn swiglu_selfcheck_delta(gate: &[f32], up: &[f32], swiglu: &[f32]) -> f32 {
    gate.iter()
        .zip(up.iter())
        .zip(swiglu.iter())
        .map(|((gate, up), swiglu)| (silu(*gate) * *up - *swiglu).abs())
        .fold(0.0f32, f32::max)
}

fn matmul_row_selfcheck_details(
    input: &[f32],
    weight: &GpuBuffer,
    ordinal: usize,
    row_idx: usize,
    row_width: usize,
    output: &[f32],
) -> Result<(f32, f32, f32)> {
    let row = read_buffer_f32_range(weight, ordinal, row_idx * row_width, row_width)?;
    let len = input.len().min(row.len());
    anyhow::ensure!(
        row_idx < output.len(),
        "matmul selfcheck row {row_idx} out of range for output len {}",
        output.len()
    );
    let reference = input
        .iter()
        .zip(row.iter())
        .take(len)
        .map(|(input, coeff)| input * coeff)
        .sum::<f32>();
    let native = output[row_idx];
    Ok((native, reference, (native - reference).abs()))
}

fn weight_row_top_delta_dims(
    native_input: &[f32],
    oracle_input: &[f32],
    weight: &GpuBuffer,
    ordinal: usize,
    row_idx: usize,
    row_width: usize,
    top_k: usize,
) -> Result<Vec<(usize, f32, f32, f32, f32, f32)>> {
    if native_input.len() != row_width || oracle_input.len() != row_width {
        anyhow::bail!(
            "weight_row_top_delta_dims input len mismatch: native={} oracle={} row_width={}",
            native_input.len(),
            oracle_input.len(),
            row_width
        );
    }
    let row = read_buffer_f32_range(weight, ordinal, row_idx * row_width, row_width)?;
    let mut dims = Vec::with_capacity(row_width);
    for dim in 0..row_width {
        let coeff = row[dim];
        let native_contrib = native_input[dim] * coeff;
        let oracle_contrib = oracle_input[dim] * coeff;
        dims.push((
            dim,
            coeff,
            native_input[dim],
            oracle_input[dim],
            native_contrib,
            oracle_contrib,
        ));
    }
    dims.sort_by(|a, b| {
        let ad = (a.4 - a.5).abs();
        let bd = (b.4 - b.5).abs();
        bd.total_cmp(&ad).then_with(|| a.0.cmp(&b.0))
    });
    dims.truncate(top_k.min(dims.len()));
    Ok(dims)
}

fn format_weight_row_top_delta_dims(entries: &[(usize, f32, f32, f32, f32, f32)]) -> String {
    entries
        .iter()
        .map(|(idx, coeff, native_input, oracle_input, native_contrib, oracle_contrib)| {
            format!(
                "{idx}:w={coeff:.4} n={native_input:.4} o={oracle_input:.4} dc={:.4} nc={native_contrib:.4} oc={oracle_contrib:.4}",
                native_contrib - oracle_contrib,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn decode_bf16_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn decode_f32_bytes_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_gpu_buffer_f32(buf: &GpuBuffer) -> Result<Vec<f32>> {
    let bytes = buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("buffer D2H: {e}"))?;
    match buf.dtype() {
        ScalarType::BF16 => Ok(decode_bf16_le(&bytes)),
        ScalarType::F32 => Ok(decode_f32_bytes_le(&bytes)),
        other => anyhow::bail!("unsupported buffer dtype for debug decode: {other:?}"),
    }
}

fn parse_trace_position_scan(spec: &str, seq_len: usize) -> Result<Vec<usize>> {
    let trimmed = spec.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok((0..seq_len).collect());
    }
    let mut positions = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let position = part
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("invalid trace position '{part}': {e}"))?;
        if position >= seq_len {
            anyhow::bail!("trace position {position} out of range for seq_len {seq_len}");
        }
        if !positions.contains(&position) {
            positions.push(position);
        }
    }
    if positions.is_empty() {
        anyhow::bail!("trace position scan produced no positions");
    }
    Ok(positions)
}

fn apply_rope_reference_for_position(
    prepared: &[f32],
    rotary: &qwen35::rotary::RotaryTables,
    position: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let rotary_dim = rotary.rotary_dim;
    let half_rot = rotary_dim / 2;
    let cos = decode_rotary_row(&rotary.cos, position, half_rot)?;
    let sin = decode_rotary_row(&rotary.sin, position, half_rot)?;
    let mut out = prepared.to_vec();
    for head in 0..num_heads {
        let base = head * head_dim;
        for i in 0..half_rot {
            let x0 = prepared[base + i];
            let x1 = prepared[base + i + half_rot];
            out[base + i] = x0 * cos[i] - x1 * sin[i];
            out[base + i + half_rot] = x1 * cos[i] + x0 * sin[i];
        }
    }
    Ok(out)
}

fn decode_rotary_row(table: &GpuBuffer, row: usize, row_elems: usize) -> Result<Vec<f32>> {
    let all = table
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("rotary table D2H: {e}"))?;
    let values = decode_bf16_le(&all);
    let start = row
        .checked_mul(row_elems)
        .ok_or_else(|| anyhow::anyhow!("rotary row overflow"))?;
    let end = start
        .checked_add(row_elems)
        .ok_or_else(|| anyhow::anyhow!("rotary row overflow"))?;
    if end > values.len() {
        anyhow::bail!(
            "rotary row {} out of range for {} values with row_elems {}",
            row,
            values.len(),
            row_elems
        );
    }
    Ok(values[start..end].to_vec())
}

struct RmsNormHeadDetail {
    head: usize,
    dim: usize,
    hidden_native: f32,
    hidden_oracle: f32,
    inv_rms_native: f32,
    inv_rms_oracle: f32,
    scale: f32,
}

fn rms_norm_head_detail(
    native_input: &[f32],
    oracle_input: &[f32],
    weight: &[f32],
    head_dim: usize,
    flat_idx: usize,
) -> Result<RmsNormHeadDetail> {
    if native_input.len() != oracle_input.len() {
        anyhow::bail!(
            "rms_norm_head_detail length mismatch: {} vs {}",
            native_input.len(),
            oracle_input.len()
        );
    }
    if weight.len() != head_dim {
        anyhow::bail!(
            "rms_norm_head_detail weight len mismatch: {} vs head_dim {}",
            weight.len(),
            head_dim
        );
    }
    if flat_idx >= native_input.len() {
        anyhow::bail!(
            "rms_norm_head_detail idx {} out of range for {} values",
            flat_idx,
            native_input.len()
        );
    }
    let head = flat_idx / head_dim;
    let dim = flat_idx % head_dim;
    let row_start = head * head_dim;
    let row_end = row_start + head_dim;
    let native_row = &native_input[row_start..row_end];
    let oracle_row = &oracle_input[row_start..row_end];
    let native_ms = native_row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
    let oracle_ms = oracle_row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
    Ok(RmsNormHeadDetail {
        head,
        dim,
        hidden_native: native_input[flat_idx],
        hidden_oracle: oracle_input[flat_idx],
        inv_rms_native: 1.0 / (native_ms + 1e-6).sqrt(),
        inv_rms_oracle: 1.0 / (oracle_ms + 1e-6).sqrt(),
        scale: weight[dim] + 1.0,
    })
}

fn apply_rms_norm_reference(
    input: &[f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if weight.len() != head_dim {
        anyhow::bail!(
            "apply_rms_norm_reference weight len mismatch: {} vs head_dim {}",
            weight.len(),
            head_dim
        );
    }
    if input.len() != num_heads * head_dim {
        anyhow::bail!(
            "apply_rms_norm_reference input len mismatch: {} vs {}",
            input.len(),
            num_heads * head_dim
        );
    }
    let mut out = vec![0.0f32; input.len()];
    for head in 0..num_heads {
        let row_start = head * head_dim;
        let row_end = row_start + head_dim;
        let row = &input[row_start..row_end];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv_rms = 1.0 / (ms + eps).sqrt();
        for dim in 0..head_dim {
            out[row_start + dim] = row[dim] * inv_rms * (weight[dim] + 1.0);
        }
    }
    Ok(out)
}

fn apply_rms_norm_gated_reference(
    hidden: &[f32],
    gate: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if hidden.len() != gate.len() {
        anyhow::bail!(
            "apply_rms_norm_gated_reference len mismatch: hidden={} gate={}",
            hidden.len(),
            gate.len()
        );
    }
    if weight.is_empty() {
        anyhow::bail!("apply_rms_norm_gated_reference weight must not be empty");
    }
    if hidden.len() % weight.len() != 0 {
        anyhow::bail!(
            "apply_rms_norm_gated_reference weight len mismatch: hidden={} weight={}",
            hidden.len(),
            weight.len()
        );
    }
    let row_cols = weight.len();
    let n_rows = hidden.len() / row_cols;
    let mut out = Vec::with_capacity(hidden.len());
    for row in 0..n_rows {
        let base = row * row_cols;
        let row_hidden = &hidden[base..base + row_cols];
        let row_gate = &gate[base..base + row_cols];
        let mean_sq = row_hidden.iter().map(|v| v * v).sum::<f32>() / row_cols as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        out.extend(
            row_hidden
                .iter()
                .zip(row_gate.iter())
                .zip(weight.iter())
                .map(|((hidden, gate), weight)| hidden * inv_rms * weight * silu(*gate)),
        );
    }
    Ok(out)
}

fn apply_matmul_rhs_transposed_reference(
    lhs_row: &[f32],
    rhs: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    if lhs_row.len() != in_dim {
        anyhow::bail!(
            "apply_matmul_rhs_transposed_reference lhs len mismatch: {} vs {}",
            lhs_row.len(),
            in_dim
        );
    }
    if rhs.len() != out_dim * in_dim {
        anyhow::bail!(
            "apply_matmul_rhs_transposed_reference rhs len mismatch: {} vs {}",
            rhs.len(),
            out_dim * in_dim
        );
    }
    let mut out = vec![0.0f32; out_dim];
    for out_idx in 0..out_dim {
        let row = &rhs[out_idx * in_dim..(out_idx + 1) * in_dim];
        out[out_idx] = lhs_row
            .iter()
            .zip(row.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
    }
    Ok(out)
}

fn silu_reference(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn apply_linear_conv_pack_row_reference(
    conv_window: &[f32],
    weights: &[f32],
    conv_dim: usize,
    kernel_size: usize,
) -> Result<Vec<f32>> {
    if conv_window.len() != conv_dim * kernel_size {
        anyhow::bail!(
            "apply_linear_conv_pack_row_reference window len mismatch: {} vs {}",
            conv_window.len(),
            conv_dim * kernel_size
        );
    }
    if weights.len() != conv_dim * kernel_size {
        anyhow::bail!(
            "apply_linear_conv_pack_row_reference weight len mismatch: {} vs {}",
            weights.len(),
            conv_dim * kernel_size
        );
    }
    let mut out = vec![0.0f32; conv_dim];
    for ch in 0..conv_dim {
        let window_row = &conv_window[ch * kernel_size..(ch + 1) * kernel_size];
        let weight_row = &weights[ch * kernel_size..(ch + 1) * kernel_size];
        let acc = window_row
            .iter()
            .zip(weight_row.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        out[ch] = silu_reference(acc);
    }
    Ok(out)
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

fn flatten_bsh(value: &serde_json::Value) -> Option<Vec<f32>> {
    value.as_array()?;
    let mut out = Vec::new();
    flatten_json_numbers(value, &mut out);
    Some(out)
}

fn encode_bf16_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

fn flatten_last_token_bsd(value: &serde_json::Value) -> Option<Vec<f32>> {
    flatten_token_bsd(value, None)
}

fn flatten_json_vector(value: &serde_json::Value) -> Option<Vec<f32>> {
    let array = value.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for elem in array {
        out.push(elem.as_f64()? as f32);
    }
    Some(out)
}

fn flatten_token_bsd(value: &serde_json::Value, position: Option<usize>) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let token = match position {
        Some(position) => batch.get(position)?,
        None => batch.last()?,
    };
    let mut out = Vec::new();
    flatten_json_numbers(token, &mut out);
    Some(out)
}

fn extract_causal_conv_window_bsd(
    value: &serde_json::Value,
    position: usize,
    dim: usize,
    kernel_size: usize,
) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    if position >= batch.len() {
        return None;
    }
    let pad = kernel_size.saturating_sub(1);
    let mut out = vec![0.0f32; dim * kernel_size];
    for tap in 0..kernel_size {
        let src_pos = position as isize - pad as isize + tap as isize;
        if src_pos < 0 {
            continue;
        }
        let row = flatten_json_vector(batch.get(src_pos as usize)?)?;
        if row.len() != dim {
            return None;
        }
        let dst_base = tap;
        for ch in 0..dim {
            out[ch * kernel_size + dst_base] = row[ch];
        }
    }
    Some(out)
}

fn flatten_last_token_bhsd(value: &serde_json::Value) -> Option<Vec<f32>> {
    flatten_token_bhsd(value, None)
}

fn flatten_token_bhsd(value: &serde_json::Value, position: Option<usize>) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let mut out = Vec::new();
    for head in batch {
        let tokens = head.as_array()?;
        let token = match position {
            Some(position) => tokens.get(position)?,
            None => tokens.last()?,
        };
        flatten_json_numbers(token, &mut out);
    }
    Some(out)
}

fn flatten_last_token_bshd(value: &serde_json::Value) -> Option<Vec<f32>> {
    flatten_token_bshd(value, None)
}

fn flatten_token_bshd(value: &serde_json::Value, position: Option<usize>) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let token = match position {
        Some(position) => batch.get(position)?,
        None => batch.last()?,
    }
    .as_array()?;
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
    if sign != 0 {
        -val
    } else {
        val
    }
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
        None,
        None,
    )?;
    if trace_layer == 0 {
        let token_id = *token_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("component input trace has empty token_ids"))?;
        let hidden_dim = engine.weights().config.hidden_size;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let row_bytes = hidden_dim * elem_bytes;
        let embedding = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("component input trace embedding alloc: {e}"))?;
        gpu_hal::copy_d2d(
            ordinal,
            embedding.as_ptr() as *mut c_void,
            engine
                .weights()
                .embed_tokens
                .offset_ptr(token_id as usize * row_bytes),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("component input trace embedding copy: {e}"))?;
        let embedding_bytes = embedding
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("component input trace embedding D2H: {e}"))?;
        let native_f32 = decode_bf16_le(native_hidden);
        let replay_f32 = decode_bf16_le(&embedding_bytes);
        let delta = validate::max_abs_delta(&native_f32, &replay_f32);
        eprintln!(
            "[trace-component-input] layer=0 token={token_id} embedding_delta={delta:.6}"
        );
        return Ok(());
    }
    let replay_hidden = {
        replay
            .layer_hidden_trace
            .as_ref()
            .and_then(|layers| layers.get(trace_layer - 1))
    };
    if let Some(replay_hidden) = replay_hidden {
        let native_f32 = decode_bf16_le(native_hidden);
        let replay_f32 = decode_bf16_le(replay_hidden);
        let delta = validate::max_abs_delta(&native_f32, &replay_f32);
        eprintln!("[trace-component-input] layer={trace_layer} hidden_delta={delta:.6}");
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
        None,
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
        eprintln!("[trace-persistent-input] layer={trace_layer} hidden_delta={delta:.6}");
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
        None,
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

    let (conv_delta, first_conv_mismatch) =
        match (&native_layer.conv_state, &replay_layer.conv_state) {
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
    let (rec_delta, first_rec_mismatch, max_rec_mismatch) =
        match (&native_layer.recurrent_state, &replay_layer.recurrent_state) {
            (Some(native), Some(replay)) => {
                let native_vals =
                    decode_f32_le(&native.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("native persistent recurrent trace D2H: {e}")
                    })?);
                let replay_vals =
                    decode_f32_le(&replay.to_host_bytes().map_err(|e| {
                        anyhow::anyhow!("replay persistent recurrent trace D2H: {e}")
                    })?);
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
    anyhow::ensure!(
        trace_layer > 0,
        "trace layer must be > 0 for full-attention input tracing"
    );

    let prefix_ids = token_ids
        .get(..token_ids.len().saturating_sub(1))
        .ok_or_else(|| {
            anyhow::anyhow!("missing prefix token ids for persistent full-attn trace")
        })?;
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
    let native_scores =
        engine.trace_persistent_full_attention_scores_after_layers(0, seqlen_offset + 1)?;
    let (_, _, _, native_token_mixer) =
        engine.trace_persistent_mlp_stage_after_layers(0, text_config.intermediate_size)?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let native_component = engine.trace_full_attention_stages_from_hidden(
        trace_layer,
        &native_hidden,
        seqlen_offset,
    )?;
    let native_component_layer = engine
        .trace_full_attention_layer_output_from_hidden_current_state(
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
        None,
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
        None,
        None,
    )?;
    let replay_hidden = replay
        .layer_hidden_trace
        .as_ref()
        .and_then(|layers| layers.get(trace_layer - 1))
        .ok_or_else(|| anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}"))?;
    let replay_component = engine.trace_full_attention_stages_from_hidden(
        trace_layer,
        replay_hidden,
        seqlen_offset,
    )?;
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
        let bf16_gated =
            decode_f32_le(&engine.trace_persistent_full_attention_gated_after_layers(0)?);
        let bf16_pre_gate =
            decode_f32_le(&engine.trace_persistent_full_attention_pre_gate_after_layers(0)?);
        let bf16_scores = decode_f32_le(
            &engine.trace_persistent_full_attention_scores_after_layers(0, seqlen_offset + 1)?,
        );
        let (_, _, _, bf16_token_mixer) =
            engine.trace_persistent_mlp_stage_after_layers(0, text_config.intermediate_size)?;
        let bf16_token_mixer_f32 = decode_f32_le(&bf16_token_mixer);
        kv_vs_bf16_pre_gate = Some(validate::max_abs_delta(
            &native_pre_gate_f32,
            &bf16_pre_gate,
        ));
        kv_vs_bf16_gated = Some(validate::max_abs_delta(&native_gated_f32, &bf16_gated));
        kv_vs_bf16_attn_hidden = Some(validate::max_abs_delta(
            &native_token_mixer_f32,
            &bf16_token_mixer_f32,
        ));
        kv_vs_bf16_scores = Some(validate::max_abs_delta(&native_scores_f32, &bf16_scores));
        kv_vs_bf16_hidden = Some(validate::max_abs_delta(&native_hidden_f32, &bf16_hidden));
        kv_vs_bf16_q = Some(validate::max_abs_delta(&native_q_f32, &bf16_q));
        kv_vs_bf16_saved_gate = Some(validate::max_abs_delta(
            &native_saved_gate_f32,
            &bf16_saved_gate,
        ));
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
                    validate::max_abs_delta(
                        &native_scores_f32[start..end],
                        &bf16_scores[start..end],
                    )
                })
                .collect::<Vec<_>>(),
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
            validate::max_abs_delta(
                &native_pre_gate_f32[start..end],
                &native_comp_pre_gate_f32[start..end],
            )
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
                    let k_val =
                        half::bf16::from_f32(fp8_e4m3_to_f32_host(k_bytes[base + d]) * scale_val)
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
        let mut tmp_k_fp8 =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::U8, &[nkv, max_t, hd])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp K cache alloc: {e}"))?;
        let mut tmp_v_fp8 =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::U8, &[nkv, max_t, hd])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp V cache alloc: {e}"))?;
        let mut tmp_k_scale =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::F32, &[nkv, max_t])
                .map_err(|e| anyhow::anyhow!("trace fp8 temp K scale alloc: {e}"))?;
        let mut tmp_v_scale =
            gpu_hal::GpuBuffer::zeros(ordinal, gpu_hal::ScalarType::F32, &[nkv, max_t])
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
            let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
                        fp8_e4m3_to_f32_host(native_v_cache_bytes[base]) * scale_val,
                    )
                    .to_f32();
                    acc += w * v_val;
                }
                host_pre_gate[qh * hd + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let native_vs_host_pre_gate = validate::max_abs_delta(&native_pre_gate_f32, &host_pre_gate);
        let per_head_host_pre_gate = (0..num_q_heads)
            .map(|h| {
                let start = h * hd;
                let end = start + hd;
                validate::max_abs_delta(
                    &native_pre_gate_f32[start..end],
                    &host_pre_gate[start..end],
                )
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
            .map(|vals| {
                vals.iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
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
        None,
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
                .ok_or_else(|| {
                    anyhow::anyhow!("missing replay hidden trace for layer {trace_layer}")
                })?,
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
        .ok_or_else(|| {
            anyhow::anyhow!("missing replay output hidden trace for layer {trace_layer}")
        })?;
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
    let native_hidden_out = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer + 1,
        0,
    )?;
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
            validate::max_abs_delta(
                &decode_bf16_le(&native_hidden),
                &decode_bf16_le(replay_hidden),
            )
        })
        .unwrap_or(0.0);
    let replay_comp_trace = if let Some(replay_hidden) = replay_hidden {
        engine.rebuild_prefill_state(prefix_ids, true)?;
        engine.set_hidden_from_bytes(replay_hidden)?;
        Some(engine.component_trace_linear_layer_from_current_hidden(trace_layer)?)
    } else {
        None
    };
    let comp_vs_replay_conv = validate::max_abs_delta(
        &decode_bf16_le(&native_comp_conv),
        &decode_bf16_le(&replay_conv),
    );
    let comp_vs_replay_recurrent = validate::max_abs_delta(
        &decode_f32_le(&native_comp_recurrent),
        &decode_f32_le(&replay_recurrent),
    );
    let comp_vs_replay_hidden = validate::max_abs_delta(
        &decode_bf16_le(&native_comp_hidden),
        &decode_bf16_le(replay_hidden_out),
    );
    let native_vs_comp_conv = validate::max_abs_delta(
        &decode_bf16_le(&native_conv),
        &decode_bf16_le(&native_comp_conv),
    );
    let native_vs_comp_recurrent = validate::max_abs_delta(
        &decode_f32_le(&native_recurrent),
        &decode_f32_le(&native_comp_recurrent),
    );
    let native_vs_comp_proj_residual = validate::max_abs_delta(
        &decode_bf16_le(&native_hidden_out),
        &bf16_residual_sum(&native_hidden, &native_comp_trace.proj_out),
    );
    let native_vs_comp_qkv_proj = validate::max_abs_delta(
        &decode_f32_le(&native_qkv_proj),
        &decode_bf16_le(&native_comp_trace.qkv),
    );
    let native_vs_comp_z_proj = validate::max_abs_delta(
        &decode_f32_le(&native_z_proj),
        &decode_bf16_le(&native_comp_trace.z),
    );
    let native_vs_comp_b_proj = validate::max_abs_delta(
        &decode_f32_le(&native_b_proj),
        &decode_bf16_le(&native_comp_trace.b),
    );
    let native_vs_comp_a_proj = validate::max_abs_delta(
        &decode_f32_le(&native_a_proj),
        &decode_bf16_le(&native_comp_trace.a),
    );
    let native_vs_comp_post_norm = validate::max_abs_delta(
        &decode_f32_le(&native_post_norm),
        &decode_bf16_le(&native_comp_layer.post_attn_norm),
    );
    let native_vs_comp_gated = validate::max_abs_delta(
        &decode_f32_le(&native_gated),
        &decode_bf16_le(&native_comp_trace.gated),
    );
    let native_vs_comp_swiglu = validate::max_abs_delta(
        &decode_f32_le(&native_swiglu),
        &decode_bf16_le(&native_comp_layer.mlp_swiglu),
    );
    let native_vs_comp_token_mixer = validate::max_abs_delta(
        &decode_f32_le(&native_token_mixer),
        &decode_bf16_le(&native_comp_layer.attn_hidden),
    );
    let native_vs_comp_mlp_down = validate::max_abs_delta(
        &decode_f32_le(&native_mlp_down),
        &decode_bf16_le(&native_comp_layer.mlp_out),
    );
    let native_vs_replay_post_norm = validate::max_abs_delta(
        &decode_f32_le(&native_post_norm),
        &decode_bf16_le(replay_post),
    );
    let native_vs_replay_gated = replay_comp_trace
        .as_ref()
        .map(|trace| {
            validate::max_abs_delta(
                &decode_f32_le(&native_gated),
                &decode_bf16_le(&trace.0.gated),
            )
        })
        .unwrap_or(0.0);
    let native_vs_replay_swiglu = validate::max_abs_delta(
        &decode_f32_le(&native_swiglu),
        &decode_bf16_le(replay_swiglu),
    );
    let native_vs_replay_token_mixer = validate::max_abs_delta(
        &decode_f32_le(&native_token_mixer),
        &decode_bf16_le(replay_attn),
    );
    let native_vs_replay_mlp_down = validate::max_abs_delta(
        &decode_f32_le(&native_mlp_down),
        &decode_bf16_le(replay_mlp_out),
    );
    let native_vs_replay_qkv_proj = replay_comp_trace
        .as_ref()
        .map(|trace| {
            validate::max_abs_delta(
                &decode_f32_le(&native_qkv_proj),
                &decode_bf16_le(&trace.0.qkv),
            )
        })
        .unwrap_or(0.0);
    let native_vs_replay_z_proj = replay_comp_trace
        .as_ref()
        .map(|trace| {
            validate::max_abs_delta(&decode_f32_le(&native_z_proj), &decode_bf16_le(&trace.0.z))
        })
        .unwrap_or(0.0);
    let native_vs_replay_b_proj = replay_comp_trace
        .as_ref()
        .map(|trace| {
            validate::max_abs_delta(&decode_f32_le(&native_b_proj), &decode_bf16_le(&trace.0.b))
        })
        .unwrap_or(0.0);
    let native_vs_replay_a_proj = replay_comp_trace
        .as_ref()
        .map(|trace| {
            validate::max_abs_delta(&decode_f32_le(&native_a_proj), &decode_bf16_le(&trace.0.a))
        })
        .unwrap_or(0.0);
    let comp_layer_vs_replay_hidden = validate::max_abs_delta(
        &decode_bf16_le(&native_comp_layer.layer_hidden),
        &decode_bf16_le(replay_hidden_out),
    );
    let native_vs_comp_layer_hidden = validate::max_abs_delta(
        &decode_bf16_le(&native_hidden_out),
        &decode_bf16_le(&native_comp_layer.layer_hidden),
    );
    let native_vs_replay_hidden = validate::max_abs_delta(
        &decode_bf16_le(&native_hidden_out),
        &decode_bf16_le(replay_hidden_out),
    );
    let native_qkv_proj_f32 = decode_f32_le(&native_qkv_proj);
    let native_z_proj_f32 = decode_f32_le(&native_z_proj);
    let comp_qkv_proj_f32 = decode_bf16_le(&native_comp_trace.qkv);
    let comp_z_proj_f32 = decode_bf16_le(&native_comp_trace.z);
    let sample_qkv_native = native_qkv_proj_f32
        .iter()
        .take(4)
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(",");
    let sample_qkv_comp = comp_qkv_proj_f32
        .iter()
        .take(4)
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(",");
    let sample_z_native = native_z_proj_f32
        .iter()
        .take(4)
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(",");
    let sample_z_comp = comp_z_proj_f32
        .iter()
        .take(4)
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(",");

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
        None,
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
    let attn_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.attn_hidden), &decode_bf16_le(attn));
    let post_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.post_attn_norm),
        &decode_bf16_le(post),
    );
    let mlp_delta = validate::max_abs_delta(&decode_bf16_le(&native.mlp_out), &decode_bf16_le(mlp));
    let hidden_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.layer_hidden),
        &decode_bf16_le(hidden),
    );
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
        None,
        None,
    )?;
    let replay = replay
        .linear_debug_trace
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing replay linear trace for layer {trace_layer}"))?;
    let normed_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.normed), &decode_bf16_le(&replay.normed));
    let qkv_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.qkv), &decode_bf16_le(&replay.qkv));
    let z_delta = validate::max_abs_delta(&decode_bf16_le(&native.z), &decode_bf16_le(&replay.z));
    let native_normed = decode_bf16_le(&native.normed);
    let replay_normed = decode_bf16_le(&replay.normed);
    let native_qkv = decode_bf16_le(&native.qkv);
    let replay_qkv = decode_bf16_le(&replay.qkv);
    let native_z = decode_bf16_le(&native.z);
    let replay_z = decode_bf16_le(&replay.z);
    let (normed_idx, normed_native_v, normed_replay_v, normed_detail_delta) =
        max_abs_delta_details(&native_normed, &replay_normed);
    if trace_layer == 0 {
        let token_id = *token_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("component linear trace has empty token_ids"))?;
        let hidden_dim = engine.weights().config.hidden_size;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let row_bytes = hidden_dim * elem_bytes;
        let embedding = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("component linear trace embedding alloc: {e}"))?;
        gpu_hal::copy_d2d(
            ordinal,
            embedding.as_ptr() as *mut c_void,
            engine
                .weights()
                .embed_tokens
                .offset_ptr(token_id as usize * row_bytes),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("component linear trace embedding copy: {e}"))?;
        let embedding_f32 = decode_bf16_le(
            &embedding
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("component linear trace embedding D2H: {e}"))?,
        );
        let norm_ref = compute_qwen_rms_norm_from_hidden_row(
            &embedding_f32,
            &engine.weights().layers[trace_layer].input_norm_w,
            engine.weights().config.rms_norm_eps as f32,
        )?;
        let (native_norm_idx, native_norm_v, native_norm_ref_v, native_norm_delta) =
            max_abs_delta_details(&native_normed, &norm_ref);
        let (replay_norm_idx, replay_norm_v, replay_norm_ref_v, replay_norm_delta) =
            max_abs_delta_details(&replay_normed, &norm_ref);
        eprintln!(
            "[trace-component-linear-normcheck] layer=0 token={token_id} native_norm_delta={native_norm_delta:.6} native_idx={native_norm_idx} native={native_norm_v:.6} ref={native_norm_ref_v:.6} replay_norm_delta={replay_norm_delta:.6} replay_idx={replay_norm_idx} replay={replay_norm_v:.6} ref={replay_norm_ref_v:.6} native_vs_replay_idx={normed_idx} native_vs_replay_native={normed_native_v:.6} native_vs_replay_replay={normed_replay_v:.6} native_vs_replay_delta={normed_detail_delta:.6}"
        );
    }
    if let Some(linear) = engine.weights().layers[trace_layer].linear.as_ref() {
        let hidden_dim = engine.weights().config.hidden_size;
        let qkv_w = decode_gpu_buffer_f32(&linear.qkv_proj_w)?;
        let z_w = decode_gpu_buffer_f32(&linear.z_proj_w)?;
        let native_qkv_ref = apply_matmul_rhs_transposed_reference(
            &native_normed,
            &qkv_w,
            native_qkv.len(),
            hidden_dim,
        )?;
        let replay_qkv_ref = apply_matmul_rhs_transposed_reference(
            &replay_normed,
            &qkv_w,
            replay_qkv.len(),
            hidden_dim,
        )?;
        let native_z_ref =
            apply_matmul_rhs_transposed_reference(&native_normed, &z_w, native_z.len(), hidden_dim)?;
        let replay_z_ref =
            apply_matmul_rhs_transposed_reference(&replay_normed, &z_w, replay_z.len(), hidden_dim)?;
        let (native_qkv_idx, native_qkv_v, native_qkv_ref_v, native_qkv_self_delta) =
            max_abs_delta_details(&native_qkv, &native_qkv_ref);
        let (replay_qkv_idx, replay_qkv_v, replay_qkv_ref_v, replay_qkv_self_delta) =
            max_abs_delta_details(&replay_qkv, &replay_qkv_ref);
        let (native_z_idx, native_z_v, native_z_ref_v, native_z_self_delta) =
            max_abs_delta_details(&native_z, &native_z_ref);
        let (replay_z_idx, replay_z_v, replay_z_ref_v, replay_z_self_delta) =
            max_abs_delta_details(&replay_z, &replay_z_ref);
        eprintln!(
            "[trace-component-linear-selfcheck] layer={trace_layer} native_qkv_delta={native_qkv_self_delta:.6} native_qkv_idx={native_qkv_idx} native_qkv={native_qkv_v:.6} native_qkv_ref={native_qkv_ref_v:.6} replay_qkv_delta={replay_qkv_self_delta:.6} replay_qkv_idx={replay_qkv_idx} replay_qkv={replay_qkv_v:.6} replay_qkv_ref={replay_qkv_ref_v:.6} native_z_delta={native_z_self_delta:.6} native_z_idx={native_z_idx} native_z={native_z_v:.6} native_z_ref={native_z_ref_v:.6} replay_z_delta={replay_z_self_delta:.6} replay_z_idx={replay_z_idx} replay_z={replay_z_v:.6} replay_z_ref={replay_z_ref_v:.6}"
        );
    }
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
        beta_delta = beta_delta
            .max((packed_native[base + 2 * khd + vhd] - packed_replay[base + 2 * khd + vhd]).abs());
        gexp_delta = gexp_delta.max(
            (packed_native[base + 2 * khd + vhd + 1] - packed_replay[base + 2 * khd + vhd + 1])
                .abs(),
        );
    }
    let rec_apply_delta = validate::max_abs_delta(
        &decode_f32_le(&native.rec_apply),
        &decode_f32_le(&replay.rec_apply),
    );
    let attn_delta =
        validate::max_abs_delta(&decode_bf16_le(&native.attn), &decode_bf16_le(&replay.attn));
    let gated_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.gated),
        &decode_bf16_le(&replay.gated),
    );
    let proj_out_delta = validate::max_abs_delta(
        &decode_bf16_le(&native.proj_out),
        &decode_bf16_le(&replay.proj_out),
    );
    eprintln!(
        "[trace-component-linear] layer={trace_layer} normed_delta={normed_delta:.6} qkv_delta={qkv_delta:.6} z_delta={z_delta:.6} packed_delta={packed_delta:.6} q_delta={q_delta:.6} k_delta={k_delta:.6} v_delta={v_delta:.6} state_vs_tail_delta={state_vs_tail_delta:.6} v_ref_native_delta={v_ref_native_delta:.6} v_ref_replay_delta={v_ref_replay_delta:.6} beta_delta={beta_delta:.6} gexp_delta={gexp_delta:.6} rec_apply_delta={rec_apply_delta:.6} attn_delta={attn_delta:.6} gated_delta={gated_delta:.6} proj_out_delta={proj_out_delta:.6}"
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
        None,
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

    let conv_delta =
        validate::max_abs_delta(&decode_bf16_le(&native_conv), &decode_bf16_le(&replay_conv));
    let rec_delta =
        validate::max_abs_delta(&decode_f32_le(&native_rec), &decode_f32_le(&replay_rec));
    eprintln!(
        "[trace-component-linear-state] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={rec_delta:.6}"
    );
    Ok(())
}

fn trace_metal_component_linear_state_layers(
    engine: &DecodeEngine,
    trace_spec: &str,
    history_token_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
) -> Result<()> {
    let mut layers = Vec::new();
    if trace_spec.eq_ignore_ascii_case("all") {
        layers.extend(
            (0..engine.weights().config.num_hidden_layers)
                .filter(|&layer| !engine.weights().config.is_full_attention(layer)),
        );
    } else {
        for raw in trace_spec.split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            layers.push(
                raw.parse::<usize>()
                    .with_context(|| format!("invalid Metal component linear state layer '{raw}'"))?,
            );
        }
    }

    if layers.is_empty() {
        anyhow::bail!("SUPERSONIC_METAL_TRACE_COMPONENT_LINEAR_STATE_LAYER selected no layers");
    }

    for layer in layers {
        if engine.weights().config.is_full_attention(layer) {
            eprintln!(
                "[trace-component-linear-state] layer={layer} skipped kind=full-attention"
            );
            continue;
        }
        trace_component_linear_state_layer(
            engine,
            layer,
            history_token_ids,
            ordinal,
            kv_chunk_size,
            prefill_chunk_size,
            use_4b_kernel,
        )?;
    }
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
        dtype: if matches!(kv_dtype, gpu_hal::ScalarType::U8) {
            "fp8"
        } else {
            "bf16"
        },
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
                            diff.first_k_mismatch =
                                Some((h, t, d, native_k[native_idx * 2], replay_k[replay_idx * 2]));
                        }
                    }
                    let nv = native_v_f32[native_idx];
                    let rv = replay_v_f32[replay_idx];
                    let vd = (nv - rv).abs();
                    diff.max_v_delta = diff.max_v_delta.max(vd);
                    if vd > 0.0 {
                        diff.v_mismatches += 1;
                        if diff.first_v_mismatch.is_none() {
                            diff.first_v_mismatch =
                                Some((h, t, d, native_v[native_idx * 2], replay_v[replay_idx * 2]));
                        }
                    }
                }
            }
        }
    }

    Ok(diff)
}
