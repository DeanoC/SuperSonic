mod decode_engine;
mod gemma4_engine;
mod gemma4_int4_engine;
mod llama31_engine;
mod oracle;
mod phi4_engine;
mod prefill_engine;
mod qwen35_dflash_engine;
mod registry;
mod validate;

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use base64::Engine as _;
use clap::Parser;

use decode_engine::{DecodeEngine, DecodeStageTimings};
use qwen35::loader::WeightLoader;
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
        },
        other => other.to_string(),
    }
}

fn model_dir_has_raw_safetensors(model_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(model_dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
    })
}

fn resolve_qwen_oracle_model_id(
    explicit_model_id: Option<&str>,
    model_dir: &Path,
    model_variant: &ModelVariant,
) -> String {
    if let Some(model_id) = explicit_model_id {
        return model_id.to_string();
    }
    if model_dir_has_raw_safetensors(model_dir) {
        return model_dir.to_string_lossy().into_owned();
    }
    model_variant.hf_model_id().to_string()
}

struct HostLmHeadRescorer {
    loader: WeightLoader,
    tensor_name: String,
}

impl HostLmHeadRescorer {
    fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        if !model_dir_has_raw_safetensors(model_dir) {
            return Ok(None);
        }
        let loader = WeightLoader::from_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("open raw lm_head weights: {e}"))?;
        let tensor_name = if loader.contains("lm_head.weight") {
            "lm_head.weight".to_string()
        } else if loader.contains("model.embed_tokens.weight") {
            "model.embed_tokens.weight".to_string()
        } else {
            return Ok(None);
        };
        Ok(Some(Self { loader, tensor_name }))
    }

    fn rescore(&self, normed: &[f32], candidate_ids: &[usize]) -> Result<u32> {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for &candidate in candidate_ids {
            let row = self
                .loader
                .load_bf16_row_f32(&self.tensor_name, candidate)
                .map_err(|e| anyhow::anyhow!("load lm_head row {candidate}: {e}"))?;
            anyhow::ensure!(
                row.len() == normed.len(),
                "lm_head row len {} != normed len {}",
                row.len(),
                normed.len()
            );
            let score = row
                .iter()
                .zip(normed.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
            if score > best_val {
                best_val = score;
                best_idx = candidate;
            }
        }
        Ok(best_idx as u32)
    }
}

fn top_k_candidate_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(usize, f32)> = Vec::new();
    for (idx, &val) in logits.iter().enumerate() {
        let pos = best
            .iter()
            .position(|&(_, best_val)| val > best_val)
            .unwrap_or(best.len());
        if pos < k {
            best.insert(pos, (idx, val));
            if best.len() > k {
                best.pop();
            }
        }
    }
    best.into_iter().map(|(idx, _)| idx).collect()
}

fn sample_qwen_logits_with_rescore(
    logits: &[f32],
    normed: Option<&[f32]>,
    rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let greedy = DecodeEngine::greedy_sample(logits);
    let Some(normed) = normed else {
        return Ok(greedy);
    };
    let Some(rescorer) = rescorer else {
        return Ok(greedy);
    };

    const RESCORE_MARGIN: f32 = 0.25;
    const RESCORE_TOP_K: usize = 4;

    let candidates = top_k_candidate_ids(logits, RESCORE_TOP_K);
    if candidates.len() < 2 {
        return Ok(greedy);
    }
    let top0 = logits[candidates[0]];
    let top1 = logits[candidates[1]];
    if top0 - top1 > RESCORE_MARGIN {
        return Ok(greedy);
    }

    let rescored = rescorer.rescore(normed, &candidates)?;
    if rescored != greedy {
        eprintln!(
            "[sample-rescore] token {} -> {} (top_margin={:.4})",
            greedy,
            rescored,
            top0 - top1
        );
    }
    Ok(rescored)
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

    /// Compute backend (`auto`, `hip`, or `cuda`)
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

    /// Load an INT8 bake produced from BitsAndBytes `load_in_8bit=True`.
    /// Currently only supported for `llama3.1-8b` on CUDA.
    #[arg(long)]
    int8: bool,

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

    /// Debug-only: run one prompt-prefill layer from the oracle's exact prefix
    /// state and compare our component layer outputs against the oracle.
    #[arg(long, hide = true)]
    trace_oracle_prefill_layer: Option<usize>,

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

    /// Legacy compatibility switch for older CUDA KV-FP8 bring-up commands.
    /// No longer required now that the validated 4B sm86 lane is public.
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
        anyhow::anyhow!("Unknown backend '{}'. Expected one of: auto, hip, cuda", cli.backend)
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

    if cli.int8 && (cli.int4 || cli.fp8_runtime) {
        anyhow::bail!("--int8 is mutually exclusive with --int4 and --fp8-runtime");
    }
    if cli.int8 && model_variant.family() != ModelFamily::Llama31 {
        anyhow::bail!("--int8 is currently supported only for llama3.1-8b on CUDA");
    }

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
        ModelFamily::Llama31 => {
            return llama31_engine::run_llama31(&cli, &model_variant, entry, ordinal, total_vram);
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
        FamilyParams::Llama31(_) => unreachable!("llama3.1 handled above"),
    };
    let host_lm_head_rescorer = HostLmHeadRescorer::from_model_dir(&cli.model_dir)?;

    // Install the per-model HIP launch preset (grid size + cooperative
    // flag) if the registry specifies one. User env vars still override
    // inside the bridge. Always called — `None` clears any stale preset
    // from a prior run, so switching models doesn't inherit the previous
    // one's grid. No-op on CUDA builds.
    {
        let (blocks, coop) = params.hip_launch_preset.unwrap_or((0, false));
        kernel_ffi::set_qwen35_4b_launch_preset(blocks, coop);
    }

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
            if model_variant != ModelVariant::Qwen3_5_4B || gpu_arch != GpuArch::Sm86 {
                anyhow::bail!("CUDA --kv-fp8 currently supports only qwen3.5-4b on sm86");
            }
            if env::var_os("SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR").is_none() {
                env::set_var("SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW", "128");
                eprintln!(
                    "[cuda] KV-FP8 BF16 sidecar window capped to the most recent 128 tokens on CUDA; \
                     set SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR=1 \
                     to restore full-prefix debug sidecar coverage"
                );
            }
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
    let allow_host_lm_head_rescore =
        cli.no_bake && !engine.weights().is_fp8 && !engine.weights().is_int4;

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
    if cli.trace_oracle_prefill_layer.is_some() && !cli.validate {
        anyhow::bail!("--trace-oracle-prefill-layer requires --validate");
    }

    // Run prefill (native GPU or oracle)
    let prefill_start = Instant::now();
    let (prefill_logits, native_prefill_trace, mut next_token) = if cli.oracle_prefill {
        let model_id = resolve_qwen_oracle_model_id(
            cli.model_id.as_deref(),
            &cli.model_dir,
            &model_variant,
        );
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("oracle/run_oracle.py");
        let output = oracle::run_oracle(
            &oracle_script, &model_id, &prompt_ids, cli.max_new_tokens,
            &cli.oracle_dtype, &oracle_device, true, false,
            fp8_oracle_dir.as_deref(),
            None,
        )?;
        engine.load_prefill_state(&output)?;
        let first = output.generated_token_ids[0];
        eprintln!("[prefill] oracle prefill done in {:.0}ms", prefill_start.elapsed().as_millis());
        (output.prefill_logits, None, first)
    } else {
        let prefill_result = if cli.trace_prefill_layers {
            engine.prefill_native_with_trace(&prompt_ids)?
        } else {
            engine.prefill_native_with_final_norm(&prompt_ids)?
        };
        let prefill_normed = prefill_result
            .final_norm_trace
            .as_deref()
            .map(decode_bf16_le);
        let first = sample_qwen_logits_with_rescore(
            &prefill_result.logits,
            prefill_normed.as_deref(),
            host_lm_head_rescorer
                .as_ref()
                .filter(|_| allow_host_lm_head_rescore),
        )?;
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
            first,
        )
    };

    // Optionally run oracle for validation
    let oracle_output = if cli.validate {
        let model_id = resolve_qwen_oracle_model_id(
            cli.model_id.as_deref(),
            &cli.model_dir,
            &model_variant,
        );
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("oracle/run_oracle.py");

        let emit_state = cli.trace_prefill_layers || cli.trace_oracle_prefill_layer.is_some();
        let output = oracle::run_oracle(
            &oracle_script,
            &model_id,
            &prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            &oracle_device,
            emit_state,
            false,
            fp8_oracle_dir.as_deref(),
            cli.trace_oracle_prefill_layer
                .filter(|layer| text_config.is_full_attention(*layer)),
        )?;

        // Compare prefill logits
        let prefill_delta = validate::max_abs_delta(&prefill_logits, &output.prefill_logits);
        eprintln!("[validate] prefill logit delta={prefill_delta:.4}");

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
	                let oracle_kv = output.kv_caches.as_ref();
	                let oracle_conv = output.conv_states.as_ref();
	                let oracle_recurrent = output.recurrent_states.as_ref();
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
	                    let state_delta = if text_config.is_full_attention(layer) {
	                        let native = engine.full_attention_prefix_cache_bf16_host(layer, 0);
	                        match (native, oracle_kv.and_then(|caches| caches.iter().find(|kv| kv.layer == layer))) {
	                            (Ok((native_k, native_v, _)), Some(oracle_kv)) => {
	                                let oracle_k = b64
	                                    .decode(&oracle_kv.k)
	                                    .map_err(|e| anyhow::anyhow!("decode oracle kv k[{layer}]: {e}"))?;
	                                let oracle_v = b64
	                                    .decode(&oracle_kv.v)
	                                    .map_err(|e| anyhow::anyhow!("decode oracle kv v[{layer}]: {e}"))?;
	                                format!(
	                                    " kv_k_delta={:.4} kv_v_delta={:.4}",
	                                    validate::max_abs_delta(&decode_bf16_le(&native_k), &decode_bf16_le(&oracle_k)),
	                                    validate::max_abs_delta(&decode_bf16_le(&native_v), &decode_bf16_le(&oracle_v)),
	                                )
	                            }
	                            _ => String::new(),
	                        }
	                    } else {
	                        let native_layer = engine.state_for_batch(0).layers.get(layer);
	                        match (
	                            native_layer,
	                            oracle_conv.and_then(|states| states.iter().find(|state| state.layer == layer)),
	                            oracle_recurrent.and_then(|states| states.iter().find(|state| state.layer == layer)),
	                        ) {
	                            (Some(native_layer), Some(oracle_conv), Some(oracle_recurrent)) => {
	                                let native_conv = native_layer
	                                    .conv_state
	                                    .as_ref()
	                                    .ok_or_else(|| anyhow::anyhow!("native linear layer {layer} missing conv_state"))?
	                                    .to_host_bytes()
	                                    .map_err(|e| anyhow::anyhow!("native conv D2H layer {layer}: {e}"))?;
	                                let native_recurrent = native_layer
	                                    .recurrent_state
	                                    .as_ref()
	                                    .ok_or_else(|| anyhow::anyhow!("native linear layer {layer} missing recurrent_state"))?
	                                    .to_host_bytes()
	                                    .map_err(|e| anyhow::anyhow!("native recurrent D2H layer {layer}: {e}"))?;
	                                let oracle_conv = b64
	                                    .decode(&oracle_conv.data)
	                                    .map_err(|e| anyhow::anyhow!("decode oracle conv[{layer}]: {e}"))?;
	                                let oracle_recurrent = b64
	                                    .decode(&oracle_recurrent.data)
	                                    .map_err(|e| anyhow::anyhow!("decode oracle recurrent[{layer}]: {e}"))?;
	                                format!(
	                                    " conv_delta={:.4} recurrent_delta={:.4}",
	                                    validate::max_abs_delta(&decode_bf16_le(&native_conv), &decode_bf16_le(&oracle_conv)),
	                                    validate::max_abs_delta(&decode_f32_le(&native_recurrent), &decode_f32_le(&oracle_recurrent)),
	                                )
	                            }
	                            _ => String::new(),
	                        }
	                    };
	                    if first_bad.is_none() && layer_delta > 0.5 {
	                        first_bad = Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta));
	                    }
	                    eprintln!(
	                        "[trace-prefill] layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}{state_delta}"
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
        }

        if let Some(trace_layer) = cli.trace_oracle_prefill_layer {
            trace_oracle_prefill_layer(
                &mut engine,
                trace_layer,
                &prompt_ids,
                &oracle_script,
                &model_id,
                &cli.oracle_dtype,
                &oracle_device,
                fp8_oracle_dir.as_deref(),
                &output,
            )?;
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
    let cuda_qwen2b_replay_default = backend == Backend::Cuda
        && model_variant == ModelVariant::Qwen3_5_2B
        && cli.batch_size == 1
        && params.use_4b_kernel
        && !cli.kv_fp8
        && !cli.force_kernel_decode
        && !cli.force_component_decode;
    let replay_decode_enabled = cli.batch_size == 1
        && params.use_4b_kernel
        && (cli.force_replay_decode || cuda_qwen2b_replay_default)
        && !cli.force_kernel_decode
        && !cli.force_component_decode
        && !cli.kv_fp8;
    let replay_kv_fp8_enabled = params.use_4b_kernel
        && cli.kv_fp8
        && cli.batch_size == 1
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
        if cuda_qwen2b_replay_default {
            eprintln!(
                "[decode] single-sequence CUDA qwen3.5-2b uses replayed GPU prefill for correctness"
            );
        } else {
            eprintln!("[decode] single-sequence 4B uses replayed GPU prefill for correctness");
        }
    } else if replay_kv_fp8_enabled && cli.batch_size == 1 {
        eprintln!("[decode] single-sequence CUDA KV-FP8 uses replayed GPU prefill for correctness");
    } else if cli.batch_size > 1 && params.use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] batched CUDA KV-FP8 uses the persistent kernel path");
    } else if component_single_decode_enabled {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the component decode path");
    } else if cli.batch_size == 1 && params.use_4b_kernel && cli.force_kernel_decode {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the kernel decode path");
    } else if cli.batch_size == 1 && params.use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] WARNING: single-sequence CUDA KV-FP8 uses the b=1 kernel path");
    } else if cuda_08b_hero_enabled {
        eprintln!("[decode] CUDA 0.8B sm86 hero path enabled");
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
            let mut can_rescore_with_normed = false;
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
                if let Some(trace_layer) = cli.trace_persistent_full_attn_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    trace_persistent_full_attn_layer(
                        &mut engine,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        &[next_token],
                        seqlen_offset,
                        ordinal,
                        params.kv_chunk_size,
                        cli.prefill_chunk_size,
                        params.use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if let Some(trace_layer) = cli.trace_persistent_linear_layer {
                    let trace_token_ids: Vec<u32> = prompt_ids
                        .iter()
                        .copied()
                        .chain(generated_ids.iter().copied())
                        .chain(std::iter::once(next_token))
                        .collect();
                    trace_persistent_linear_layer(
                        &mut engine,
                        trace_layer,
                        trace_token_ids.as_slice(),
                        &[next_token],
                        seqlen_offset,
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
                    can_rescore_with_normed = true;
                    logits
                } else {
                    can_rescore_with_normed = true;
                    engine.decode_step_batch(&[next_token], seqlen_offset)?.remove(0)
                }
            } else if cli.emit_stage_timings {
                let (logits, timings) = engine.decode_step_with_timings(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                can_rescore_with_normed = true;
                logits
            } else {
                can_rescore_with_normed = true;
                engine.decode_step(next_token, seqlen_offset)?
            };
            let native_token = if let Some(token) = maybe_fast_token {
                token
            } else {
                let normed = if can_rescore_with_normed && allow_host_lm_head_rescore {
                    Some(engine.last_normed_host_f32()?)
                } else {
                    None
                };
                sample_qwen_logits_with_rescore(
                    &logits,
                    normed.as_deref(),
                    host_lm_head_rescorer
                        .as_ref()
                        .filter(|_| allow_host_lm_head_rescore),
                )?
            };

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
        FamilyParams::Llama31(_) => unreachable!("dispatch filtered to Gemma4"),
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
    let pre_gate_best_match = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            let native_head = &native_pre_gate_f32[start..end];
            let (best_idx, best_delta) = (0..num_q_heads)
                .map(|cand| {
                    let cand_start = cand * head_dim;
                    let cand_end = cand_start + head_dim;
                    (
                        cand,
                        validate::max_abs_delta(
                            native_head,
                            &native_comp_pre_gate_f32[cand_start..cand_end],
                        ),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((h, f32::INFINITY));
            format!("{h}->{best_idx}:{best_delta:.6}")
        })
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
    let q_best_match = (0..num_q_heads)
        .map(|h| {
            let start = h * head_dim;
            let end = start + head_dim;
            let native_head = &native_q_f32[start..end];
            let (best_idx, best_delta) = (0..num_q_heads)
                .map(|cand| {
                    let cand_start = cand * head_dim;
                    let cand_end = cand_start + head_dim;
                    (
                        cand,
                        validate::max_abs_delta(
                            native_head,
                            &native_q_rope_f32[cand_start..cand_end],
                        ),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((h, f32::INFINITY));
            format!("{h}->{best_idx}:{best_delta:.6}")
        })
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
            "[trace-persistent-full-attn] layer={trace_layer} hidden_delta={hidden_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} native_vs_component_q={native_vs_component_q:.6} per_head_q=[{per_head_q_str}] q_best_match=[{q_best_match}] native_comp_vs_replay_k={native_vs_replay_k:.6} native_comp_vs_replay_v={native_vs_replay_v:.6} native_vs_component_saved_gate={native_vs_component_saved_gate:.6} native_vs_component_pre_gate={native_vs_component_pre_gate:.6} native_vs_host_pre_gate={native_vs_host_pre_gate:.6} kv_vs_bf16_hidden={kv_vs_bf16_hidden:.6} kv_vs_bf16_cache_k={kv_vs_bf16_cache_k:.6} kv_vs_bf16_cache_v={kv_vs_bf16_cache_v:.6} kv_vs_bf16_q={kv_vs_bf16_q:.6} kv_vs_bf16_saved_gate={kv_vs_bf16_saved_gate:.6} kv_vs_bf16_scores={kv_vs_bf16_scores:.6} kv_vs_bf16_scores_heads=[{kv_vs_bf16_scores_heads_str}] kv_vs_bf16_pre_gate={kv_vs_bf16_pre_gate:.6} per_head_host_pre_gate=[{per_head_host_pre_gate_str}] native_score_row_delta={score_row_delta:.6} per_head_score=[{per_head_score_str}] native_vs_component_gated={native_vs_component_gated:.6} native_vs_host_gated={native_vs_host_gated:.6} kv_vs_bf16_gated={kv_vs_bf16_gated:.6} native_vs_component_attn_hidden={native_vs_component_attn_hidden:.6} native_vs_host_o_proj={native_vs_host_o_proj:.6} kv_vs_bf16_attn_hidden={kv_vs_bf16_attn_hidden:.6} native_vs_replay_attn_hidden={native_vs_replay_attn_hidden:.6} native_cache_vs_replay_cache_attn_hidden={native_cache_vs_replay_cache_attn_hidden:.6} component_vs_replay_attn_hidden={component_vs_replay_attn_hidden:.6} per_head_pre_gate=[{per_head_pre_gate_str}] pre_gate_best_match=[{pre_gate_best_match}] cache_vs_quant_k_mismatches={cache_vs_quant_k} cache_vs_quant_v_mismatches={cache_vs_quant_v} cache_vs_quant_k_scale_delta={scale_vs_quant_k:.6} cache_vs_quant_v_scale_delta={scale_vs_quant_v:.6}"
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
            "[trace-persistent-full-attn] layer={trace_layer} hidden_delta={hidden_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} native_vs_component_q={native_vs_component_q:.6} per_head_q=[{per_head_q_str}] q_best_match=[{q_best_match}] native_comp_vs_replay_k={native_vs_replay_k:.6} native_comp_vs_replay_v={native_vs_replay_v:.6} native_vs_component_saved_gate={native_vs_component_saved_gate:.6} native_vs_component_pre_gate={native_vs_component_pre_gate:.6} native_score_row_delta={score_row_delta:.6} per_head_score=[{per_head_score_str}] native_vs_component_gated={native_vs_component_gated:.6} native_vs_component_attn_hidden={native_vs_component_attn_hidden:.6} native_vs_host_o_proj={native_vs_host_o_proj:.6} native_vs_replay_attn_hidden={native_vs_replay_attn_hidden:.6} native_cache_vs_replay_cache_attn_hidden={native_cache_vs_replay_cache_attn_hidden:.6} component_vs_replay_attn_hidden={component_vs_replay_attn_hidden:.6} per_head_pre_gate=[{per_head_pre_gate_str}] pre_gate_best_match=[{pre_gate_best_match}] cache_vs_component_k={cache_vs_component_k:.6} cache_vs_component_v={cache_vs_component_v:.6} cache_vs_replay_k={cache_vs_replay_k:.6} cache_vs_replay_v={cache_vs_replay_v:.6}"
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
    let pre_step_conv = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing pre-step layer {trace_layer}"))?
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing pre-step conv state for layer {trace_layer}"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("pre-step conv D2H layer {trace_layer}: {e}"))?;
    engine.set_hidden_from_bytes(&native_hidden)?;
    let (native_comp_trace, native_comp_conv, native_comp_recurrent, native_comp_hidden) =
        engine.component_trace_linear_layer_from_current_hidden(trace_layer)?;
    engine.rebuild_prefill_state(prefix_ids, true)?;
    engine.set_hidden_from_bytes(&native_hidden)?;
    let native_comp_layer = engine.component_trace_full_layer_from_current_hidden_with_seqlen(
        trace_layer,
        seqlen_offset,
    )?;

    engine.rebuild_prefill_state(prefix_ids, true)?;
    let native_hidden_out = engine
        .decode_step_batch_trace_hidden_after_layers(trace_tokens, seqlen_offset, trace_layer + 1, 0)?;
    let native_partial_conv = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native partial layer {trace_layer}"))?
        .conv_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing native partial conv state for layer {trace_layer}"))?
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("native partial conv D2H layer {trace_layer}: {e}"))?;
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
    let conv_state_len = cfg.linear_conv_kernel_dim - 1;
    let expected_conv_tail = {
        let start = decode_bf16_le(&pre_step_conv);
        let qkv = decode_bf16_le(&native_comp_trace.qkv);
        let mut expected = vec![0.0f32; qkv_dim * conv_state_len];
        for c in 0..qkv_dim {
            let base = c * conv_state_len;
            for t in 0..conv_state_len.saturating_sub(1) {
                expected[base + t] = start[base + t + 1];
            }
            expected[base + conv_state_len - 1] = qkv[c];
        }
        expected
    };
    let native_conv_vs_expected_tail =
        validate::max_abs_delta(&decode_bf16_le(&native_conv), &expected_conv_tail);
    let comp_conv_vs_expected_tail =
        validate::max_abs_delta(&decode_bf16_le(&native_comp_conv), &expected_conv_tail);
    let replay_conv_vs_expected_tail =
        validate::max_abs_delta(&decode_bf16_le(&replay_conv), &expected_conv_tail);
    let native_conv_tap_deltas = {
        let native = decode_bf16_le(&native_conv);
        let mut deltas = vec![0.0f32; conv_state_len];
        for c in 0..qkv_dim {
            let base = c * conv_state_len;
            for t in 0..conv_state_len {
                deltas[t] = deltas[t].max((native[base + t] - expected_conv_tail[base + t]).abs());
            }
        }
        deltas
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let native_qkv_proj_f32 = decode_f32_le(&native_qkv_proj);
    let native_z_proj_f32 = decode_f32_le(&native_z_proj);
    let comp_qkv_proj_f32 = decode_bf16_le(&native_comp_trace.qkv);
    let comp_z_proj_f32 = decode_bf16_le(&native_comp_trace.z);
    let (max_append_channel, max_append_mismatch) = {
        let native = decode_bf16_le(&native_conv);
        let start = decode_bf16_le(&pre_step_conv);
        let qkv = decode_bf16_le(&native_comp_trace.qkv);
        let conv_w = decode_bf16_le(
            &engine
                .weights()
                .layers
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("missing weights for layer {trace_layer}"))?
                .linear
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {trace_layer} is not linear"))?
                .conv1d_w
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("conv1d_w D2H layer {trace_layer}: {e}"))?,
        );
        let mut best = (0usize, 0.0f32);
        for c in 0..qkv_dim {
            let idx = c * conv_state_len + (conv_state_len - 1);
            let delta = (native[idx] - expected_conv_tail[idx]).abs();
            if delta > best.1 {
                best = (c, delta);
            }
        }
        let channel = best.0;
        let idx = channel * conv_state_len + (conv_state_len - 1);
        let weight_base = channel * cfg.linear_conv_kernel_dim;
        let state_base = channel * conv_state_len;
        let mut conv_acc = 0.0f32;
        for tap in 0..cfg.linear_conv_kernel_dim {
            let x = if tap + 1 == cfg.linear_conv_kernel_dim {
                qkv[channel]
            } else {
                start[state_base + tap]
            };
            conv_acc += x * conv_w[weight_base + tap];
        }
        let conv_out = bf16_round(conv_acc * sigmoid_fast(conv_acc));
        let native_last = native[idx];
        let mut nearest_qkv = (0usize, f32::INFINITY);
        for (i, &v) in native_qkv_proj_f32.iter().enumerate() {
            let delta = (v - native_last).abs();
            if delta < nearest_qkv.1 {
                nearest_qkv = (i, delta);
            }
        }
        (
            channel,
            format!(
                "channel={channel},native={:.6},expected={:.6},prev_last={:.6},qkv_comp={:.6},qkv_native={:.6},conv_out={:.6},nearest_qkv=(channel={},value={:.6},delta={:.6}),delta={:.6}",
                native[idx],
                expected_conv_tail[idx],
                start[idx],
                qkv[channel],
                native_qkv_proj_f32[channel],
                conv_out,
                nearest_qkv.0,
                native_qkv_proj_f32[nearest_qkv.0],
                nearest_qkv.1,
                best.1
            )
        )
    };
    engine.rebuild_prefill_state(prefix_ids, true)?;
    let step_b_debug_raw = engine.trace_persistent_linear_step_b_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer + 1,
        trace_layer,
        max_append_channel,
    )?;
    let step_b_debug = {
        let debug = decode_f32_le(&step_b_debug_raw);
        let native = decode_bf16_le(&native_conv);
        let partial = decode_bf16_le(&native_partial_conv);
        let start = decode_bf16_le(&pre_step_conv);
        let base = max_append_channel * conv_state_len;
        let idx = base + (conv_state_len - 1);
        let state_values = debug
            .iter()
            .take(conv_state_len)
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        let step_b_last = debug
            .get(conv_state_len - 1)
            .copied()
            .unwrap_or_default();
        let step_b_vs_expected = (step_b_last - expected_conv_tail[idx]).abs();
        let final_vs_step_b = (native[idx] - step_b_last).abs();
        let partial_vs_step_b = (partial[idx] - step_b_last).abs();
        let partial_vs_expected = (partial[idx] - expected_conv_tail[idx]).abs();
        format!(
            "channel={max_append_channel},state=[{state_values}],qkv={:.6},conv_out={:.6},shift0_expected={:.6},shift1_expected={:.6},append_expected={:.6},step_b_vs_expected={:.6},partial_last={:.6},partial_vs_step_b={:.6},partial_vs_expected={:.6},final_vs_step_b={:.6}",
            debug.get(conv_state_len).copied().unwrap_or_default(),
            debug.get(conv_state_len + 1).copied().unwrap_or_default(),
            start.get(base + 1).copied().unwrap_or_default(),
            start.get(base + 2).copied().unwrap_or_default(),
            expected_conv_tail[idx],
            step_b_vs_expected,
            partial[idx],
            partial_vs_step_b,
            partial_vs_expected,
            final_vs_step_b,
        )
    };
    let first_later_clobber = {
        let native = decode_bf16_le(&native_conv);
        let partial = decode_bf16_le(&native_partial_conv);
        let base = max_append_channel * conv_state_len;
        let idx = base + (conv_state_len - 1);
        let step_b_last = partial[idx];
        if (native[idx] - step_b_last).abs() == 0.0 || trace_layer + 1 >= text_config.num_hidden_layers {
            "none".to_string()
        } else {
            let mut lo = trace_layer + 1;
            let mut hi = text_config.num_hidden_layers;
            let mut hi_last = native[idx];
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                engine.rebuild_prefill_state(prefix_ids, true)?;
                let _ = engine.decode_step_batch_trace_hidden_after_layers(
                    trace_tokens,
                    seqlen_offset,
                    mid,
                    0,
                )?;
                let mid_conv = engine
                    .state_for_batch(0)
                    .layers
                    .get(trace_layer)
                    .ok_or_else(|| anyhow::anyhow!("missing binary-search layer {trace_layer}"))?
                    .conv_state
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing binary-search conv state for layer {trace_layer}"))?
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("binary-search conv D2H layer {trace_layer}: {e}"))?;
                let mid_vals = decode_bf16_le(&mid_conv);
                let mid_last = mid_vals[idx];
                if (mid_last - step_b_last).abs() > 0.0 {
                    hi = mid;
                    hi_last = mid_last;
                } else {
                    lo = mid;
                }
            }
            format!(
                "after_layers={hi},clobber_layer={},partial_last={:.6},clobbered_last={:.6},delta={:.6}",
                hi - 1,
                step_b_last,
                hi_last,
                (hi_last - step_b_last).abs()
            )
        }
    };
    let pointer_debug = {
        let state0 = engine.state_for_batch(0);
        let trace_layer_state = state0
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing trace layer state {trace_layer}"))?;
        let final_layer_idx = text_config.num_hidden_layers.saturating_sub(1);
        let final_layer_state = state0
            .layers
            .get(final_layer_idx)
            .ok_or_else(|| anyhow::anyhow!("missing final layer state {final_layer_idx}"))?;
        let trace_conv = trace_layer_state
            .conv_state
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        let trace_rec = trace_layer_state
            .recurrent_state
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        let final_k = final_layer_state
            .kv_cache_k
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        let final_v = final_layer_state
            .kv_cache_v
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        let final_shadow_k = final_layer_state
            .kv_shadow_k
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        let final_shadow_v = final_layer_state
            .kv_shadow_v
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        format!(
            "trace_conv=0x{trace_conv:x},trace_rec=0x{trace_rec:x},final_k=0x{final_k:x},final_v=0x{final_v:x},final_shadow_k=0x{final_shadow_k:x},final_shadow_v=0x{final_shadow_v:x},workspace=0x{:x}",
            engine.scratch_debug_ptr(),
        )
    };
    let isolated_tail_windows = {
        let final_layer_idx = text_config.num_hidden_layers.saturating_sub(1);
        let starts = [4usize, 5, 6, 7, 8];
        let mut samples = Vec::new();
        for &start_layer in &starts {
            if start_layer >= text_config.num_hidden_layers {
                continue;
            }
            let window_layers = text_config.num_hidden_layers - start_layer;
            engine.rebuild_prefill_state(prefix_ids, true)?;
            let pre_hidden = engine.decode_step_batch_trace_hidden_after_layers(
                trace_tokens,
                seqlen_offset,
                start_layer,
                0,
            )?;
            let before_conv = engine
                .state_for_batch(0)
                .layers
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("missing pre-window layer {trace_layer}"))?
                .conv_state
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing pre-window conv state for layer {trace_layer}"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("pre-window conv D2H layer {trace_layer}: {e}"))?;
            let _ = engine.debug_decode_window_from_hidden_bf16(
                &pre_hidden,
                seqlen_offset,
                start_layer,
                window_layers,
                0,
            )?;
            let after_conv = engine
                .state_for_batch(0)
                .layers
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("missing post-window layer {trace_layer}"))?
                .conv_state
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing post-window conv state for layer {trace_layer}"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("post-window conv D2H layer {trace_layer}: {e}"))?;
            let before_vals = decode_bf16_le(&before_conv);
            let after_vals = decode_bf16_le(&after_conv);
            let base = max_append_channel * conv_state_len;
            let idx = base + (conv_state_len - 1);
            samples.push(format!(
                "{}:{}:{:.6}",
                start_layer,
                window_layers,
                (after_vals[idx] - before_vals[idx]).abs()
            ));
        }
        samples.join(",")
    };
    let append_mismatch_samples = {
        let native = decode_bf16_le(&native_conv);
        let mut mismatches = Vec::with_capacity(qkv_dim);
        for c in 0..qkv_dim {
            let idx = c * conv_state_len + (conv_state_len - 1);
            mismatches.push((c, native[idx], expected_conv_tail[idx], (native[idx] - expected_conv_tail[idx]).abs()));
        }
        mismatches.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        mismatches
            .into_iter()
            .take(8)
            .map(|(channel, native_last, expected_last, delta)| {
                let mut nearest_qkv = (0usize, f32::INFINITY);
                for (i, &v) in native_qkv_proj_f32.iter().enumerate() {
                    let qkv_delta = (v - native_last).abs();
                    if qkv_delta < nearest_qkv.1 {
                        nearest_qkv = (i, qkv_delta);
                    }
                }
                format!(
                    "c{channel}->q{} native={:.6} expected={:.6} self_q={:.6} match_q={:.6} match_delta={:.6} delta={:.6}",
                    nearest_qkv.0,
                    native_last,
                    expected_last,
                    native_qkv_proj_f32[channel],
                    native_qkv_proj_f32[nearest_qkv.0],
                    nearest_qkv.1,
                    delta
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let comp_conv_tap_deltas = {
        let comp = decode_bf16_le(&native_comp_conv);
        let mut deltas = vec![0.0f32; conv_state_len];
        for c in 0..qkv_dim {
            let base = c * conv_state_len;
            for t in 0..conv_state_len {
                deltas[t] = deltas[t].max((comp[base + t] - expected_conv_tail[base + t]).abs());
            }
        }
        deltas
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
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
    let sample_qkv_native = native_qkv_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    let sample_qkv_comp = comp_qkv_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    let sample_z_native = native_z_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
    let sample_z_comp = comp_z_proj_f32.iter().take(4).map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");

    eprintln!(
        "[trace-persistent-linear] layer={trace_layer} hidden_delta={hidden_delta:.6} comp_vs_replay_conv={comp_vs_replay_conv:.6} comp_vs_replay_recurrent={comp_vs_replay_recurrent:.6} comp_linear_hidden_vs_replay={comp_vs_replay_hidden:.6} native_vs_comp_qkv_proj={native_vs_comp_qkv_proj:.6} native_vs_replay_qkv_proj={native_vs_replay_qkv_proj:.6} native_vs_comp_z_proj={native_vs_comp_z_proj:.6} native_vs_replay_z_proj={native_vs_replay_z_proj:.6} native_vs_comp_b_proj={native_vs_comp_b_proj:.6} native_vs_replay_b_proj={native_vs_replay_b_proj:.6} native_vs_comp_a_proj={native_vs_comp_a_proj:.6} native_vs_replay_a_proj={native_vs_replay_a_proj:.6} native_vs_comp_conv={native_vs_comp_conv:.6} native_vs_comp_recurrent={native_vs_comp_recurrent:.6} native_conv_vs_expected_tail={native_conv_vs_expected_tail:.6} comp_conv_vs_expected_tail={comp_conv_vs_expected_tail:.6} replay_conv_vs_expected_tail={replay_conv_vs_expected_tail:.6} native_conv_tap_deltas=[{native_conv_tap_deltas}] comp_conv_tap_deltas=[{comp_conv_tap_deltas}] max_append_mismatch=({max_append_mismatch}) step_b_debug=({step_b_debug}) first_later_clobber=({first_later_clobber}) pointer_debug=({pointer_debug}) isolated_tail_windows=[{isolated_tail_windows}] append_mismatch_samples=[{append_mismatch_samples}] native_vs_comp_token_mixer={native_vs_comp_token_mixer:.6} native_vs_replay_token_mixer={native_vs_replay_token_mixer:.6} native_vs_comp_post_norm={native_vs_comp_post_norm:.6} native_vs_replay_post_norm={native_vs_replay_post_norm:.6} native_vs_comp_gated={native_vs_comp_gated:.6} native_vs_replay_gated={native_vs_replay_gated:.6} native_vs_comp_swiglu={native_vs_comp_swiglu:.6} native_vs_replay_swiglu={native_vs_replay_swiglu:.6} native_vs_comp_mlp_down={native_vs_comp_mlp_down:.6} native_vs_replay_mlp_down={native_vs_replay_mlp_down:.6} native_vs_comp_proj_residual={native_vs_comp_proj_residual:.6} comp_layer_hidden_vs_replay={comp_layer_vs_replay_hidden:.6} native_vs_comp_layer_hidden={native_vs_comp_layer_hidden:.6} native_vs_replay_hidden={native_vs_replay_hidden:.6} sample_qkv_native=[{sample_qkv_native}] sample_qkv_comp=[{sample_qkv_comp}] sample_z_native=[{sample_z_native}] sample_z_comp=[{sample_z_comp}]"
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

fn trace_oracle_prefill_layer(
    engine: &mut DecodeEngine,
    trace_layer: usize,
    prompt_ids: &[u32],
    oracle_script: &Path,
    model_id: &str,
    oracle_dtype: &str,
    oracle_device: &str,
    fp8_oracle_dir: Option<&Path>,
    oracle_full: &oracle::OracleOutput,
) -> Result<()> {
    anyhow::ensure!(trace_layer > 0, "--trace-oracle-prefill-layer currently requires layer > 0");
    let row_bytes = engine.weights().config.hidden_size * 2;
    let prefix_ids = &prompt_ids[..prompt_ids.len() - 1];
    let prefix_oracle = if prefix_ids.is_empty() {
        None
    } else {
        Some(oracle::run_oracle(
            oracle_script,
            model_id,
            prefix_ids,
            1,
            oracle_dtype,
            oracle_device,
            true,
            false,
            fp8_oracle_dir,
            None,
        )?)
    };
    let mut native_prefix_k_delta = None;
    let mut native_prefix_v_delta = None;
    let mut native_prefix_conv_delta = None;
    let mut native_prefix_recurrent_delta = None;
    if engine.weights().config.is_full_attention(trace_layer) {
        if let Some(prefix_oracle) = prefix_oracle.as_ref() {
        engine.reset()?;
        let _ = engine.prefill_native(prefix_ids)?;
        let (native_prefix_k, native_prefix_v, native_prefix_len) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        engine.reset()?;
        engine.load_prefill_state(&prefix_oracle)?;
        let (oracle_prefix_k, oracle_prefix_v, oracle_prefix_len) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        anyhow::ensure!(
            native_prefix_len == oracle_prefix_len,
            "trace layer {trace_layer} native prefix len {} != oracle prefix len {}",
            native_prefix_len,
            oracle_prefix_len,
        );
        native_prefix_k_delta = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_prefix_k),
            &decode_bf16_le(&oracle_prefix_k),
        ));
        native_prefix_v_delta = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_prefix_v),
            &decode_bf16_le(&oracle_prefix_v),
        ));
        }
    } else {
        if let Some(prefix_oracle) = prefix_oracle.as_ref() {
        engine.reset()?;
        let _ = engine.prefill_native(prefix_ids)?;
        let native_layer = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing native prefix layer {trace_layer}"))?;
        let native_conv = native_layer
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native prefix layer {trace_layer} missing conv_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("native prefix conv D2H layer {trace_layer}: {e}"))?;
        let native_recurrent = native_layer
            .recurrent_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("native prefix layer {trace_layer} missing recurrent_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("native prefix recurrent D2H layer {trace_layer}: {e}"))?;

        engine.reset()?;
        engine.load_prefill_state(&prefix_oracle)?;
        let oracle_layer = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing oracle prefix layer {trace_layer}"))?;
        let oracle_conv = oracle_layer
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("oracle prefix layer {trace_layer} missing conv_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("oracle prefix conv D2H layer {trace_layer}: {e}"))?;
        let oracle_recurrent = oracle_layer
            .recurrent_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("oracle prefix layer {trace_layer} missing recurrent_state"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("oracle prefix recurrent D2H layer {trace_layer}: {e}"))?;

        native_prefix_conv_delta = Some(validate::max_abs_delta(
            &decode_bf16_le(&native_conv),
            &decode_bf16_le(&oracle_conv),
        ));
        native_prefix_recurrent_delta = Some(validate::max_abs_delta(
            &decode_f32_le(&native_recurrent),
            &decode_f32_le(&oracle_recurrent),
        ));
        }
    }
    engine.reset()?;
    if let Some(prefix_oracle) = prefix_oracle.as_ref() {
        engine.load_prefill_state(prefix_oracle)?;
    }

    let b64 = base64::engine::general_purpose::STANDARD;
    let last_row = |bytes: Vec<u8>, label: &str| -> Result<Vec<u8>> {
        anyhow::ensure!(
            bytes.len() % row_bytes == 0,
            "{label} bytes {} not divisible by row_bytes {}",
            bytes.len(),
            row_bytes,
        );
        if bytes.len() == row_bytes {
            return Ok(bytes);
        }
        let start = bytes.len() - row_bytes;
        Ok(bytes[start..].to_vec())
    };
    let oracle_inputs = oracle_full
        .layer_hidden_states
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_hidden_states"))?;
    let oracle_attn = oracle_full
        .layer_attn_residual_states
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_attn_residual_states"))?;
    let oracle_post = oracle_full
        .layer_post_attn_norm_states
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_post_attn_norm_states"))?;
    let oracle_mlp = oracle_full
        .layer_mlp_outputs
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("oracle output missing layer_mlp_outputs"))?;

    let oracle_input_bytes = last_row(
        b64.decode(
            oracle_inputs
                .get(trace_layer - 1)
                .ok_or_else(|| anyhow::anyhow!("oracle layer_hidden_states missing layer {}", trace_layer - 1))?,
        )
        .map_err(|e| anyhow::anyhow!("decode oracle input hidden for layer {trace_layer}: {e}"))?,
        "oracle input hidden",
    )?;
    let oracle_attn_bytes = last_row(
        b64.decode(
            oracle_attn
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("oracle layer_attn_residual_states missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow::anyhow!("decode oracle attn for layer {trace_layer}: {e}"))?,
        "oracle attn",
    )?;
    let oracle_post_bytes = last_row(
        b64.decode(
            oracle_post
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("oracle layer_post_attn_norm_states missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow::anyhow!("decode oracle post-norm for layer {trace_layer}: {e}"))?,
        "oracle post-norm",
    )?;
    let oracle_mlp_bytes = last_row(
        b64.decode(
            oracle_mlp
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("oracle layer_mlp_outputs missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow::anyhow!("decode oracle mlp for layer {trace_layer}: {e}"))?,
        "oracle mlp",
    )?;
    let oracle_hidden_bytes = last_row(
        b64.decode(
            oracle_inputs
                .get(trace_layer)
                .ok_or_else(|| anyhow::anyhow!("oracle layer_hidden_states missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow::anyhow!("decode oracle hidden for layer {trace_layer}: {e}"))?,
        "oracle hidden",
    )?;

    engine.set_hidden_from_bytes(&oracle_input_bytes)?;
    let trace = engine.component_trace_full_layer_from_current_hidden_with_seqlen(
        trace_layer,
        prefix_ids.len(),
    )?;
    let attn_delta =
        validate::max_abs_delta(&decode_bf16_le(&trace.attn_hidden), &decode_bf16_le(&oracle_attn_bytes));
    let post_delta =
        validate::max_abs_delta(&decode_bf16_le(&trace.post_attn_norm), &decode_bf16_le(&oracle_post_bytes));
    let mlp_delta =
        validate::max_abs_delta(&decode_bf16_le(&trace.mlp_out), &decode_bf16_le(&oracle_mlp_bytes));
    let hidden_delta =
        validate::max_abs_delta(&decode_bf16_le(&trace.layer_hidden), &decode_bf16_le(&oracle_hidden_bytes));
    eprintln!(
        "[trace-oracle-prefill-layer] layer={trace_layer} attn_delta={attn_delta:.6} post_norm_delta={post_delta:.6} mlp_delta={mlp_delta:.6} hidden_delta={hidden_delta:.6}"
    );
    if let (Some(k_delta), Some(v_delta)) = (native_prefix_k_delta, native_prefix_v_delta) {
        eprintln!(
            "[trace-oracle-prefix-kv] layer={trace_layer} k_delta={k_delta:.6} v_delta={v_delta:.6}"
        );
    }
    if let (Some(conv_delta), Some(recurrent_delta)) =
        (native_prefix_conv_delta, native_prefix_recurrent_delta)
    {
        eprintln!(
            "[trace-oracle-prefix-linear] layer={trace_layer} conv_delta={conv_delta:.6} recurrent_delta={recurrent_delta:.6}"
        );
    }

    if engine.weights().config.is_full_attention(trace_layer)
        && oracle_full.traced_full_attn_layer == Some(trace_layer)
    {
        let prefix_oracle = if let Some(prefix_oracle) = prefix_oracle.as_ref() {
            prefix_oracle
        } else {
            return Ok(());
        };
        // `component_trace_full_layer_from_current_hidden()` reuses the mutable
        // component decode path and overwrites slot 0 of the full-attention KV
        // cache for `trace_layer`. Reload the prefix oracle state so the deeper
        // attention trace below sees the original prefix cache, not the
        // trace-mutated one.
        engine.reset()?;
        engine.load_prefill_state(prefix_oracle)?;
        let decode_opt_bf16 = |field: &Option<String>, label: &str| -> Result<Vec<f32>> {
            let bytes = b64
                .decode(field.as_ref().ok_or_else(|| anyhow::anyhow!("oracle output missing {label}"))?)
                .map_err(|e| anyhow::anyhow!("decode oracle {label}: {e}"))?;
            Ok(decode_bf16_le(&bytes))
        };
        let oracle_normed = decode_opt_bf16(&oracle_full.traced_full_attn_normed, "traced_full_attn_normed")?;
        let oracle_q_proj = decode_opt_bf16(&oracle_full.traced_full_attn_q_proj, "traced_full_attn_q_proj")?;
        let oracle_gate = decode_opt_bf16(&oracle_full.traced_full_attn_gate_proj, "traced_full_attn_gate_proj")?;
        let oracle_k_proj = decode_opt_bf16(&oracle_full.traced_full_attn_k_proj, "traced_full_attn_k_proj")?;
        let oracle_v_proj = decode_opt_bf16(&oracle_full.traced_full_attn_v_proj, "traced_full_attn_v_proj")?;
        let oracle_q_rope = decode_opt_bf16(&oracle_full.traced_full_attn_q_rope, "traced_full_attn_q_rope")?;
        let oracle_k_rope = decode_opt_bf16(&oracle_full.traced_full_attn_k_rope, "traced_full_attn_k_rope")?;
        let oracle_pre_gate = decode_opt_bf16(&oracle_full.traced_full_attn_pre_gate, "traced_full_attn_pre_gate")?;
        let oracle_gated = decode_opt_bf16(&oracle_full.traced_full_attn_gated, "traced_full_attn_gated")?;
        let oracle_gated_actual = decode_opt_bf16(
            &oracle_full.traced_full_attn_gated_actual,
            "traced_full_attn_gated_actual",
        )?;
        let (prefix_k_bytes, prefix_v_bytes, prefix_len) =
            engine.full_attention_prefix_cache_bf16_host(trace_layer, 0)?;
        let oracle_prefix_kv = prefix_oracle
            .kv_caches
            .as_ref()
            .and_then(|caches| caches.iter().find(|kv| kv.layer == trace_layer))
            .ok_or_else(|| anyhow::anyhow!("prefix oracle missing kv cache for layer {trace_layer}"))?;
        let oracle_prefix_k = decode_bf16_le(
            &b64.decode(&oracle_prefix_kv.k)
                .map_err(|e| anyhow::anyhow!("decode prefix oracle K cache layer {trace_layer}: {e}"))?,
        );
        let oracle_prefix_v = decode_bf16_le(
            &b64.decode(&oracle_prefix_kv.v)
                .map_err(|e| anyhow::anyhow!("decode prefix oracle V cache layer {trace_layer}: {e}"))?,
        );

        let stage = engine.trace_full_attention_stages_from_hidden(trace_layer, &oracle_input_bytes, prefix_ids.len())?;
        let stage_out = engine.trace_full_attention_layer_output_from_hidden_current_state(
            trace_layer,
            0,
            &oracle_input_bytes,
            prefix_ids.len(),
        )?;
        let normed_delta = validate::max_abs_delta(&decode_bf16_le(&stage.normed), &oracle_normed);
        let q_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.q_proj), &oracle_q_proj);
        let gate_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.gate_proj), &oracle_gate);
        let k_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.k_proj), &oracle_k_proj);
        let v_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.v_proj), &oracle_v_proj);
        let q_rope_delta = validate::max_abs_delta(&decode_bf16_le(&stage.q_rope), &oracle_q_rope);
        let k_rope_delta = validate::max_abs_delta(&decode_bf16_le(&stage.k_rope), &oracle_k_rope);
        let pre_gate_stage_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.pre_gate), &oracle_pre_gate);
        let gated_stage_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.gated), &oracle_gated);
        let gated_actual_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.gated), &oracle_gated_actual);
        let gated_reconstruct_delta =
            validate::max_abs_delta(&oracle_gated, &oracle_gated_actual);
        let head_dim = engine.weights().config.head_dim;
        let num_heads = engine.weights().config.num_attention_heads;
        let num_kv_heads = engine.weights().config.num_key_value_heads;
        let kv_groups = num_heads / num_kv_heads;
        let pre_gate_host = decode_bf16_le(&stage_out.pre_gate);
        let q_rope_host = decode_bf16_le(&stage.q_rope);
        let k_rope_step = decode_bf16_le(&stage.k_rope);
        let v_step = decode_bf16_le(&stage.v_proj);
        let prefix_k = decode_bf16_le(&prefix_k_bytes);
        let prefix_v = decode_bf16_le(&prefix_v_bytes);
        let loaded_layer = engine
            .state_for_batch(0)
            .layers
            .get(trace_layer)
            .ok_or_else(|| anyhow::anyhow!("missing loaded layer {trace_layer}"))?;
        let loaded_raw_k = decode_bf16_le(
            &loaded_layer
                .kv_cache_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("loaded layer {trace_layer} missing K cache"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("loaded layer {trace_layer} K cache D2H: {e}"))?,
        );
        let loaded_raw_v = decode_bf16_le(
            &loaded_layer
                .kv_cache_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("loaded layer {trace_layer} missing V cache"))?
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("loaded layer {trace_layer} V cache D2H: {e}"))?,
        );
        anyhow::ensure!(
            prefix_len == prefix_ids.len(),
            "trace layer {trace_layer} prefix len {} != prompt prefix len {}",
            prefix_len,
            prefix_ids.len(),
        );
        let kv_len = prefix_len + 1;
        let mut full_k = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        let mut full_v = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        for kvh in 0..num_kv_heads {
            let prefix_base = kvh * prefix_len * head_dim;
            let full_base = kvh * kv_len * head_dim;
            let step_base = kvh * head_dim;
            full_k[full_base..full_base + prefix_len * head_dim]
                .copy_from_slice(&prefix_k[prefix_base..prefix_base + prefix_len * head_dim]);
            full_v[full_base..full_base + prefix_len * head_dim]
                .copy_from_slice(&prefix_v[prefix_base..prefix_base + prefix_len * head_dim]);
            full_k[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&k_rope_step[step_base..step_base + head_dim]);
            full_v[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&v_step[step_base..step_base + head_dim]);
        }
        let mut host_attn_pre_gate = vec![0.0f32; num_heads * head_dim];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for qh in 0..num_heads {
            let kvh = qh / kv_groups;
            let q_base = qh * head_dim;
            let mut scores = vec![0.0f32; kv_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_base = (kvh * kv_len + t) * head_dim;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q_rope_host[q_base + d] * full_k[k_base + d];
                }
                *score = acc * scale;
            }
            let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut weights = vec![0.0f32; kv_len];
            for (idx, score) in scores.iter().copied().enumerate() {
                let w = (score - row_max).exp();
                weights[idx] = w;
                denom += w;
            }
            let out_base = qh * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    let v_base = (kvh * kv_len + t) * head_dim;
                    acc += w * full_v[v_base + d];
                }
                host_attn_pre_gate[out_base + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let mut oracle_host_pre_gate = vec![0.0f32; num_heads * head_dim];
        let mut oracle_full_k = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        let mut oracle_full_v = vec![0.0f32; num_kv_heads * kv_len * head_dim];
        for kvh in 0..num_kv_heads {
            let prefix_base = kvh * prefix_len * head_dim;
            let full_base = kvh * kv_len * head_dim;
            let step_base = kvh * head_dim;
            oracle_full_k[full_base..full_base + prefix_len * head_dim]
                .copy_from_slice(&oracle_prefix_k[prefix_base..prefix_base + prefix_len * head_dim]);
            oracle_full_v[full_base..full_base + prefix_len * head_dim]
                .copy_from_slice(&oracle_prefix_v[prefix_base..prefix_base + prefix_len * head_dim]);
            oracle_full_k[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&oracle_k_rope[step_base..step_base + head_dim]);
            oracle_full_v[full_base + prefix_len * head_dim..full_base + kv_len * head_dim]
                .copy_from_slice(&oracle_v_proj[step_base..step_base + head_dim]);
        }
        for qh in 0..num_heads {
            let kvh = qh / kv_groups;
            let q_base = qh * head_dim;
            let mut scores = vec![0.0f32; kv_len];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_base = (kvh * kv_len + t) * head_dim;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += oracle_q_rope[q_base + d] * oracle_full_k[k_base + d];
                }
                *score = acc * scale;
            }
            let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut weights = vec![0.0f32; kv_len];
            for (idx, score) in scores.iter().copied().enumerate() {
                let w = (score - row_max).exp();
                weights[idx] = w;
                denom += w;
            }
            let out_base = qh * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in weights.iter().enumerate() {
                    let v_base = (kvh * kv_len + t) * head_dim;
                    acc += w * oracle_full_v[v_base + d];
                }
                oracle_host_pre_gate[out_base + d] = if denom > 0.0 { acc / denom } else { 0.0 };
            }
        }
        let host_pre_gate_vs_stage =
            validate::max_abs_delta(&host_attn_pre_gate, &pre_gate_host);
        let host_pre_gate_vs_oracle =
            validate::max_abs_delta(&host_attn_pre_gate, &oracle_pre_gate);
        let oracle_host_pre_gate_vs_oracle =
            validate::max_abs_delta(&oracle_host_pre_gate, &oracle_pre_gate);
        let kernel_pre_gate_direct = {
            let ordinal = engine.ordinal();
            let q_gpu = gpu_hal::GpuBuffer::from_host_bytes(
                ordinal,
                gpu_hal::ScalarType::BF16,
                &[num_heads, 1, head_dim],
                &f32_to_bf16_bytes(q_rope_host.iter().copied()),
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn q H2D: {e}"))?;
            let k_gpu = gpu_hal::GpuBuffer::from_host_bytes(
                ordinal,
                gpu_hal::ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &f32_to_bf16_bytes(full_k.iter().copied()),
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn k H2D: {e}"))?;
            let v_gpu = gpu_hal::GpuBuffer::from_host_bytes(
                ordinal,
                gpu_hal::ScalarType::BF16,
                &[num_kv_heads, kv_len, head_dim],
                &f32_to_bf16_bytes(full_v.iter().copied()),
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn v H2D: {e}"))?;
            let mut out_gpu = gpu_hal::GpuBuffer::zeros(
                ordinal,
                gpu_hal::ScalarType::F32,
                &[num_heads, 1, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn out alloc: {e}"))?;
            kernel_ffi::prefill_ffi::full_attention_prefill(
                ordinal,
                gpu_hal::ScalarType::BF16,
                1,
                num_heads,
                num_kv_heads,
                1,
                kv_len,
                head_dim,
                scale,
                prefix_len,
                &q_gpu,
                &k_gpu,
                &v_gpu,
                &mut out_gpu,
            )
            .map_err(|e| anyhow::anyhow!("trace direct attn kernel: {e}"))?;
            decode_f32_le(
                &out_gpu
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("trace direct attn out D2H: {e}"))?,
            )
        };
        let direct_kernel_vs_host =
            validate::max_abs_delta(&kernel_pre_gate_direct, &host_attn_pre_gate);
        let direct_kernel_vs_oracle =
            validate::max_abs_delta(&kernel_pre_gate_direct, &oracle_pre_gate);
        let loaded_prefix_k_vs_oracle =
            validate::max_abs_delta(&prefix_k, &oracle_prefix_k);
        let loaded_prefix_v_vs_oracle =
            validate::max_abs_delta(&prefix_v, &oracle_prefix_v);
        let loaded_raw_k_vs_oracle =
            validate::max_abs_delta(&loaded_raw_k, &oracle_prefix_k);
        let loaded_raw_v_vs_oracle =
            validate::max_abs_delta(&loaded_raw_v, &oracle_prefix_v);
        let mut head_deltas = Vec::with_capacity(num_heads);
        for head in 0..num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            head_deltas.push(validate::max_abs_delta(
                &pre_gate_host[start..end],
                &oracle_pre_gate[start..end],
            ));
        }
        eprintln!(
            "[trace-oracle-full-attn] layer={trace_layer} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} k_rope_delta={k_rope_delta:.6} pre_gate_delta={pre_gate_stage_delta:.6} host_pre_gate_vs_stage={host_pre_gate_vs_stage:.6} host_pre_gate_vs_oracle={host_pre_gate_vs_oracle:.6} oracle_host_pre_gate_vs_oracle={oracle_host_pre_gate_vs_oracle:.6} direct_kernel_vs_host={direct_kernel_vs_host:.6} direct_kernel_vs_oracle={direct_kernel_vs_oracle:.6} loaded_prefix_k_vs_oracle={loaded_prefix_k_vs_oracle:.6} loaded_prefix_v_vs_oracle={loaded_prefix_v_vs_oracle:.6} loaded_raw_k_vs_oracle={loaded_raw_k_vs_oracle:.6} loaded_raw_v_vs_oracle={loaded_raw_v_vs_oracle:.6} gated_delta={gated_stage_delta:.6} gated_actual_delta={gated_actual_delta:.6} gated_reconstruct_delta={gated_reconstruct_delta:.6} pre_gate_head_deltas={head_deltas:?}"
        );
    }
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
