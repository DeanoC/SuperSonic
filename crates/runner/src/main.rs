#![recursion_limit = "512"]

mod bakes;
mod backend_runtime;
mod certified_kv;
mod decode_engine;
mod gemma4_engine;
mod gemma4_int4_engine;
mod gemma4_runtime;
mod llama31_engine;
mod model_files;
mod oracle;
mod phi4_engine;
mod policy;
mod prefill_engine;
mod profiling;
mod qwen35_dflash_engine;
mod qwen35_kv_trace;
mod qwen35_runtime;
mod qwen35_trace;
mod qwen35_trace_utils;
#[path = "qwen36_moe/mod.rs"]
mod qwen36_moe_cli;
#[path = "qwen36_moe/decode.rs"]
mod qwen36_moe_decode;
#[path = "qwen36_moe/logits.rs"]
mod qwen36_moe_logits;
#[path = "qwen36_moe/mtp.rs"]
mod qwen36_moe_mtp;
#[path = "qwen36_moe/persistent_decode.rs"]
mod qwen36_moe_persistent_decode;
#[path = "qwen36_moe/residency.rs"]
mod qwen36_moe_residency;
#[path = "qwen36_moe/residency_pages.rs"]
mod qwen36_moe_residency_pages;
#[path = "qwen36_moe/residency_types.rs"]
mod qwen36_moe_residency_types;
#[path = "qwen36_moe/speculative.rs"]
mod qwen36_moe_speculative;
#[path = "qwen36_moe/state.rs"]
mod qwen36_moe_state;
#[path = "qwen36_moe/telemetry.rs"]
mod qwen36_moe_telemetry;
#[path = "qwen36_moe/types.rs"]
mod qwen36_moe_types;
mod registry;
mod specprefill;
mod specprefill_engine;
mod tensor_bytes;
mod validate;

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use bakes::ensure_hf_metadata_present;
pub(crate) use bakes::{should_fetch_exact_bake, try_download_bake};
use backend_runtime::resolve_backend;
use decode_engine::{DecodeEngine, DecodeStageTimings};
use gemma4_runtime::run_gemma4;
pub(crate) use model_files::{load_tokenizer, resolve_prompt_token_ids};
pub(crate) use backend_runtime::resolve_oracle_device;
use policy::{
    q4km_like, validate_dflash_flags, validate_gfx942_policy, validate_global_flags,
    validate_specprefill_flags,
};
use profiling::MetalProfileScope;
use qwen35::state::ModelState;
use qwen35_kv_trace::trace_kv_cache;
use qwen35_runtime::{
    check_qwen35_vram, load_qwen35_engine, load_qwen35_startup, qwen35_oracle_script_path,
    report_qwen35_virtual_kv_after_prefill, resolve_qwen_oracle_model_id,
    run_qwen35_oracle_validation, run_qwen35_prefill, sample_qwen_logits_with_rescore,
    trace_qwen35_oracle_prefill_layer, validate_qwen35_startup, HostLmHeadRescorer,
    Qwen35EngineSetup, Qwen35Policy, Qwen35Prefill, Qwen35Startup,
};
use qwen35_trace::{
    trace_component_input_layer, trace_component_layer, trace_component_linear_layer,
    trace_component_linear_state_layer, trace_persistent_input_layer,
    trace_persistent_linear_state_layer,
};
use qwen35_trace_utils::{
    bf16_residual_sum, bf16_round, fp8_e4m3_to_f32_host, sigmoid_fast,
};
use registry::{Backend, FamilyParams, GpuArch, ModelFamily, ModelVariant};
use supersonic_core::backend::{BackendChoice, BACKEND_CHOICES};
use tensor_bytes::{
    bf16_bytes_to_f32 as decode_bf16_le, f32_bytes_to_f32 as decode_f32_le, f32_to_bf16_bytes,
};

#[derive(Parser)]
#[command(name = "supersonic", about = "SuperSonic — optimized LLM inference")]
pub(crate) struct Cli {
    /// Model variant (e.g. "qwen3.5-0.8b")
    #[arg(long, default_value = "qwen3.5-0.8b")]
    model: String,

    /// Path to HuggingFace model directory (containing config.json + safetensors)
    #[arg(long)]
    model_dir: PathBuf,

    /// Text prompt (will be tokenized). Required unless --dry-run is set.
    #[arg(long, required_unless_present = "dry_run", default_value = "")]
    prompt: String,

    /// Do not add tokenizer special tokens when encoding --prompt.
    #[arg(long)]
    prompt_no_special_tokens: bool,

    /// Maximum tokens to generate
    #[arg(long, default_value = "8")]
    max_new_tokens: usize,

    /// Sampling temperature. 0 = greedy argmax (default; reproducible).
    /// Typical sampled values: 0.7–1.0.
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Top-K filter: keep only the K highest-probability tokens before
    /// sampling. 0 = no cap (full vocab considered).
    #[arg(long, default_value = "0")]
    top_k: usize,

    /// Top-P (nucleus) filter: keep the smallest set of tokens whose
    /// cumulative probability ≥ top_p. 1.0 = no truncation.
    #[arg(long, default_value = "1.0")]
    top_p: f32,

    /// RNG seed for non-greedy sampling. Same seed + same prompt + same
    /// model = bit-identical generation.
    #[arg(long, default_value = "42")]
    sampling_seed: u64,

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

    /// Enable Qwen3.6-MoE self-speculative decode.
    ///
    /// When set, the engine loads the multi-token-prediction (MTP) head
    /// from the bake and runs draft generation plus base-model verification.
    /// When unset (default), MTP weights are not loaded and self-spec decode
    /// is unavailable, leaving the extra VRAM for KV cache and scratch.
    ///
    /// Currently HIP/qwen3.6-MoE only; ignored for other model families.
    #[arg(long)]
    speculative_decode: bool,

    /// Enable batched-K speculative verify (Phase 6.4c.2 experimental).
    ///
    /// When set together with --speculative-decode, the engine routes
    /// the verify path through `run_speculative_decode_step_batched`
    /// + `LinearAttnSnapshot` save/restore instead of the per-step
    /// closure with early termination. The batched path runs all K+1
    /// verify chains, batches the K+1 lm_head GEMVs into one launch
    /// (Phase 6.4a kernel), and on rejection restores linear-attn
    /// state to the pre-spec snapshot then replays the accepted
    /// prefix sequentially.
    ///
    /// Trade-off: net win when MTP accept rate is high (fewer total
    /// base steps per emitted token); net loss when it's low (the
    /// always-K+1 chains + replay outweigh batched lm_head savings).
    /// Default off — opt in to measure on a specific workload.
    /// Bit-identical greedy output to the per-step path.
    #[arg(long)]
    batched_spec_verify: bool,

    /// **No-op.** The Qwen3.6-MoE persistent megakernel
    /// (`kernels/qwen36_moe_persistent/persistent_decode.hip`,
    /// PR #126) is now the default decode path; this flag is kept as
    /// a no-op so existing benchmark harnesses don't break. Use
    /// `--no-persistent-decode` to opt out into the legacy chained
    /// per-step launcher.
    ///
    /// Background: one cooperative HIP launch processes all 40
    /// layers per token vs. 80 step launches in the chained path —
    /// recovers ~2.5-3 ms/token of HIP launch overhead. Validated
    /// bit-identical across PG-19 + RULER × {128, 512, 2K, 4K, 8K}
    /// at INT4 (verify-suite `chained_vs_persistent` block, all
    /// 10/10 cases logits/hidden/generated_ids byte-identical).
    /// Mode-agnostic: the kernel handles INT4 / BF16 / FP8 sidecars
    /// identically. HIP/qwen3.6-MoE only.
    #[arg(long, hide = true)]
    persistent_decode: bool,

    /// Opt out of the persistent-decode megakernel and run the
    /// legacy chained per-step launcher (~80 step launches per
    /// token, ~2.5-3 ms/token slower than the persistent path on
    /// gfx1100). Use for A/B perf comparison or to bisect a
    /// suspected megakernel-side regression.
    #[arg(long)]
    no_persistent_decode: bool,

    /// Emit the generated suffix as a JSON string for benchmark harnesses.
    #[arg(long)]
    emit_generated_json: bool,

    /// Score the prompt with teacher forcing instead of generating new tokens.
    /// Currently wired for the Llama 3.1 CUDA path as the PG-19/perplexity QA
    /// surface.
    #[arg(long)]
    teacher_forced: bool,

    /// For certified teacher-forced scoring, use dense decode through this
    /// prefix length, skip the boundary target, then score the suffix with the
    /// certified path. This matches the DotCache PG-19 protocol.
    #[arg(long, hide = true)]
    teacher_forced_dense_prefix_len: Option<usize>,

    /// Debug-only: force the legacy one-token-prefill + decode-step teacher-forced scorer.
    #[arg(long, hide = true)]
    teacher_forced_decode_step: bool,

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

    /// Use a GGUF-like Q4KM bake in SuperSonic's native low-bit runtime layout.
    /// Runtime executes the translated bake, not GGML blocks directly.
    #[arg(long)]
    q4km: bool,

    /// Use a Q4KM-sourced GPTQ bake in SuperSonic's native INT4 runtime layout.
    #[arg(long)]
    q4km_gptq: bool,

    /// Optional GGUF source file to translate into a native q4km bake.
    #[arg(long)]
    gguf_file: Option<PathBuf>,

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

    /// Enable the staged certified tiered KV path for Llama 3.1 CUDA INT8.
    #[arg(long)]
    certified_kv: bool,

    /// Hidden research preset for certified KV defaults.
    #[arg(long, default_value = "legacy", hide = true)]
    certified_kv_preset: certified_kv::CertifiedKvPreset,

    /// Optional JSONL telemetry path for certified KV stage and fallback counters.
    #[arg(long)]
    certified_kv_telemetry: Option<PathBuf>,

    /// Quantize real BF16 KV caches after prefill and report shadow-cache telemetry.
    #[arg(long, hide = true)]
    certified_kv_shadow_validate: bool,

    /// Debug-only: compare one full-attention layer using dense KV vs certified KV.
    #[arg(long, hide = true)]
    certified_kv_trace_layer: Option<usize>,

    /// Debug-only: sweep all full-attention layers using dense KV vs certified KV.
    #[arg(long, hide = true)]
    certified_kv_trace_all: bool,

    /// Certified KV cache block size in tokens.
    #[arg(long, default_value = "16", hide = true)]
    certified_kv_block_size: usize,

    /// Certified KV INT4 value quantization group size.
    #[arg(long, default_value = "16", hide = true)]
    certified_kv_value_group_size: usize,

    /// Debug-only: use BF16 values with INT8 keys in the certified KV trace path.
    #[arg(long, hide = true)]
    certified_kv_bf16_values: bool,

    /// Certified KV adaptive coverage target.
    #[arg(long, default_value = "0.995", hide = true)]
    certified_kv_tau_cov: f32,

    /// Certified KV minimum FP16 key blocks per head.
    #[arg(long, default_value = "2", hide = true)]
    certified_kv_k_min: usize,

    /// Certified KV maximum FP16 key blocks per head before Rung 1 expansion (0 = unclamped).
    #[arg(long, default_value = "128", hide = true)]
    certified_kv_k_max: usize,

    /// Certified KV value error promotion threshold.
    #[arg(long, default_value = "0.05", hide = true)]
    certified_kv_v_tol: f32,

    /// Promoted BF16 value blocks cached per layer/KV head (0 disables cache).
    #[arg(long, default_value = "128", hide = true)]
    certified_kv_value_cache_blocks: usize,

    /// Certified KV ranking-consistency depth.
    #[arg(long, default_value = "1", hide = true)]
    certified_kv_ranking_r: usize,

    /// Certified KV Rung 1 tail-mass threshold.
    #[arg(long, default_value = "0.005", hide = true)]
    certified_kv_rung1_threshold: f32,

    /// Certified KV Rung 1 K expansion multiplier.
    #[arg(long, default_value = "2.0", hide = true)]
    certified_kv_rung1_multiplier: f32,

    /// FP16 key scratch cache capacity in blocks per layer/KV head.
    #[arg(long, default_value = "256", hide = true)]
    certified_kv_key_cache_blocks: usize,

    /// Certified KV guard exponent applied to tail-mass certificates.
    #[arg(long, default_value = "3.0", hide = true)]
    certified_kv_delta_guard_factor: f32,

    /// Fraction of tail blocks to score-check opportunistically.
    #[arg(long, default_value = "0.01", hide = true)]
    certified_kv_score_exploration_rate: f32,

    /// Debug-only: allow certified KV to continue when the tail certificate misses threshold.
    #[arg(long, hide = true)]
    certified_kv_allow_uncertified_tail: bool,

    /// Certified KV score-consistency guard epsilon.
    #[arg(long, default_value = "0.0001", hide = true)]
    certified_kv_eps_guard: f32,

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
    /// Default 1. Supported on Qwen3.5 through the 4B-capable persistent
    /// kernel and Gemma 4 BF16 + INT4 via per-family batched megakernels.
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

    /// When set, after the final decode step the runner prints
    /// `LAST_LOGITS: <comma-separated f32>` to stdout exactly once.
    /// Used by integration parity tests to compare logits across
    /// runs (e.g., --kv-fp8 vs BF16 KV).
    #[arg(long, default_value_t = false)]
    dump_last_logits: bool,

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

    /// Enumerate the model checkpoint, compute analytic + on-disk VRAM
    /// accounting, print the decode budget report, and exit.
    #[arg(long)]
    dry_run: bool,

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

    /// Path to the SpecPrefill (arXiv 2502.02789) draft model directory
    /// (e.g. `/mnt/data/models/Qwen3.5-0.8B`). Presence of this flag
    /// enables sparse target prefill via the speculator's importance
    /// signal. Currently only supported for `--model qwen3.5-9b`.
    #[arg(long)]
    specprefill_draft_dir: Option<PathBuf>,

    /// SpecPrefill keep ratio per chunk: fraction of tokens kept by the
    /// chunked top-K selection. Phase A2 measurements pin 0.50 as the
    /// quality-stable default on Qwen3.5-9B (cossim ≥ 0.927, argmax
    /// match). Range: [0.05, 1.0]. Default applied in run_specprefill: 0.50.
    #[arg(long)]
    specprefill_keep_ratio: Option<f32>,

    /// SpecPrefill chunk size for top-K selection (paper §3.4).
    /// Default applied in run_specprefill: 32.
    #[arg(long)]
    specprefill_chunk_size: Option<usize>,

    /// SpecPrefill 1-D average-pool window for score smoothing. Must be
    /// odd. Paper uses 5-10. Default applied in run_specprefill: 5.
    #[arg(long)]
    specprefill_pool_window: Option<usize>,

    /// SpecPrefill look-ahead decode steps on the draft (paper §3.3
    /// default 4). Total query rows harvested = lookahead + 1.
    /// Default applied in run_specprefill: 4.
    #[arg(long)]
    specprefill_lookahead: Option<usize>,

    /// SpecPrefill always-keep prefix (BOS + system) length.
    /// Default applied in run_specprefill: 4.
    #[arg(long)]
    specprefill_always_keep_prefix: Option<usize>,

    /// SpecPrefill always-keep suffix (final query) length.
    /// Default applied in run_specprefill: 4.
    #[arg(long)]
    specprefill_always_keep_suffix: Option<usize>,

    /// Free the draft weights after selection runs and before the target
    /// prefill, to claw back ~1.6 GiB on a tight 24 GiB budget.
    #[arg(long, default_value_t = false)]
    specprefill_unload_draft: bool,

    /// SpecPrefill scoring algorithm.
    ///
    /// - `cosine` (Phase D, default): per-block cosine(block_mean_K,
    ///   last_K) from one drafter K cache. Fast path — drops the
    ///   lookahead decode steps entirely. Default scoring layer is the
    ///   shallowest full-attention layer; override via env var
    ///   `SUPERSONIC_SPECPREFILL_SCORE_LAYER=N`. Per-layer aggregation
    ///   mode controlled by env var `SUPERSONIC_SPECPREFILL_LAYERS=
    ///   shallowest|all_max` (default `shallowest`). Measured 4134 ms
    ///   TTFT on a 1353-token prompt vs 5192 ms dense (1.26× faster)
    ///   on gfx1100. See docs/specprefill.md § Algorithm.
    /// - `lookahead` (Phase C, legacy): per-layer softmax(Q·Kᵀ) over
    ///   look-ahead query rows. Correctness-validated but NET SLOWER
    ///   than dense prefill on gfx1100 (7895 ms in the same
    ///   measurement) because the lookahead decode steps route through
    ///   the component decode path. Kept available for parity-test
    ///   coverage and future research.
    #[arg(long, default_value = "cosine")]
    specprefill_algorithm: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let _metal_profile_scope = MetalProfileScope::new();
    let ordinal = cli.device;
    let backend_choice = BackendChoice::parse(&cli.backend).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown backend '{}'. Expected one of: {}",
            cli.backend,
            BACKEND_CHOICES
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

    validate_global_flags(&cli, &model_variant, backend)?;
    let q4km_like = q4km_like(&cli);
    // 2. Detect GPU
    let (arch_name, total_vram, warp_size) = match backend {
        Backend::Hip => {
            let (arch_name, total_vram) = kernel_ffi::query_gpu_info(ordinal)
                .map_err(|e| anyhow::anyhow!("GPU query failed for device {ordinal}: {e}"))?;
            let base_arch = arch_name.split(':').next().unwrap_or(&arch_name);
            let warp_size = if base_arch.starts_with("gfx9") {
                64
            } else {
                32
            };
            (arch_name, total_vram, warp_size)
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
    validate_gfx942_policy(&cli, &model_variant, backend, &gpu_arch)?;

    // Install per-arch policy so gpu_hal::alloc dispatches correctly.
    // `MemoryArchitecture` is informational (used downstream for VRAM
    // budgeting on APUs); `BufferPolicy` maps caller-side `BufferKind`
    // intent to the actual `AllocStrategy`. Persistent always uses the
    // classic device allocator (GPU-cacheable); Scratch may opt into
    // host-mapped on arches where that's a win — today only gfx1150.
    // Must be set before any GpuBuffer::alloc, which starts during weight
    // loading below.
    let arch_profile = registry::ArchProfile::for_arch(&entry.arch);
    gpu_hal::set_memory_architecture(arch_profile.memory);
    gpu_hal::set_buffer_policy(arch_profile.buffer_policy);
    fn strategy_label(s: gpu_hal::AllocStrategy) -> &'static str {
        match s {
            gpu_hal::AllocStrategy::Default => "hipMalloc / cudaMalloc / metal",
            gpu_hal::AllocStrategy::HostMapped => "hipHostMalloc(MAPPED) + GetDevicePointer",
        }
    }
    eprintln!(
        "[gpu] memory={:?}, buffer_policy: persistent={} scratch={}",
        arch_profile.memory,
        strategy_label(arch_profile.buffer_policy.persistent),
        strategy_label(arch_profile.buffer_policy.scratch),
    );

    // Run before family dispatch so DFlash flags are not silently ignored by
    // non-Qwen branches.
    validate_dflash_flags(&cli, &model_variant)?;
    validate_specprefill_flags(&cli, &model_variant, backend)?;

    match model_variant.family() {
        ModelFamily::Gemma4 => return run_gemma4(&cli, &model_variant, entry, ordinal, total_vram),
        ModelFamily::Phi4 => {
            return phi4_engine::run_phi4(&cli, &model_variant, entry, ordinal, total_vram);
        }
        ModelFamily::Llama31 => {
            return llama31_engine::run_llama31(&cli, &model_variant, entry, ordinal, total_vram);
        }
        ModelFamily::Qwen36Moe => {
            return qwen36_moe_cli::run(&cli, entry, total_vram);
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
            if should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir)) {
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

    // --specprefill-* dispatch. Validation already ran in
    // validate_specprefill_flags; the presence of --specprefill-draft-dir
    // is the gate that switches to the SpecPrefill orchestrator.
    if cli.specprefill_draft_dir.is_some() {
        return specprefill_engine::run_specprefill(
            &cli,
            &model_variant,
            entry,
            ordinal,
            total_vram,
        );
    }

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => p,
        FamilyParams::Qwen36Moe(_) => unreachable!("qwen3.6-moe handled above"),
        FamilyParams::Gemma4(_) => unreachable!("gemma4 handled above"),
        FamilyParams::Phi4(_) => unreachable!("phi4 handled above"),
        FamilyParams::Llama31(_) => unreachable!("llama3.1 handled above"),
    };
    let host_lm_head_rescorer = HostLmHeadRescorer::from_model_dir(&cli.model_dir)?;

    // Install the per-(arch, model) HIP launch preset (grid size +
    // cooperative flag) if one is registered. User env vars still override
    // inside the bridge. Always called — `(0, false)` clears any stale
    // preset from a prior run, so switching models doesn't inherit the
    // previous one's grid. No-op on CUDA builds.
    {
        let preset = registry::qwen35_4b_launch_preset(&entry.arch, &entry.model);
        let (blocks, coop) = preset.unwrap_or((0, false));
        kernel_ffi::set_qwen35_4b_launch_preset(blocks, coop);
        if let Some((blocks, coop)) = preset {
            eprintln!("[preset] qwen35_4b launch: blocks={blocks} cooperative={coop}");
        }
    }

    let Qwen35Policy {
        trace_kv_cache_enabled,
    } = validate_qwen35_startup(
        &cli,
        &model_variant,
        params,
        backend,
        &entry.arch,
        q4km_like,
    )?;

    // If --model-dir is pristine (no config.json), fetch a bake first so the
    // downloader can populate HF metadata before we try to read it.
    let bootstrap_downloaded = ensure_hf_metadata_present(&cli, &model_variant)?;

    let Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
    } = load_qwen35_startup(&cli)?;
    check_qwen35_vram(
        &cli,
        &text_config,
        &entry.vram,
        context_tokens,
        params.kv_chunk_size,
        total_vram,
    )?;

    let gpu_validate_enabled = cli.gpu_validate && cli.batch_size == 1;
    if cli.gpu_validate && cli.batch_size > 1 {
        eprintln!("[gpu-validate] GPU oracle disabled for batch_size > 1");
    }
    let Qwen35EngineSetup {
        mut engine,
        use_4b_kernel,
        cuda_08b_hero_enabled,
        allow_host_lm_head_rescore,
    } = load_qwen35_engine(
        &cli,
        &model_variant,
        &text_config,
        params,
        backend,
        gpu_arch,
        ordinal,
        bootstrap_downloaded,
        q4km_like,
        context_tokens,
    )?;

    // When using FP8 runtime weights, tell the oracle to use the same FP8 weights
    // (dequanted to BF16) so we compare apples-to-apples.
    let fp8_oracle_dir = if cli.fp8_runtime {
        Some(cli.model_dir.clone())
    } else {
        None
    };
    let oracle_device = resolve_oracle_device(&cli.oracle_device, backend, ordinal);

    // Run prefill (native GPU or oracle)
    let qwen_oracle_model_id =
        resolve_qwen_oracle_model_id(cli.model_id.as_deref(), &cli.model_dir, &model_variant);
    let Qwen35Prefill {
        logits: prefill_logits,
        native_trace: native_prefill_trace,
        mut next_token,
    } = run_qwen35_prefill(
        &cli,
        &mut engine,
        &prompt_ids,
        &qwen_oracle_model_id,
        &oracle_device,
        fp8_oracle_dir.as_deref(),
        host_lm_head_rescorer.as_ref(),
        allow_host_lm_head_rescore,
    )?;

    if cli.dump_last_logits {
        use std::io::Write as _;
        print!("\nLAST_LOGITS: ");
        for (i, x) in prefill_logits.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    report_qwen35_virtual_kv_after_prefill(&mut engine)?;

    // Optionally run oracle for validation
    let oracle_output = run_qwen35_oracle_validation(
        &cli,
        &engine,
        &text_config,
        &prompt_ids,
        &qwen_oracle_model_id,
        &oracle_device,
        fp8_oracle_dir.as_deref(),
        &prefill_logits,
        native_prefill_trace.as_ref(),
        next_token,
    )?;

    if let (Some(trace_layer), Some(output)) =
        (cli.trace_oracle_prefill_layer, oracle_output.as_ref())
    {
        let oracle_script = qwen35_oracle_script_path();
        trace_qwen35_oracle_prefill_layer(
            &mut engine,
            trace_layer,
            &prompt_ids,
            &oracle_script,
            &qwen_oracle_model_id,
            &cli.oracle_dtype,
            &oracle_device,
            fp8_oracle_dir.as_deref(),
            output,
        )?;
    }

    // Replicate prefill state to batch items if batch_size > 1
    if cli.batch_size > 1 {
        eprintln!(
            "[batch] replicating prefill state to {} sequences",
            cli.batch_size
        );
        engine.replicate_state_to_batch()?;
    }

    if gpu_validate_enabled {
        eprintln!(
            "[gpu-validate] replaying full token history through GPU prefill for reference..."
        );
    }
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
        && use_4b_kernel
        && !cli.kv_fp8
        && !cli.force_kernel_decode
        && !cli.force_component_decode;
    // Metal v2 wires per-op incremental decode through the standard
    // `engine.decode_step` path; only the legacy 4B / qwen3.5-2b CUDA replay
    // gates remain.
    let metal_v2_incremental = backend == Backend::Metal && cli.batch_size == 1;
    let replay_decode_enabled = cli.batch_size == 1
        && !cli.force_kernel_decode
        && !cli.force_component_decode
        && !cli.kv_fp8
        && use_4b_kernel
        && (cli.force_replay_decode || cuda_qwen2b_replay_default);
    let replay_kv_fp8_enabled =
        use_4b_kernel && cli.kv_fp8 && cli.batch_size == 1 && !cli.force_kernel_decode;
    let component_single_decode_enabled =
        cli.batch_size == 1 && use_4b_kernel && cli.force_component_decode;
    // Use the batched persistent megakernel path (decode_step_batch with b=1)
    // for 4B single-seq decode by default — measured ~300 ms/tok on gfx1150
    // vs ~500 ms/tok for decode_step() and ~2500 ms/tok for the legacy
    // replay path. Opt-out via --force-replay-decode (legacy parity) or
    // --force-component-decode (primitive-chain correctness).
    let kernel_single_decode_enabled = cli.batch_size == 1
        && use_4b_kernel
        && !cli.force_replay_decode
        && !cli.force_component_decode;
    let cuda_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_CUDA_FAST_GREEDY").is_some();
    let cuda_fast_greedy_enabled = backend == Backend::Cuda
        && !use_4b_kernel
        && cli.batch_size == 1
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && oracle_output.is_none()
        && !cuda_08b_hero_enabled
        && !cuda_fast_greedy_disabled;
    // Metal fast-greedy: same trigger conditions as CUDA's fast-greedy. Uses the
    // fused lm_head + argmax kernel and returns just the sampled token, skipping
    // the per-token 250k-element BF16 D2H + host argmax loop. Disable via env
    // var for bring-up/bisect.
    let metal_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_METAL_FAST_GREEDY").is_some();
    let metal_fast_greedy_enabled = backend == Backend::Metal
        && metal_v2_incremental
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && oracle_output.is_none()
        && !metal_fast_greedy_disabled;
    if metal_v2_incremental {
        if metal_fast_greedy_enabled {
            eprintln!("[decode] Metal v2 incremental decode (fast-greedy: fused argmax)");
        } else {
            eprintln!("[decode] Metal v2 incremental decode");
        }
    }
    if replay_decode_enabled {
        if cuda_qwen2b_replay_default {
            eprintln!(
                "[decode] single-sequence CUDA qwen3.5-2b uses replayed GPU prefill for correctness"
            );
        } else {
            eprintln!("[decode] single-sequence 4B uses replayed GPU prefill for correctness");
        }
    } else if replay_kv_fp8_enabled && cli.batch_size == 1 {
        eprintln!("[decode] single-sequence KV-FP8 uses replayed GPU prefill for correctness");
    } else if cli.batch_size > 1 && use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] batched KV-FP8 uses the persistent kernel path");
    } else if component_single_decode_enabled {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the component decode path");
    } else if cli.batch_size == 1 && use_4b_kernel && cli.force_kernel_decode {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the kernel decode path");
    } else if cli.batch_size == 1 && use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] WARNING: single-sequence KV-FP8 uses the b=1 kernel path");
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
                    use_4b_kernel,
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
                    use_4b_kernel,
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
                    use_4b_kernel,
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
                    use_4b_kernel,
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
                    use_4b_kernel,
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
                let (token, timings) =
                    engine.decode_step_cuda_fast_greedy(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if metal_fast_greedy_enabled {
                let token = engine.decode_step_metal_fast_greedy(next_token, seqlen_offset)?;
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
                    use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
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
                        use_4b_kernel,
                    )?;
                    engine.rebuild_prefill_state(&trace_token_ids, false)?;
                }
                if cli.emit_stage_timings {
                    let (logits, timings) = engine
                        .decode_step_4b_single_kernel_with_timings(next_token, seqlen_offset)?;
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    can_rescore_with_normed = true;
                    logits
                } else {
                    can_rescore_with_normed = true;
                    engine
                        .decode_step_batch(&[next_token], seqlen_offset)?
                        .remove(0)
                }
            } else if cli.emit_stage_timings {
                let (logits, timings) =
                    engine.decode_step_with_timings(next_token, seqlen_offset)?;
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
                    if delta > max_delta {
                        max_delta = delta;
                    }
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
                    use_4b_kernel,
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
                    use_4b_kernel,
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
    let generated_text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("detokenize generated suffix: {e}"))?;

    println!("{text}");
    if cli.emit_generated_json {
        println!(
            "[generated_json] {}",
            serde_json::to_string(&generated_text)?
        );
    }
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
        &f32_to_bf16_bytes(&native_gated_f32),
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
    let native_comp_layer = engine
        .component_trace_full_layer_from_current_hidden_with_seqlen(trace_layer, seqlen_offset)?;

    engine.rebuild_prefill_state(prefix_ids, true)?;
    let native_hidden_out = engine.decode_step_batch_trace_hidden_after_layers(
        trace_tokens,
        seqlen_offset,
        trace_layer + 1,
        0,
    )?;
    let native_partial_conv = engine
        .state_for_batch(0)
        .layers
        .get(trace_layer)
        .ok_or_else(|| anyhow::anyhow!("missing native partial layer {trace_layer}"))?
        .conv_state
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!("missing native partial conv state for layer {trace_layer}")
        })?
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
        let step_b_last = debug.get(conv_state_len - 1).copied().unwrap_or_default();
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
        if (native[idx] - step_b_last).abs() == 0.0
            || trace_layer + 1 >= text_config.num_hidden_layers
        {
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
                    .ok_or_else(|| {
                        anyhow::anyhow!("missing binary-search conv state for layer {trace_layer}")
                    })?
                    .to_host_bytes()
                    .map_err(|e| {
                        anyhow::anyhow!("binary-search conv D2H layer {trace_layer}: {e}")
                    })?;
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
                .ok_or_else(|| {
                    anyhow::anyhow!("missing pre-window conv state for layer {trace_layer}")
                })?
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
                .ok_or_else(|| {
                    anyhow::anyhow!("missing post-window conv state for layer {trace_layer}")
                })?
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
            mismatches.push((
                c,
                native[idx],
                expected_conv_tail[idx],
                (native[idx] - expected_conv_tail[idx]).abs(),
            ));
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
        "[trace-persistent-linear] layer={trace_layer} hidden_delta={hidden_delta:.6} comp_vs_replay_conv={comp_vs_replay_conv:.6} comp_vs_replay_recurrent={comp_vs_replay_recurrent:.6} comp_linear_hidden_vs_replay={comp_vs_replay_hidden:.6} native_vs_comp_qkv_proj={native_vs_comp_qkv_proj:.6} native_vs_replay_qkv_proj={native_vs_replay_qkv_proj:.6} native_vs_comp_z_proj={native_vs_comp_z_proj:.6} native_vs_replay_z_proj={native_vs_replay_z_proj:.6} native_vs_comp_b_proj={native_vs_comp_b_proj:.6} native_vs_replay_b_proj={native_vs_replay_b_proj:.6} native_vs_comp_a_proj={native_vs_comp_a_proj:.6} native_vs_replay_a_proj={native_vs_replay_a_proj:.6} native_vs_comp_conv={native_vs_comp_conv:.6} native_vs_comp_recurrent={native_vs_comp_recurrent:.6} native_conv_vs_expected_tail={native_conv_vs_expected_tail:.6} comp_conv_vs_expected_tail={comp_conv_vs_expected_tail:.6} replay_conv_vs_expected_tail={replay_conv_vs_expected_tail:.6} native_conv_tap_deltas=[{native_conv_tap_deltas}] comp_conv_tap_deltas=[{comp_conv_tap_deltas}] max_append_mismatch=({max_append_mismatch}) step_b_debug=({step_b_debug}) first_later_clobber=({first_later_clobber}) pointer_debug=({pointer_debug}) isolated_tail_windows=[{isolated_tail_windows}] append_mismatch_samples=[{append_mismatch_samples}] native_vs_comp_token_mixer={native_vs_comp_token_mixer:.6} native_vs_replay_token_mixer={native_vs_replay_token_mixer:.6} native_vs_comp_post_norm={native_vs_comp_post_norm:.6} native_vs_replay_post_norm={native_vs_replay_post_norm:.6} native_vs_comp_gated={native_vs_comp_gated:.6} native_vs_replay_gated={native_vs_replay_gated:.6} native_vs_comp_swiglu={native_vs_comp_swiglu:.6} native_vs_replay_swiglu={native_vs_replay_swiglu:.6} native_vs_comp_mlp_down={native_vs_comp_mlp_down:.6} native_vs_replay_mlp_down={native_vs_replay_mlp_down:.6} native_vs_comp_proj_residual={native_vs_comp_proj_residual:.6} comp_layer_hidden_vs_replay={comp_layer_vs_replay_hidden:.6} native_vs_comp_layer_hidden={native_vs_comp_layer_hidden:.6} native_vs_replay_hidden={native_vs_replay_hidden:.6} sample_qkv_native=[{sample_qkv_native}] sample_qkv_comp=[{sample_qkv_comp}] sample_z_native=[{sample_z_native}] sample_z_comp=[{sample_z_comp}]"
    );
    Ok(())
}
