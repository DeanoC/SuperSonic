use std::path::PathBuf;

use clap::Parser;

use crate::certified_kv;

#[derive(Parser)]
#[command(name = "supersonic", about = "SuperSonic — optimized LLM inference")]
pub(crate) struct Cli {
    /// Model variant (e.g. "qwen3.5-0.8b")
    #[arg(long, default_value = "qwen3.5-0.8b")]
    pub(crate) model: String,

    /// Path to a HuggingFace model directory or self-contained FLM container.
    #[arg(long)]
    pub(crate) model_dir: PathBuf,

    /// Compatibility override for FLM weights. Prefer --model-dir model.flm.
    #[arg(long)]
    pub(crate) flm_file: Option<PathBuf>,

    /// Verify BLAKE3 FLM tensor payload hashes while loading FLM weights.
    #[arg(long)]
    pub(crate) verify_flm_hashes: bool,

    /// Text prompt (will be tokenized). Required unless --dry-run is set.
    #[arg(long, required_unless_present = "dry_run", default_value = "")]
    pub(crate) prompt: String,

    /// Do not add tokenizer special tokens when encoding --prompt.
    #[arg(long)]
    pub(crate) prompt_no_special_tokens: bool,

    /// Maximum tokens to generate
    #[arg(long, default_value = "8")]
    pub(crate) max_new_tokens: usize,

    /// Continue until --max-new-tokens even if the model emits EOS.
    #[arg(long)]
    pub(crate) ignore_eos: bool,

    /// Sampling temperature. 0 = greedy argmax (default; reproducible).
    /// Typical sampled values: 0.7–1.0.
    #[arg(long, default_value = "0.0")]
    pub(crate) temperature: f32,

    /// Top-K filter: keep only the K highest-probability tokens before
    /// sampling. 0 = no cap (full vocab considered).
    #[arg(long, default_value = "0")]
    pub(crate) top_k: usize,

    /// Top-P (nucleus) filter: keep the smallest set of tokens whose
    /// cumulative probability ≥ top_p. 1.0 = no truncation.
    #[arg(long, default_value = "1.0")]
    pub(crate) top_p: f32,

    /// RNG seed for non-greedy sampling. Same seed + same prompt + same
    /// model = bit-identical generation.
    #[arg(long, default_value = "42")]
    pub(crate) sampling_seed: u64,

    /// Maximum context size in tokens (prompt + generated). Used for VRAM estimation.
    /// Defaults to prompt length + max_new_tokens if not specified.
    #[arg(long)]
    pub(crate) context_size: Option<usize>,

    /// Compute backend (`auto`, `hip`, or `cuda`)
    #[arg(long, default_value = "auto")]
    pub(crate) backend: String,

    /// Device ordinal on the selected backend
    #[arg(long, default_value = "0")]
    pub(crate) device: usize,

    /// Emit aggregated native decode stage timings at the end of the run.
    #[arg(long)]
    pub(crate) emit_stage_timings: bool,

    /// Emit a HAL-level allocation/copy/memset profile for the native prefill.
    #[arg(long)]
    pub(crate) profile_prefill: bool,

    /// Write the native prefill profile as JSON to this path.
    #[arg(long)]
    pub(crate) profile_prefill_json: Option<PathBuf>,

    /// Emit coarse runner progress heartbeats while long decode phases are active.
    /// Disabled by default; useful for long-context profiling where prefill can be quiet.
    #[arg(long, default_value = "0.0")]
    pub(crate) progress_heartbeat_seconds: f64,

    /// Enable Qwen3.6-MoE self-speculative decode.
    ///
    /// When set, the engine loads the multi-token-prediction (MTP) head
    /// from the bake and runs draft generation plus base-model verification.
    /// When unset (default), MTP weights are not loaded and self-spec decode
    /// is unavailable, leaving the extra VRAM for KV cache and scratch.
    ///
    /// Currently HIP/qwen3.6-MoE only; ignored for other model families.
    #[arg(long)]
    pub(crate) speculative_decode: bool,

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
    pub(crate) batched_spec_verify: bool,

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
    pub(crate) persistent_decode: bool,

    /// Opt out of the persistent-decode megakernel and run the
    /// legacy chained per-step launcher (~80 step launches per
    /// token, ~2.5-3 ms/token slower than the persistent path on
    /// gfx1100). Use for A/B perf comparison or to bisect a
    /// suspected megakernel-side regression.
    #[arg(long)]
    pub(crate) no_persistent_decode: bool,

    /// Emit the generated suffix as a JSON string for benchmark harnesses.
    #[arg(long)]
    pub(crate) emit_generated_json: bool,

    /// Score the prompt with teacher forcing instead of generating new tokens.
    /// Supported for Llama 3.1 (CUDA) and Qwen3.5 (HIP/CUDA/Metal).
    /// Other model families bail with a "not yet implemented" message.
    #[arg(long)]
    pub(crate) teacher_forced: bool,

    /// For certified teacher-forced scoring, use dense decode through this
    /// prefix length, skip the boundary target, then score the suffix with the
    /// certified path. This matches the DotCache PG-19 protocol.
    #[arg(long, hide = true)]
    pub(crate) teacher_forced_dense_prefix_len: Option<usize>,

    /// Debug-only: force the legacy one-token-prefill + decode-step teacher-forced scorer.
    #[arg(long, hide = true)]
    pub(crate) teacher_forced_decode_step: bool,

    /// Run PyTorch oracle and compare logits
    #[arg(long)]
    pub(crate) validate: bool,

    /// Oracle dtype (bf16 or fp32)
    #[arg(long, default_value = "bf16")]
    pub(crate) oracle_dtype: String,

    /// Oracle device (`auto`, `cpu`, `cuda:0`, etc.)
    #[arg(long, default_value = "auto")]
    pub(crate) oracle_device: String,

    /// HuggingFace model ID (for oracle; defaults based on model variant)
    #[arg(long)]
    pub(crate) model_id: Option<String>,

    /// Skip baked format and load directly from safetensors (for debugging)
    #[arg(long)]
    pub(crate) no_bake: bool,

    /// Use oracle (Python) for prefill instead of native GPU prefill
    #[arg(long)]
    pub(crate) oracle_prefill: bool,

    /// Unified weight quantization profile. Canonical values: bf16,
    /// fp8-native, int4-gptq, int4-awq, int4-autoround, int4-hqq, higgs4,
    /// quip-e8, qtip-trellis2. Legacy flags remain accepted as aliases.
    #[arg(long)]
    pub(crate) weight_quant: Option<String>,

    /// Keep FP8 weights in native format on GPU for runtime dequantization.
    /// Halves weight VRAM (~8.8→4.8 GiB for 4B). Requires FP8 model weights.
    #[arg(long)]
    pub(crate) fp8_runtime: bool,

    /// Quantize weights to INT4 (4-bit) with group quantization for ~4x weight compression.
    /// Bakes BF16→INT4 on first run. Targets ~200 ms/tok on bandwidth-limited GPUs.
    #[arg(long)]
    pub(crate) int4: bool,

    /// Use a GGUF-like Q4KM bake in SuperSonic's native low-bit runtime layout.
    /// Runtime executes the translated bake, not GGML blocks directly.
    #[arg(long)]
    pub(crate) q4km: bool,

    /// Use a Q4KM-sourced GPTQ bake in SuperSonic's native INT4 runtime layout.
    #[arg(long)]
    pub(crate) q4km_gptq: bool,

    /// Optional GGUF source file to translate into a native q4km bake.
    #[arg(long)]
    pub(crate) gguf_file: Option<PathBuf>,

    /// Load an INT8 bake produced from BitsAndBytes `load_in_8bit=True`.
    /// Currently only supported for `llama3.1-8b` on CUDA.
    #[arg(long)]
    pub(crate) int8: bool,

    /// Process prompt in chunks of this size (0 = no chunking, process entire prompt at once).
    /// Reduces activation VRAM for long prompts. Typical values: 128, 256, 512.
    #[arg(long, default_value = "0")]
    pub(crate) prefill_chunk_size: usize,

    /// Store KV cache in FP8 E4M3 with dynamic per-head scaling.
    /// Halves KV cache VRAM, nearly doubling max context length.
    #[arg(long)]
    pub(crate) kv_fp8: bool,

    /// Enable the staged certified tiered KV path for Llama 3.1 CUDA INT8.
    #[arg(long)]
    pub(crate) certified_kv: bool,

    /// Hidden research preset for certified KV defaults.
    #[arg(long, default_value = "legacy", hide = true)]
    pub(crate) certified_kv_preset: certified_kv::CertifiedKvPreset,

    /// Optional JSONL telemetry path for certified KV stage and fallback counters.
    #[arg(long)]
    pub(crate) certified_kv_telemetry: Option<PathBuf>,

    /// Quantize real BF16 KV caches after prefill and report shadow-cache telemetry.
    #[arg(long, hide = true)]
    pub(crate) certified_kv_shadow_validate: bool,

    /// Debug-only: compare one full-attention layer using dense KV vs certified KV.
    #[arg(long, hide = true)]
    pub(crate) certified_kv_trace_layer: Option<usize>,

    /// Debug-only: sweep all full-attention layers using dense KV vs certified KV.
    #[arg(long, hide = true)]
    pub(crate) certified_kv_trace_all: bool,

    /// Certified KV cache block size in tokens.
    #[arg(long, default_value = "16", hide = true)]
    pub(crate) certified_kv_block_size: usize,

    /// Certified KV INT4 value quantization group size.
    #[arg(long, default_value = "16", hide = true)]
    pub(crate) certified_kv_value_group_size: usize,

    /// Debug-only: use BF16 values with INT8 keys in the certified KV trace path.
    #[arg(long, hide = true)]
    pub(crate) certified_kv_bf16_values: bool,

    /// Certified KV adaptive coverage target.
    #[arg(long, default_value = "0.995", hide = true)]
    pub(crate) certified_kv_tau_cov: f32,

    /// Certified KV minimum FP16 key blocks per head.
    #[arg(long, default_value = "2", hide = true)]
    pub(crate) certified_kv_k_min: usize,

    /// Certified KV maximum FP16 key blocks per head before Rung 1 expansion (0 = unclamped).
    #[arg(long, default_value = "128", hide = true)]
    pub(crate) certified_kv_k_max: usize,

    /// Certified KV value error promotion threshold.
    #[arg(long, default_value = "0.05", hide = true)]
    pub(crate) certified_kv_v_tol: f32,

    /// Promoted BF16 value blocks cached per layer/KV head (0 disables cache).
    #[arg(long, default_value = "128", hide = true)]
    pub(crate) certified_kv_value_cache_blocks: usize,

    /// Certified KV ranking-consistency depth.
    #[arg(long, default_value = "1", hide = true)]
    pub(crate) certified_kv_ranking_r: usize,

    /// Certified KV Rung 1 tail-mass threshold.
    #[arg(long, default_value = "0.005", hide = true)]
    pub(crate) certified_kv_rung1_threshold: f32,

    /// Certified KV Rung 1 K expansion multiplier.
    #[arg(long, default_value = "2.0", hide = true)]
    pub(crate) certified_kv_rung1_multiplier: f32,

    /// FP16 key scratch cache capacity in blocks per layer/KV head.
    #[arg(long, default_value = "256", hide = true)]
    pub(crate) certified_kv_key_cache_blocks: usize,

    /// Certified KV guard exponent applied to tail-mass certificates.
    #[arg(long, default_value = "3.0", hide = true)]
    pub(crate) certified_kv_delta_guard_factor: f32,

    /// Fraction of tail blocks to score-check opportunistically.
    #[arg(long, default_value = "0.01", hide = true)]
    pub(crate) certified_kv_score_exploration_rate: f32,

    /// Debug-only: allow certified KV to continue when the tail certificate misses threshold.
    #[arg(long, hide = true)]
    pub(crate) certified_kv_allow_uncertified_tail: bool,

    /// Certified KV score-consistency guard epsilon.
    #[arg(long, default_value = "0.0001", hide = true)]
    pub(crate) certified_kv_eps_guard: f32,

    /// Validate decode against a replayed GPU prefill reference.
    /// Replays the full token history through native GPU prefill on each step and
    /// compares the resulting last-token logits against decode. Slower than decode,
    /// but avoids the stale incremental component-oracle path.
    #[arg(long)]
    pub(crate) gpu_validate: bool,

    /// Dump and compare per-layer prefill hidden states against the oracle.
    /// Debug-only path intended to localize long-context divergence.
    #[arg(long)]
    pub(crate) trace_prefill_layers: bool,

    /// Debug-only: run one prompt-prefill layer from the oracle's exact prefix
    /// state and compare our component layer outputs against the oracle.
    #[arg(long, hide = true)]
    pub(crate) trace_oracle_prefill_layer: Option<usize>,

    /// Batch size for decode (number of sequences decoded in parallel).
    /// Default 1. Supported on Qwen3.5 through the 4B-capable persistent
    /// kernel and Gemma 4 BF16 + INT4 via per-family batched megakernels.
    #[arg(long, default_value = "1")]
    pub(crate) batch_size: usize,

    /// Debug-only: force single-sequence 4B decode to use the actual kernel path
    /// instead of the replayed prefill correctness path.
    /// (Historical — `replayed prefill` is no longer the default; the kernel
    /// path is used by default. This flag is kept as a no-op for callers
    /// that still pass it explicitly.)
    #[arg(long, hide = true)]
    pub(crate) force_kernel_decode: bool,

    /// Debug-only: force single-sequence 4B decode to use the component decode path
    /// instead of replayed prefill or the persistent kernel.
    #[arg(long, hide = true)]
    pub(crate) force_component_decode: bool,

    /// Debug-only: restore the legacy "replay prefill each decode step" path
    /// that was the default before 2026-04-20. Scales O(N) per token with
    /// context length and was ~7x slower than the persistent megakernel path,
    /// so retained only for parity validation. Mutually exclusive with
    /// --force-kernel-decode / --force-component-decode.
    #[arg(long, hide = true)]
    pub(crate) force_replay_decode: bool,

    /// Legacy compatibility switch for older CUDA KV-FP8 bring-up commands.
    /// No longer required now that the validated 4B sm86 lane is public.
    #[arg(long, hide = true)]
    pub(crate) allow_unstable_cuda_kv_fp8: bool,

    /// Debug-only: compare decode-appended KV-FP8 cache contents against a replayed
    /// prefill KV-FP8 reference after each decode step.
    #[arg(long, hide = true)]
    pub(crate) trace_kv_fp8_cache: bool,

    /// Debug-only: compare decode-appended KV cache contents against a replayed
    /// prefill reference after each decode step. Works for BF16 and FP8 KV.
    #[arg(long, hide = true)]
    pub(crate) trace_kv_cache: bool,

    /// When set, after the final decode step the runner prints
    /// `LAST_LOGITS: <comma-separated f32>` to stdout exactly once.
    /// Used by integration parity tests to compare logits across
    /// runs (e.g., --kv-fp8 vs BF16 KV).
    #[arg(long, default_value_t = false)]
    pub(crate) dump_last_logits: bool,

    /// Debug-only: on the component decode path, capture the BF16 hidden state
    /// immediately before this layer and compare it to replayed prefill.
    #[arg(long, hide = true)]
    pub(crate) trace_component_input_layer: Option<usize>,

    /// Debug-only: on the component decode path, compare one layer's stage outputs
    /// against replayed prefill (token mixer output, post-attn norm, mlp out, final hidden).
    #[arg(long, hide = true)]
    pub(crate) trace_component_layer: Option<usize>,

    /// Debug-only: on the component decode path, compare one linear-attention layer's
    /// internal tensors (qkv, z, attn, gated, proj_out) against replayed prefill.
    #[arg(long, hide = true)]
    pub(crate) trace_component_linear_layer: Option<usize>,

    /// Debug-only: compare one linear-attention layer's conv/recurrent state against
    /// replayed prefill before the decode step runs.
    #[arg(long, hide = true)]
    pub(crate) trace_component_linear_state_layer: Option<usize>,

    /// Debug-only: run the real persistent 4B kernel for the first N layers and compare
    /// the resulting hidden state against replayed prefill's input to that layer.
    #[arg(long, hide = true)]
    pub(crate) trace_persistent_input_layer: Option<usize>,

    /// Debug-only: run the real persistent 4B kernel through one selected linear layer
    /// and compare that layer's conv/recurrent state against replayed prefill.
    #[arg(long, hide = true)]
    pub(crate) trace_persistent_linear_state_layer: Option<usize>,

    /// Debug-only: compare one full-attention layer's K/V production on the real
    /// persistent path against the component full-attention path using the same
    /// hidden-state input.
    #[arg(long, hide = true)]
    pub(crate) trace_persistent_full_attn_layer: Option<usize>,

    /// Debug-only: compare one linear-attention layer's production on the real
    /// persistent path against the component linear path using the same
    /// hidden-state input and pre-step state.
    #[arg(long, hide = true)]
    pub(crate) trace_persistent_linear_layer: Option<usize>,

    /// Run on an arch without a registry entry by reusing another arch's kernel.
    /// Pass the arch name whose kernel you want to reuse (e.g. "gfx1150"). Emits
    /// a loud warning — correctness is not guaranteed. Intended for archs that
    /// are binary-compatible (same wavefront size, similar CU/LDS) but haven't
    /// been explicitly tuned.
    #[arg(long)]
    pub(crate) allow_untested_gpu: Option<String>,

    /// Enumerate the model checkpoint, compute analytic + on-disk VRAM
    /// accounting, print the decode budget report, and exit.
    #[arg(long)]
    pub(crate) dry_run: bool,

    /// Disable downloading pre-baked weights from GitHub releases when the
    /// local bake is missing. Prints the manual bake guidance instead.
    #[arg(long)]
    pub(crate) no_download: bool,

    /// Force downloading a pre-baked package even if a valid local bake exists.
    #[arg(long)]
    pub(crate) download_bake: bool,

    /// Override the GitHub release/tag used for bake downloads.
    #[arg(long)]
    pub(crate) bake_release: Option<String>,

    /// Enable DFlash speculative decoding. Requires `--model qwen3.5-9b`
    /// or `qwen3.6-27b`, a low-bit target bake, and `--dflash-draft-dir`.
    /// Draft is the DFlash checkpoint shared with target embeddings/head via Arc.
    #[arg(long)]
    pub(crate) dflash: bool,

    /// Path to the DFlash draft checkpoint config directory (e.g.
    /// `z-lab/Qwen3.5-9B-DFlash` extracted locally). Normally contains
    /// `config.json` and `model.safetensors`; GGUF draft weights can be
    /// selected by setting `SUPERSONIC_DFLASH_DRAFT_GGUF`.
    #[arg(long)]
    pub(crate) dflash_draft_dir: Option<PathBuf>,

    /// Override the DFlash block size (draft candidates per round). Must
    /// be 1..=draft_config.block_size. Default is 3 — the fused verify
    /// megakernel on Qwen3.5-9B is LDS-bound and caps B at 3 on gfx1150
    /// (block_size + B*hidden + fp8_lut must fit in 64 KiB shared mem).
    /// Launches with B >= 4 fail fast with a shared-memory diagnostic.
    #[arg(long)]
    pub(crate) dflash_block: Option<usize>,

    /// Override the DFlash tap layers as a comma-separated list of
    /// target-model layer indices (e.g. `1,8,15,22,29`). Must match the
    /// count implied by the draft's `fc.in_features`. Defaults to the
    /// checkpoint's `dflash_config.target_layer_ids`.
    #[arg(long)]
    pub(crate) dflash_tap_layers: Option<String>,

    /// Path to the SpecPrefill (arXiv 2502.02789) draft model directory
    /// (e.g. `/mnt/data/models/Qwen3.5-0.8B`). Presence of this flag
    /// enables sparse target prefill via the speculator's importance
    /// signal. Supported for `--model qwen3.5-9b`; CUDA/HIP cosine
    /// cross-family mode is also supported for `--model qwen3.6-35b-a3b`
    /// with a Qwen3.5-0.8B drafter.
    #[arg(long)]
    pub(crate) specprefill_draft_dir: Option<PathBuf>,

    /// SpecPrefill keep ratio per chunk: fraction of tokens kept by the
    /// chunked top-K selection. Phase A2 measurements pin 0.50 as the
    /// quality-stable default on Qwen3.5-9B (cossim ≥ 0.927, argmax
    /// match). CUDA Qwen3.6 cross-family SpecPrefill defaults to 0.75
    /// because the broader logits gate shows 0.50 is a balanced/perf lane
    /// with occasional argmax drift. Range: [0.05, 1.0].
    #[arg(long)]
    pub(crate) specprefill_keep_ratio: Option<f32>,

    /// SpecPrefill chunk size for top-K selection (paper §3.4).
    /// Default applied in run_specprefill: 32.
    #[arg(long)]
    pub(crate) specprefill_chunk_size: Option<usize>,

    /// SpecPrefill 1-D average-pool window for score smoothing. Must be
    /// odd. Paper uses 5-10. Default applied in run_specprefill: 5.
    #[arg(long)]
    pub(crate) specprefill_pool_window: Option<usize>,

    /// SpecPrefill look-ahead decode steps on the draft (paper §3.3
    /// default 4). Total query rows harvested = lookahead + 1.
    /// Default applied in run_specprefill: 4.
    #[arg(long)]
    pub(crate) specprefill_lookahead: Option<usize>,

    /// SpecPrefill always-keep prefix (BOS + system) length.
    /// Default applied in run_specprefill: 4.
    #[arg(long)]
    pub(crate) specprefill_always_keep_prefix: Option<usize>,

    /// SpecPrefill always-keep suffix (final query) length.
    /// Default applied in run_specprefill: 4.
    #[arg(long)]
    pub(crate) specprefill_always_keep_suffix: Option<usize>,

    /// Free the draft weights after selection runs and before the target
    /// prefill, to claw back ~1.6 GiB on a tight 24 GiB budget.
    #[arg(long, default_value_t = false)]
    pub(crate) specprefill_unload_draft: bool,

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
    pub(crate) specprefill_algorithm: String,
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::Cli;

    #[test]
    fn help_advertises_flm_model_dir_as_main_path() {
        let mut cmd = Cli::command();
        let mut help = Vec::new();
        cmd.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("self-contained FLM container"), "{help}");
        assert!(help.contains("Prefer --model-dir model.flm"), "{help}");
        assert!(help.contains("--verify-flm-hashes"), "{help}");
    }
}
