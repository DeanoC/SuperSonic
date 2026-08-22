use std::path::PathBuf;

use clap::Parser;

use crate::certified_kv;

#[derive(Debug, Parser)]
#[command(
    name = "supersonic",
    about = "SuperSonic — ROCm/HIP Qwen3.8-27B GQH inference"
)]
pub struct Cli {
    /// The only supported model variant.
    #[arg(long, required = true, value_parser = ["qwen3.8-27b"])]
    pub model: String,

    /// Directory containing config.json, tokenizer data, and the chat template.
    #[arg(long, required = true)]
    pub model_dir: PathBuf,

    /// Project-specific GQH GGUF artifact containing the model weights.
    #[arg(long, required = true)]
    pub gguf_file: Option<PathBuf>,

    /// Text prompt to tokenize. Use --chat to apply the model chat template.
    #[arg(long, default_value = "")]
    pub prompt: String,

    /// Apply the model chat template to the prompt.
    #[arg(long)]
    pub chat: bool,

    /// Do not add tokenizer special tokens when encoding the prompt.
    #[arg(long)]
    pub prompt_no_special_tokens: bool,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value = "8")]
    pub max_new_tokens: usize,

    /// Continue until --max-new-tokens even when EOS is generated.
    #[arg(long)]
    pub ignore_eos: bool,

    /// Greedy sampling temperature. The product default is 0.0.
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    /// Greedy top-k control. The product default is 0 (all tokens).
    #[arg(long, default_value = "0")]
    pub top_k: usize,

    /// Greedy top-p control. The product default is 1.0 (no truncation).
    #[arg(long, default_value = "1.0")]
    pub top_p: f32,

    /// Deterministic generation seed.
    #[arg(long, default_value = "42")]
    pub sampling_seed: u64,

    /// Maximum context size in tokens. Defaults to prompt plus generation.
    #[arg(long)]
    pub context_size: Option<usize>,

    /// AMD device ordinal.
    #[arg(long, default_value = "0")]
    pub device: usize,

    /// Emit structured decode-stage timing data.
    #[arg(long)]
    pub emit_stage_timings: bool,

    /// Emit a native prefill allocation/copy profile.
    #[arg(long)]
    pub profile_prefill: bool,

    /// Write the native prefill profile as JSON to this path.
    #[arg(long)]
    pub profile_prefill_json: Option<PathBuf>,

    /// Enable Qwen3.8 NextN/MTP speculative generation.
    #[arg(long)]
    pub speculative_decode: bool,

    /// Process the prompt in chunks (0 means no chunking).
    #[arg(long, default_value = "0")]
    pub prefill_chunk_size: usize,

    /// Emit generated tokens as a JSON string.
    #[arg(long)]
    pub emit_generated_json: bool,

    // The binary still compiles implementation modules that are removed by
    // later dependency-ordered cleanup tasks. These fields are deliberately
    // skipped by clap and are not part of the public CLI contract.
    #[arg(skip)]
    pub(crate) flm_file: Option<PathBuf>,
    #[arg(skip)]
    pub(crate) verify_flm_hashes: bool,
    #[arg(skip)]
    pub(crate) flm_virtual_transfer_backend: Option<String>,
    #[arg(skip = String::from("auto"))]
    pub(crate) backend: String,
    #[arg(skip)]
    pub(crate) progress_heartbeat_seconds: f64,
    #[arg(skip)]
    pub(crate) batched_spec_verify: bool,
    #[arg(skip)]
    pub(crate) persistent_decode: bool,
    #[arg(skip)]
    pub(crate) no_persistent_decode: bool,
    #[arg(skip)]
    pub(crate) teacher_forced: bool,
    #[arg(skip)]
    pub(crate) teacher_forced_dense_prefix_len: Option<usize>,
    #[arg(skip)]
    pub(crate) teacher_forced_decode_step: bool,
    #[arg(skip)]
    pub(crate) validate: bool,
    #[arg(skip = String::from("bf16"))]
    pub(crate) oracle_dtype: String,
    #[arg(skip = String::from("auto"))]
    pub(crate) oracle_device: String,
    #[arg(skip)]
    pub(crate) model_id: Option<String>,
    #[arg(skip)]
    pub(crate) no_bake: bool,
    #[arg(skip)]
    pub(crate) oracle_prefill: bool,
    #[arg(skip)]
    pub(crate) weight_quant: Option<String>,
    #[arg(skip)]
    pub(crate) fp8_runtime: bool,
    #[arg(skip)]
    pub(crate) int4: bool,
    #[arg(skip)]
    pub(crate) q4km: bool,
    #[arg(skip)]
    pub(crate) q4km_gptq: bool,
    #[arg(skip)]
    pub(crate) int8: bool,
    #[arg(skip)]
    pub(crate) kv_fp8: bool,
    #[arg(skip)]
    pub(crate) certified_kv: bool,
    #[arg(skip = certified_kv::CertifiedKvPreset::Legacy)]
    pub(crate) certified_kv_preset: certified_kv::CertifiedKvPreset,
    #[arg(skip)]
    pub(crate) certified_kv_telemetry: Option<PathBuf>,
    #[arg(skip)]
    pub(crate) certified_kv_shadow_validate: bool,
    #[arg(skip)]
    pub(crate) certified_kv_trace_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) certified_kv_trace_all: bool,
    #[arg(skip = 16usize)]
    pub(crate) certified_kv_block_size: usize,
    #[arg(skip = 16usize)]
    pub(crate) certified_kv_value_group_size: usize,
    #[arg(skip)]
    pub(crate) certified_kv_bf16_values: bool,
    #[arg(skip = 0.995f32)]
    pub(crate) certified_kv_tau_cov: f32,
    #[arg(skip = 2usize)]
    pub(crate) certified_kv_k_min: usize,
    #[arg(skip = 128usize)]
    pub(crate) certified_kv_k_max: usize,
    #[arg(skip = 0.05f32)]
    pub(crate) certified_kv_v_tol: f32,
    #[arg(skip = 128usize)]
    pub(crate) certified_kv_value_cache_blocks: usize,
    #[arg(skip = 1usize)]
    pub(crate) certified_kv_ranking_r: usize,
    #[arg(skip = 0.005f32)]
    pub(crate) certified_kv_rung1_threshold: f32,
    #[arg(skip = 2.0f32)]
    pub(crate) certified_kv_rung1_multiplier: f32,
    #[arg(skip = 256usize)]
    pub(crate) certified_kv_key_cache_blocks: usize,
    #[arg(skip = 3.0f32)]
    pub(crate) certified_kv_delta_guard_factor: f32,
    #[arg(skip = 0.01f32)]
    pub(crate) certified_kv_score_exploration_rate: f32,
    #[arg(skip)]
    pub(crate) certified_kv_allow_uncertified_tail: bool,
    #[arg(skip = 0.0001f32)]
    pub(crate) certified_kv_eps_guard: f32,
    #[arg(skip)]
    pub(crate) gpu_validate: bool,
    #[arg(skip)]
    pub(crate) trace_prefill_layers: bool,
    #[arg(skip)]
    pub(crate) trace_oracle_prefill_layer: Option<usize>,
    #[arg(skip = 1usize)]
    pub(crate) batch_size: usize,
    #[arg(skip)]
    pub(crate) force_kernel_decode: bool,
    #[arg(skip)]
    pub(crate) force_component_decode: bool,
    #[arg(skip)]
    pub(crate) force_replay_decode: bool,
    #[arg(skip)]
    pub(crate) allow_unstable_cuda_kv_fp8: bool,
    #[arg(skip)]
    pub(crate) trace_kv_fp8_cache: bool,
    #[arg(skip)]
    pub(crate) trace_kv_cache: bool,
    #[arg(skip)]
    pub(crate) dump_last_logits: bool,
    #[arg(skip)]
    pub(crate) trace_component_input_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_component_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_component_linear_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_component_linear_state_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_persistent_input_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_persistent_linear_state_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_persistent_full_attn_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) trace_persistent_linear_layer: Option<usize>,
    #[arg(skip)]
    pub(crate) allow_untested_gpu: Option<String>,
    #[arg(skip)]
    pub(crate) dry_run: bool,
    #[arg(skip)]
    pub(crate) no_download: bool,
    #[arg(skip)]
    pub(crate) download_bake: bool,
    #[arg(skip)]
    pub(crate) bake_release: Option<String>,
    #[arg(skip)]
    pub(crate) dflash: bool,
    #[arg(skip)]
    pub(crate) dflash_draft_dir: Option<PathBuf>,
    #[arg(skip)]
    pub(crate) dflash_block: Option<usize>,
    #[arg(skip)]
    pub(crate) dflash_tap_layers: Option<String>,
    #[arg(skip)]
    pub(crate) specprefill_draft_dir: Option<PathBuf>,
    #[arg(skip)]
    pub(crate) specprefill_keep_ratio: Option<f32>,
    #[arg(skip)]
    pub(crate) specprefill_chunk_size: Option<usize>,
    #[arg(skip)]
    pub(crate) specprefill_pool_window: Option<usize>,
    #[arg(skip)]
    pub(crate) specprefill_lookahead: Option<usize>,
    #[arg(skip)]
    pub(crate) specprefill_always_keep_prefix: Option<usize>,
    #[arg(skip)]
    pub(crate) specprefill_always_keep_suffix: Option<usize>,
    #[arg(skip)]
    pub(crate) specprefill_unload_draft: bool,
    #[arg(skip = String::from("cosine"))]
    pub(crate) specprefill_algorithm: String,
}

/// Parse the public command-line contract without reading process arguments.
pub fn parse_cli_from<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::{parse_cli_from, Cli};

    fn args() -> [&'static str; 9] {
        [
            "supersonic",
            "--model",
            "qwen3.8-27b",
            "--model-dir",
            "/models/qwen38",
            "--gguf-file",
            "/models/qwen38.gqh.gguf",
            "--prompt",
            "Hello",
        ]
    }

    #[test]
    fn help_describes_the_narrow_rocm_product() {
        let mut cmd = Cli::command();
        let mut help = Vec::new();
        cmd.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("ROCm/HIP"), "{help}");
        assert!(help.contains("Qwen3.8"), "{help}");
        assert!(help.contains("GQH"), "{help}");
        for term in [
            "backend",
            "FLM",
            "CUDA",
            "Metal",
            "Gemma",
            "Phi",
            "Llama",
            "DFlash",
            "SpecPrefill",
            "Certified",
        ] {
            assert!(!help.contains(term), "forbidden term {term:?} in {help}");
        }
    }

    #[test]
    fn parses_the_retained_contract() {
        let cli = parse_cli_from(args()).unwrap();
        assert_eq!(cli.model, "qwen3.8-27b");
        assert_eq!(cli.max_new_tokens, 8);
        assert_eq!(
            cli.gguf_file.as_deref().unwrap().to_str(),
            Some("/models/qwen38.gqh.gguf")
        );
    }
}
