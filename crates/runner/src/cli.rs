use std::path::PathBuf;

use clap::Parser;

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

    /// Emit a native decode allocation/copy/kernel profile.
    #[arg(long)]
    pub profile_decode: bool,

    /// Write the native decode profile as JSON to this path.
    #[arg(long)]
    pub profile_decode_json: Option<PathBuf>,

    /// Enable Qwen3.8 NextN/MTP speculative generation.
    #[arg(long)]
    pub speculative_decode: bool,

    /// Process the prompt in chunks (0 means no chunking).
    #[arg(long, default_value = "0")]
    pub prefill_chunk_size: usize,

    /// Emit generated tokens as a JSON string.
    #[arg(long)]
    pub emit_generated_json: bool,
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
