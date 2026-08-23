use anyhow::Result;

use crate::Cli;

/// Load the only public weight format: the validated custom Qwen3.8 GQH
/// GGUF. Validation is intentionally performed by startup before this
/// function is called, so this call is the first operation that can allocate
/// GPU memory for model weights.
pub(crate) fn load_qwen38_weights(
    cli: &Cli,
    text_config: &qwen38::config::TextConfig,
    ordinal: usize,
) -> Result<qwen38::weights::Qwen38Weights> {
    let gguf_path = cli.gguf_file.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--gguf-file is required for the Qwen3.8 GQH startup path")
    })?;
    eprintln!(
        "[weights] loading Qwen3.8 GQH GGUF from {} (HIP fused dequant-matvec)",
        gguf_path.display()
    );
    qwen38::weights::Qwen38Weights::load_gguf(gguf_path, text_config, ordinal)
        .map_err(|e| anyhow::anyhow!("load Qwen3.8 GQH GGUF {}: {e}", gguf_path.display()))
}
