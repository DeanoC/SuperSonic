use anyhow::Result;

use crate::registry::{Backend, GpuArch, ModelVariant};
use crate::Cli;

/// The direct Qwen3.8 GQH path does not select a bake profile.
///
/// This helper remains temporarily for the pre-reduction binary call site;
/// startup cleanup removes the obsolete bake dispatch that consumes it.
pub(crate) fn q4km_like(_cli: &Cli) -> bool {
    false
}

/// Validate the product-level generation invariants.
///
/// The public product is intentionally single-sequence and greedy. MTP is an
/// optional generation optimization, so it shares the same constraints as
/// ordinary generation.
pub(crate) fn validate_global_flags(
    cli: &Cli,
    _model_variant: &ModelVariant,
    _backend: Backend,
) -> Result<()> {
    if cli.batch_size != 1 {
        anyhow::bail!("Qwen3.8 inference supports only single-sequence generation");
    }
    if cli.temperature != 0.0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--temperature 0)");
    }
    if cli.top_k != 0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--top-k 0)");
    }
    if cli.top_p != 1.0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--top-p 1)");
    }
    Ok(())
}

/// Legacy dispatch hooks are intentionally inert: the corresponding options
/// are no longer parseable by the public CLI. They remain as narrow call-site
/// boundaries until the startup-collapse task removes the old dispatch tree.
pub(crate) fn validate_dflash_flags(_cli: &Cli, _model_variant: &ModelVariant) -> Result<()> {
    Ok(())
}

pub(crate) fn validate_specprefill_flags(
    _cli: &Cli,
    _model_variant: &ModelVariant,
    _backend: Backend,
) -> Result<()> {
    Ok(())
}

pub(crate) fn validate_gfx942_policy(
    _cli: &Cli,
    _model_variant: &ModelVariant,
    _backend: Backend,
    _gpu_arch: &GpuArch,
) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::validate_global_flags;
    use crate::registry::{Backend, ModelVariant};
    use crate::Cli;

    fn cli(extra: &[&str]) -> Cli {
        let mut args = vec![
            "supersonic",
            "--model",
            "qwen3.8-27b",
            "--model-dir",
            "/tmp/model",
            "--gguf-file",
            "/tmp/model.gqh.gguf",
            "--prompt",
            "hello",
        ];
        args.extend_from_slice(extra);
        Cli::parse_from(args)
    }

    #[test]
    fn accepts_default_greedy_single_sequence() {
        validate_global_flags(&cli(&[]), &ModelVariant::Qwen3_8_27B, Backend::Hip).unwrap();
    }

    #[test]
    fn rejects_sampling_temperature() {
        let error = validate_global_flags(
            &cli(&["--temperature", "0.7"]),
            &ModelVariant::Qwen3_8_27B,
            Backend::Hip,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("greedy"), "{error}");
    }
}
