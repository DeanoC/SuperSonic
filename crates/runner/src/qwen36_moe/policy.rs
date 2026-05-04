use anyhow::Result;
use gpu_hal::Backend;

use crate::qwen36_moe_cli::dry_run::ContextSizeSource;
use crate::registry::RegistryEntry;

pub fn resolve_context_size(cli: &crate::Cli) -> (usize, ContextSizeSource) {
    let max_new = cli.max_new_tokens.max(1);
    if let Some(ctx) = cli.context_size {
        (ctx, ContextSizeSource::Explicit)
    } else if !cli.prompt.is_empty() {
        (
            cli.prompt.chars().count() + max_new,
            ContextSizeSource::EstimatedFromPrompt,
        )
    } else {
        (max_new, ContextSizeSource::MaxNewTokensOnly)
    }
}

pub fn validate_persistent_kv_fp8_flags(cli: &crate::Cli) -> Result<()> {
    if cli.kv_fp8 && cli.no_persistent_decode {
        anyhow::bail!(
            "--kv-fp8 for Qwen3.6-35B-A3B requires the persistent megakernel; \
             remove --no-persistent-decode (persistent is on by default). The \
             back-compat step kernels stay BF16-KV."
        );
    }
    Ok(())
}

pub fn validate_decode_backend(entry: &RegistryEntry) -> Result<()> {
    if entry.backend != Backend::Hip {
        anyhow::bail!(
            "qwen3.6-35b-a3b decode kernels are HIP-only at this stage; \
             registry-selected backend was {:?}. Re-run with --backend hip, \
             or use --dry-run for analytic accounting.",
            entry.backend,
        );
    }
    Ok(())
}
