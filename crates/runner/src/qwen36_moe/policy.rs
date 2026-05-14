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
    if !matches!(entry.backend, Backend::Hip | Backend::Cuda | Backend::Metal) {
        anyhow::bail!(
            "qwen3.6-35b-a3b decode kernels are wired for HIP, CUDA sm86, and \
             the Metal chained fallback path; registry-selected backend was {:?}. \
             Re-run with --backend hip/cuda/metal, \
             or use --dry-run for analytic accounting.",
            entry.backend,
        );
    }
    Ok(())
}

pub fn validate_metal_v1_flags(cli: &crate::Cli, entry: &RegistryEntry) -> Result<()> {
    if entry.backend != Backend::Metal {
        return Ok(());
    }
    if cli.fp8_runtime {
        anyhow::bail!(
            "Qwen3.6-35B-A3B Metal v1 supports INT4 chained decode only; --fp8-runtime is not wired."
        );
    }
    if cli.kv_fp8 {
        anyhow::bail!("Qwen3.6-35B-A3B Metal v1 keeps BF16 KV cache; --kv-fp8 is not wired.");
    }
    if cli.speculative_decode || cli.batched_spec_verify {
        anyhow::bail!(
            "Qwen3.6-35B-A3B Metal v1 does not wire the MTP/speculative decode path yet."
        );
    }
    Ok(())
}

pub fn validate_cuda_v1_flags(cli: &crate::Cli, entry: &RegistryEntry) -> Result<()> {
    if entry.backend != Backend::Cuda {
        return Ok(());
    }
    if cli.fp8_runtime {
        anyhow::bail!(
            "Qwen3.6-35B-A3B CUDA v1 supports INT4/q4km decode only; \
             --fp8-runtime is still HIP-only."
        );
    }
    if cli.kv_fp8 {
        anyhow::bail!("Qwen3.6-35B-A3B CUDA v1 keeps BF16 KV cache; --kv-fp8 is still HIP-only.");
    }
    if cli.speculative_decode || cli.batched_spec_verify {
        anyhow::bail!("Qwen3.6-35B-A3B CUDA v1 does not wire the MTP/speculative decode path yet.");
    }
    if std::env::var("SUPERSONIC_VMM_KV").ok().as_deref() == Some("1") {
        anyhow::bail!(
            "Qwen3.6-35B-A3B CUDA v1 uses dense KV buffers; SUPERSONIC_VMM_KV=1 is not supported."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use clap::Parser;

    use super::*;
    use crate::registry::{lookup, GpuArch, ModelVariant};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn cuda_entry() -> &'static RegistryEntry {
        lookup(
            &ModelVariant::Qwen3_6_35B_A3B,
            &Backend::Cuda,
            &GpuArch::Sm86,
        )
        .expect("qwen3.6-35b-a3b CUDA sm86 registry entry")
    }

    fn cli(extra: &[&str]) -> crate::Cli {
        let mut args = vec![
            "supersonic",
            "--model",
            "qwen3.6-35b-a3b",
            "--model-dir",
            "/tmp/qwen36",
            "--dry-run",
        ];
        args.extend_from_slice(extra);
        crate::Cli::parse_from(args)
    }

    fn cuda_v1_error(extra: &[&str]) -> String {
        validate_cuda_v1_flags(&cli(extra), cuda_entry())
            .expect_err("CUDA v1 policy should reject this flag set")
            .to_string()
    }

    #[test]
    fn cuda_v1_accepts_default_int4_path() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("SUPERSONIC_VMM_KV");

        validate_cuda_v1_flags(&cli(&[]), cuda_entry()).expect("default CUDA v1 flags");
    }

    #[test]
    fn cuda_v1_rejects_hip_only_weight_and_kv_modes() {
        assert!(cuda_v1_error(&["--fp8-runtime"]).contains("--fp8-runtime"));
        assert!(cuda_v1_error(&["--kv-fp8"]).contains("--kv-fp8"));
    }

    #[test]
    fn cuda_v1_rejects_speculative_decode_modes() {
        assert!(cuda_v1_error(&["--speculative-decode"]).contains("speculative"));
        assert!(cuda_v1_error(&["--batched-spec-verify"]).contains("speculative"));
    }

    #[test]
    fn cuda_v1_rejects_forced_kv_vmm() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::set_var("SUPERSONIC_VMM_KV", "1");
        let err = cuda_v1_error(&[]);
        std::env::remove_var("SUPERSONIC_VMM_KV");

        assert!(err.contains("SUPERSONIC_VMM_KV=1"));
    }
}
