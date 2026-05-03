use anyhow::Result;

use crate::certified_kv;
use crate::registry::{Backend, GpuArch, ModelFamily, ModelVariant};
use crate::Cli;

pub(crate) fn q4km_like(cli: &Cli) -> bool {
    cli.q4km || cli.q4km_gptq
}

pub(crate) fn validate_global_flags(
    cli: &Cli,
    model_variant: &ModelVariant,
    backend: Backend,
) -> Result<()> {
    let q4km_like = q4km_like(cli);
    if cli.q4km && cli.q4km_gptq {
        anyhow::bail!("--q4km is mutually exclusive with --q4km-gptq");
    }
    if q4km_like && (cli.int4 || cli.int8 || cli.fp8_runtime) {
        anyhow::bail!(
            "--q4km/--q4km-gptq are mutually exclusive with --int4, --int8, and --fp8-runtime"
        );
    }
    if q4km_like
        && !matches!(
            model_variant.family(),
            ModelFamily::Qwen35 | ModelFamily::Qwen36Moe
        )
    {
        anyhow::bail!("--q4km/--q4km-gptq are currently supported only for Qwen models");
    }
    if q4km_like && backend != Backend::Cuda {
        anyhow::bail!("--q4km/--q4km-gptq are currently supported only on CUDA");
    }
    if cli.gguf_file.is_some() && !cli.q4km {
        anyhow::bail!("--gguf-file requires --q4km");
    }
    if cli.no_bake && q4km_like {
        anyhow::bail!("--q4km/--q4km-gptq require a baked package; omit --no-bake");
    }
    if cli.int8 && (cli.int4 || cli.fp8_runtime) {
        anyhow::bail!("--int8 is mutually exclusive with --int4 and --fp8-runtime");
    }
    if cli.int8 && model_variant.family() != ModelFamily::Llama31 {
        anyhow::bail!("--int8 is currently supported only for llama3.1-8b on CUDA");
    }
    if cli.certified_kv {
        if model_variant.family() != ModelFamily::Llama31 {
            anyhow::bail!("--certified-kv is currently supported only for llama3.1-8b on CUDA");
        }
        if backend != Backend::Cuda {
            anyhow::bail!("--certified-kv requires --backend cuda");
        }
        if !cli.int8 {
            anyhow::bail!("--certified-kv requires the Llama 3.1 INT8 bake path (--int8)");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("--certified-kv is single-sequence at launch (--batch-size must be 1)");
        }
        let _ = certified_kv::CertifiedKvConfig::from_cli(cli)?;
    } else if cli.certified_kv_shadow_validate {
        anyhow::bail!("--certified-kv-shadow-validate requires --certified-kv");
    }
    Ok(())
}

pub(crate) fn validate_dflash_flags(cli: &Cli, model_variant: &ModelVariant) -> Result<()> {
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
    Ok(())
}

pub(crate) fn validate_specprefill_flags(
    cli: &Cli,
    model_variant: &ModelVariant,
) -> Result<()> {
    let any_specprefill_flag = cli.specprefill_draft_dir.is_some()
        || cli.specprefill_unload_draft
        || cli.specprefill_keep_ratio.is_some()
        || cli.specprefill_chunk_size.is_some()
        || cli.specprefill_pool_window.is_some()
        || cli.specprefill_lookahead.is_some()
        || cli.specprefill_always_keep_prefix.is_some()
        || cli.specprefill_always_keep_suffix.is_some();
    if cli.specprefill_draft_dir.is_some() {
        if !matches!(model_variant, ModelVariant::Qwen3_5_9B) {
            anyhow::bail!(
                "--specprefill-draft-dir is only supported on --model qwen3.5-9b in Phase C \
                 (got {model_variant})."
            );
        }
        if cli.batch_size != 1 {
            anyhow::bail!("SpecPrefill requires --batch-size 1");
        }
        if cli.dflash {
            anyhow::bail!("--specprefill-* and --dflash cannot be combined");
        }
        if let Some(keep) = cli.specprefill_keep_ratio {
            if !(0.05..=1.0).contains(&keep) {
                anyhow::bail!(
                    "--specprefill-keep-ratio must be in [0.05, 1.0] (got {keep})"
                );
            }
        }
        if let Some(window) = cli.specprefill_pool_window {
            if window % 2 != 1 || window == 0 {
                anyhow::bail!(
                    "--specprefill-pool-window must be odd and > 0 (got {window})"
                );
            }
        }
        if let Some(lookahead) = cli.specprefill_lookahead {
            if !(1..=16).contains(&lookahead) {
                anyhow::bail!(
                    "--specprefill-lookahead must be in [1, 16] (got {lookahead})"
                );
            }
        }
    } else if any_specprefill_flag {
        anyhow::bail!(
            "--specprefill-* flags require --specprefill-draft-dir (no specprefill is configured)"
        );
    }
    Ok(())
}

pub(crate) fn validate_gfx942_policy(
    cli: &Cli,
    model_variant: &ModelVariant,
    backend: Backend,
    gpu_arch: &GpuArch,
) -> Result<()> {
    if backend == Backend::Hip
        && *gpu_arch == GpuArch::Gfx1100
        && matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B)
        && cli.fp8_runtime
    {
        anyhow::bail!(
            "FP8 weights for Qwen3.6-35B-A3B require gfx942 or larger; on \
             gfx1100 use --int4 (+ optional --kv-fp8). 35 GiB FP8 weights do \
             not fit 24 GiB VRAM; expert streaming is tracked separately."
        );
    }

    if backend != Backend::Hip || *gpu_arch != GpuArch::Gfx942 {
        return Ok(());
    }

    if !matches!(
        model_variant,
        ModelVariant::Qwen3_5_0_8B
            | ModelVariant::Qwen3_5_2B
            | ModelVariant::Qwen3_5_4B
            | ModelVariant::Qwen3_5_9B
            | ModelVariant::Qwen3_6_35B_A3B
            | ModelVariant::Gemma4_E2B
            | ModelVariant::Gemma4_E4B
            | ModelVariant::Phi4_Mini
    ) {
        anyhow::bail!(
            "HIP gfx942 bring-up currently validates only Qwen3.5 models up to 9B BF16/INT4/FP8-runtime/KV-FP8, Qwen3.6 35B A3B INT4/FP8-runtime, Gemma 4 E2B BF16/INT4, Gemma 4 E4B BF16, and Phi-4-mini BF16/INT4/FP8-runtime/KV-FP8"
        );
    }
    let qwen35_model = matches!(
        model_variant,
        ModelVariant::Qwen3_5_0_8B
            | ModelVariant::Qwen3_5_2B
            | ModelVariant::Qwen3_5_4B
            | ModelVariant::Qwen3_5_9B
    );
    if (cli.fp8_runtime
        && !(qwen35_model
            || matches!(
                model_variant,
                ModelVariant::Qwen3_6_35B_A3B | ModelVariant::Phi4_Mini
            )))
        || (cli.kv_fp8
            && !(qwen35_model
                || matches!(
                    model_variant,
                    ModelVariant::Qwen3_6_35B_A3B | ModelVariant::Phi4_Mini
                )))
        || cli.q4km
        || cli.q4km_gptq
    {
        anyhow::bail!(
            "HIP gfx942 bring-up currently validates only BF16/INT4/FP8-runtime/KV-FP8 Qwen3.5 lanes, Qwen3.6 35B A3B INT4/FP8-runtime/KV-FP8, Gemma 4 E2B BF16/INT4, Gemma 4 E4B BF16, and Phi-4-mini BF16/INT4/FP8-runtime/KV-FP8"
        );
    }
    if matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B)
        && !(cli.int4 || cli.fp8_runtime || cli.kv_fp8)
    {
        anyhow::bail!(
            "HIP gfx942 Qwen3.6 35B A3B bring-up currently validates only --int4, --fp8-runtime, or --kv-fp8"
        );
    }
    if matches!(model_variant, ModelVariant::Gemma4_E4B) && cli.int4 {
        anyhow::bail!("HIP gfx942 Gemma 4 E4B bring-up currently validates only BF16");
    }
    if cli.batch_size != 1 {
        anyhow::bail!("HIP gfx942 bring-up currently supports only --batch-size 1");
    }
    Ok(())
}
