use anyhow::Result;
use model_store::manifest::QuantProfile;

use crate::certified_kv;
use crate::registry::{Backend, GpuArch, ModelFamily, ModelVariant};
use crate::Cli;

pub(crate) fn q4km_like(cli: &Cli) -> bool {
    matches!(
        crate::bakes::effective_quant_profile(cli),
        Ok(QuantProfile::Q4Km | QuantProfile::Q4KmGptq)
    )
}

pub(crate) fn validate_global_flags(
    cli: &Cli,
    model_variant: &ModelVariant,
    backend: Backend,
) -> Result<()> {
    let q4km_like = q4km_like(cli);
    let profile = crate::bakes::effective_quant_profile(cli)?;
    if cli.q4km && cli.q4km_gptq {
        anyhow::bail!("--q4km is mutually exclusive with --q4km-gptq");
    }
    crate::bakes::validate_effective_flm_source_model(cli, model_variant)?;
    if q4km_like && cli.int8 {
        anyhow::bail!("--q4km/--q4km-gptq are mutually exclusive with --int8");
    }
    if q4km_like
        && !matches!(
            model_variant.family(),
            ModelFamily::Qwen35 | ModelFamily::Qwen3Moe | ModelFamily::Qwen36Moe
        )
    {
        anyhow::bail!("--q4km/--q4km-gptq are currently supported only for Qwen models");
    }
    if cli.q4km
        && !(backend == Backend::Cuda
            || (backend == Backend::Hip
                && matches!(
                    model_variant.family(),
                    ModelFamily::Qwen35 | ModelFamily::Qwen36Moe
                ))
            || (backend == Backend::Metal && model_variant.family() == ModelFamily::Qwen36Moe))
    {
        anyhow::bail!(
            "--q4km raw GGML blocks are currently supported only on CUDA Qwen paths, HIP Qwen3.5/3.6 paths, and Metal Qwen3.5/3.6 MoE"
        );
    }
    if cli.q4km_gptq
        && !(backend == Backend::Cuda
            || (backend == Backend::Hip
                && matches!(
                    model_variant.family(),
                    ModelFamily::Qwen35 | ModelFamily::Qwen36Moe
                ))
            || (backend == Backend::Metal && model_variant.family() == ModelFamily::Qwen36Moe))
    {
        anyhow::bail!(
            "--q4km-gptq is currently supported on CUDA Qwen paths, HIP Qwen3.5/3.6 paths, and Metal Qwen3.5/3.6 MoE"
        );
    }
    if cli.flm_file.is_some() && cli.no_bake {
        anyhow::bail!("--flm-file and --no-bake are mutually exclusive");
    }
    if cli.flm_file.is_some() && q4km_like {
        anyhow::bail!("--flm-file is not wired for --q4km/--q4km-gptq bakes");
    }
    if cli.flm_file.is_some() && cli.int8 {
        anyhow::bail!("--flm-file is not wired for --int8 bakes");
    }
    if matches!(model_variant.family(), ModelFamily::Qwen3Moe) {
        if !cli.int4 || profile != QuantProfile::Int4Gptq {
            anyhow::bail!("qwen3-30b-a3b v1 requires --int4 / int4-gptq");
        }
        if !matches!(backend, Backend::Hip | Backend::Metal) {
            anyhow::bail!("qwen3-30b-a3b v1 is supported only on HIP and Metal");
        }
    }
    if cli.gguf_file.is_some() && profile != QuantProfile::Q4Km {
        anyhow::bail!("--gguf-file requires --q4km");
    }
    if cli.no_bake && q4km_like {
        anyhow::bail!("--q4km/--q4km-gptq require a baked package; omit --no-bake");
    }
    if cli.int8 && profile != QuantProfile::Bf16 {
        anyhow::bail!(
            "--int8 is mutually exclusive with --weight-quant and legacy weight quant flags"
        );
    }
    if profile.is_runtime_backed_lowbit() {
        anyhow::bail!(
            "--weight-quant {profile} has manifest and package naming support, but Qwen HIP runtime kernels/loaders for this profile are not implemented yet"
        );
    }
    if matches!(
        (profile, model_variant.family()),
        (
            QuantProfile::Int4Awq | QuantProfile::Int4Autoround | QuantProfile::Int4Hqq,
            ModelFamily::Qwen3Moe | ModelFamily::Qwen36Moe
        )
    ) {
        anyhow::bail!(
            "--weight-quant {profile} is currently supported only for Qwen3.5; Qwen3/Qwen3.6-MoE bake selection/runtime support is not wired for this profile yet"
        );
    }
    if profile == QuantProfile::Int4Awq && backend == Backend::Cuda {
        anyhow::bail!(
            "--weight-quant int4-awq is currently supported only on HIP; CUDA INT4 kernels do not apply AWQ sidecars yet"
        );
    }
    if matches!(
        profile,
        QuantProfile::Int4Awq | QuantProfile::Int4Autoround | QuantProfile::Int4Hqq
    ) && !matches!(
        model_variant.family(),
        ModelFamily::Qwen35 | ModelFamily::Qwen3Moe | ModelFamily::Qwen36Moe
    ) {
        anyhow::bail!(
            "--weight-quant {profile} is Qwen-first; Gemma 4 and Phi 4 ports are follow-up work"
        );
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::validate_global_flags;
    use crate::registry::{Backend, ModelVariant};
    use crate::Cli;

    fn cli(extra: &[&str]) -> Cli {
        let mut args = vec!["supersonic", "--model-dir", "/tmp/model", "--dry-run"];
        args.extend_from_slice(extra);
        Cli::parse_from(args)
    }

    fn cli_with_model_dir(model_dir: &str, extra: &[&str]) -> Cli {
        let mut args = vec!["supersonic", "--model-dir", model_dir, "--dry-run"];
        args.extend_from_slice(extra);
        Cli::parse_from(args)
    }

    #[test]
    fn rejects_flm_file_for_non_qwen36_dense_model() {
        let err = validate_global_flags(
            &cli(&["--flm-file", "/tmp/model.flm"]),
            &ModelVariant::Gemma4_E2B,
            Backend::Hip,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("FLM"), "{err}");
        assert!(err.contains("qwen3.6-27b"), "{err}");
    }

    #[test]
    fn rejects_flm_model_dir_for_non_qwen36_dense_model() {
        let err = validate_global_flags(
            &cli_with_model_dir("/tmp/model.flm", &[]),
            &ModelVariant::Phi4_Mini,
            Backend::Hip,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("FLM"), "{err}");
        assert!(err.contains("qwen3.6-27b"), "{err}");
    }

    #[test]
    fn rejects_new_native_int4_profiles_for_qwen36_until_selection_is_wired() {
        for profile in ["int4-awq", "int4-autoround", "int4-hqq"] {
            let err = validate_global_flags(
                &cli(&["--weight-quant", profile]),
                &ModelVariant::Qwen3_6_35B_A3B,
                Backend::Hip,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("Qwen3.6-MoE bake selection/runtime support is not wired"));
        }
    }

    #[test]
    fn rejects_awq_on_cuda_until_sidecars_are_applied() {
        let err = validate_global_flags(
            &cli(&["--weight-quant", "int4-awq"]),
            &ModelVariant::Qwen3_5_4B,
            Backend::Cuda,
        )
        .expect_err("CUDA AWQ should fail until sidecars are applied")
        .to_string();
        assert!(err.contains("CUDA INT4 kernels do not apply AWQ sidecars"));
    }

    #[test]
    fn allows_gptq_for_qwen36_native_int4_layout() {
        validate_global_flags(
            &cli(&["--weight-quant", "int4-gptq"]),
            &ModelVariant::Qwen3_6_35B_A3B,
            Backend::Hip,
        )
        .expect("GPTQ remains the wired Qwen3.6 native INT4 path");
    }

    #[test]
    fn allows_q4km_gptq_for_qwen35_moe_on_metal() {
        validate_global_flags(
            &cli(&["--q4km-gptq"]),
            &ModelVariant::Qwen3_5_35B_A3B,
            Backend::Metal,
        )
        .expect("Qwen3.5-35B-A3B uses the Metal MoE native INT4 path");
    }

    #[test]
    fn allows_raw_q4km_for_qwen35_moe_on_metal() {
        validate_global_flags(
            &cli(&["--q4km"]),
            &ModelVariant::Qwen3_5_35B_A3B,
            Backend::Metal,
        )
        .expect("Qwen3.5/3.6 MoE Metal can load raw GGML q4km through the staged path");
    }

    #[test]
    fn allows_raw_q4km_for_qwen36_dense_on_hip() {
        validate_global_flags(&cli(&["--q4km"]), &ModelVariant::Qwen3_6_27B, Backend::Hip)
            .expect("Qwen3.6-27B HIP can load raw GGML q4km through the Qwen3.5 dense path");
    }

    #[test]
    fn allows_q4km_gptq_for_qwen36_dense_on_hip() {
        validate_global_flags(
            &cli(&["--q4km-gptq"]),
            &ModelVariant::Qwen3_6_27B,
            Backend::Hip,
        )
        .expect("Qwen3.6-27B HIP can load Q4KM-sourced native INT4 through the Qwen3.5 dense path");
    }
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
    backend: Backend,
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
        let qwen36_cross_family_cuda =
            backend == Backend::Cuda && matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B);
        if backend != Backend::Hip && !qwen36_cross_family_cuda {
            anyhow::bail!(
                "SpecPrefill is supported on HIP, plus CUDA for the Qwen3.6-35B-A3B \
                 cross-family cosine path (got backend={backend:?}, model={model_variant}). \
                 Re-run with a supported backend or omit --specprefill-draft-dir to use \
                 the dense path."
            );
        }
        // SpecPrefill is enabled for Qwen3.5-9B (Phase C/D) and Qwen3.6-35B-A3B
        // (cross-model drafting research, R1). The 27B variant is excluded:
        // its weights (~27 GiB FP8) don't fit on 24 GiB and the Qwen36Moe-family
        // VMM relief is unavailable to it (it sits in `ModelFamily::Qwen35`,
        // not `Qwen36Moe`). Qwen3.6-35B-A3B (MoE) fits via INT4 + expert VMM
        // and is the only viable Qwen3.6 target on gfx1100.
        if !matches!(
            model_variant,
            ModelVariant::Qwen3_5_9B | ModelVariant::Qwen3_6_35B_A3B
        ) {
            anyhow::bail!(
                "--specprefill-draft-dir is supported on qwen3.5-9b and qwen3.6-35b-a3b \
                 (got {model_variant}). Qwen3.6-27B is excluded: doesn't fit on 24 GiB \
                 and its Qwen35 family lacks VMM relief."
            );
        }
        if cli.batch_size != 1 {
            anyhow::bail!("SpecPrefill requires --batch-size 1");
        }
        if cli.dflash {
            anyhow::bail!("--specprefill-* and --dflash cannot be combined");
        }
        match cli.specprefill_algorithm.as_str() {
            "cosine" | "lookahead" => {}
            other => {
                anyhow::bail!(
                    "--specprefill-algorithm must be \"cosine\" or \"lookahead\" (got {other:?})"
                );
            }
        }
        if matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B)
            && cli.specprefill_algorithm == "lookahead"
        {
            anyhow::bail!(
                "--specprefill-algorithm lookahead is not supported for qwen3.6-35b-a3b; \
                 use cosine, the default cross-family scorer"
            );
        }
        if cli.kv_fp8 && cli.specprefill_algorithm == "lookahead" {
            anyhow::bail!(
                "--specprefill-algorithm lookahead cannot be combined with --kv-fp8: the \
                 BF16 step-copy fallback used by the speculator's lookahead decode \
                 (`kernel_ffi::certified_kv::copy_step_bf16`, added in PR #177) requires \
                 BF16 destination buffers, but --kv-fp8 makes the K/V cache U8. The combo \
                 trips a runtime error several seconds into decode. Use \
                 `--specprefill-algorithm cosine` (the default) to combine SpecPrefill \
                 with --kv-fp8 — the cosine path doesn't run drafter decode."
            );
        }
        if let Some(keep) = cli.specprefill_keep_ratio {
            if !(0.05..=1.0).contains(&keep) {
                anyhow::bail!("--specprefill-keep-ratio must be in [0.05, 1.0] (got {keep})");
            }
        }
        if let Some(window) = cli.specprefill_pool_window {
            if window % 2 != 1 || window == 0 {
                anyhow::bail!("--specprefill-pool-window must be odd and > 0 (got {window})");
            }
        }
        if let Some(suffix) = cli.specprefill_always_keep_suffix {
            if suffix == 0 {
                anyhow::bail!(
                    "--specprefill-always-keep-suffix must be >= 1 (the last prompt \
                     token must be kept; the first decode logits come from this slot)"
                );
            }
        }
        if let Some(lookahead) = cli.specprefill_lookahead {
            if !(1..=16).contains(&lookahead) {
                anyhow::bail!("--specprefill-lookahead must be in [1, 16] (got {lookahead})");
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
