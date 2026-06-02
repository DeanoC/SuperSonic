use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use gpu_hal::Backend;
use model_store::manifest::QuantProfile;

use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
use crate::registry::RegistryEntry;

pub(crate) struct DecodeBakeSelection {
    pub(crate) bake_dir: PathBuf,
    pub(crate) weight_mode: Qwen36WeightMode,
}

pub(crate) fn ensure_qwen36_bake(cli: &crate::Cli, entry: &RegistryEntry) -> Result<()> {
    // Auto-download the requested release bake if missing or stale. 35B-A3B
    // INT4 calibration OOMs on 24 GiB hosts, so release-hosted bakes are the
    // realistic default for decode and for dry-run residency probes. Run this
    // before dry-run reporting so `SUPERSONIC_VMM_*_PROBE` can inspect a
    // freshly populated bake.
    let profile = crate::bakes::effective_quant_profile(cli)?;
    let variant = if profile == QuantProfile::Bf16
        && matches!(entry.backend, Backend::Cuda | Backend::Metal)
    {
        model_store::fetch::BakeVariant::Int4Gptq
    } else {
        model_store::fetch::variant_from_quant_profile(profile)
    };
    let bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    // `should_fetch_exact_bake` honors --download-bake (force) and refuses to
    // fetch when an up-to-date bake is already present.
    let force_download = cli.download_bake;
    if !cli.no_download
        && crate::should_fetch_exact_bake(
            force_download,
            model_store::fetch::version_ok_for_variant(variant, &bake_dir),
        )
    {
        let canonical_model = entry.model.to_string();
        match crate::try_download_bake(cli, variant, &canonical_model, &bake_dir) {
            Ok(true) => eprintln!(
                "[fetch] installed qwen3.5/3.6-MoE {variant} bake at {}",
                bake_dir.display()
            ),
            Ok(false) => {}
            Err(e) => eprintln!("[fetch] qwen3.5/3.6-MoE {variant} bake fetch failed: {e}"),
        }
    }
    Ok(())
}

pub(crate) fn select_decode_bake(
    model_dir: &Path,
    profile: QuantProfile,
    int4_runtime: bool,
) -> Result<DecodeBakeSelection> {
    // INT4 remains the default small-VRAM path; explicit --fp8-runtime
    // selects the native FP8 bake.
    let fp8_dir = model_store::bake_dir_fp8(model_dir);
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let q4km_gptq_dir = model_store::bake_dir_q4km_gptq(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, weight_mode) = if profile == QuantProfile::Fp8Native {
        if !fp8_dir.exists() {
            return Err(anyhow!(
                "--fp8-runtime requested but no FP8-native bake exists at {}. \
                 Create one with `python3 oracle/bake_fp8.py --model-dir {}`.",
                fp8_dir.display(),
                model_dir.display()
            ));
        }
        (fp8_dir, Qwen36WeightMode::Fp8)
    } else if profile == QuantProfile::Q4Km {
        return Err(anyhow!(
            "--q4km raw GGML Q4_K_M bakes are not supported by the Qwen3.5/3.6 MoE decode runtime because dense attention and shared-expert projections require native INT4 sidecars. Use --q4km-gptq instead."
        ));
    } else if profile == QuantProfile::Q4KmGptq {
        if !q4km_gptq_dir.exists() {
            return Err(anyhow!(
                "--q4km-gptq requested but no Q4KM-sourced GPTQ bake exists at {}. \
                 Create one with `python3 oracle/q4km_stream_gptq_bake.py --model-dir {} --model <qwen3.5-35b-a3b-or-qwen3.6-35b-a3b> --gguf-file /path/to/model.gguf` \
                 or allow the release bake download.",
                q4km_gptq_dir.display(),
                model_dir.display()
            ));
        }
        (q4km_gptq_dir, Qwen36WeightMode::Int4)
    } else if int4_dir.exists() {
        (int4_dir, Qwen36WeightMode::Int4)
    } else if int4_runtime && q4km_gptq_dir.exists() {
        (q4km_gptq_dir, Qwen36WeightMode::Int4)
    } else if int4_runtime {
        return Err(anyhow!(
             "an INT4-compatible bake is required, but neither INT4-GPTQ ({}) nor Q4KM-GPTQ ({}) exists. \
             Create one with `python3 oracle/bake_int4.py --model-dir {}` \
             or `python3 oracle/q4km_stream_gptq_bake.py --model-dir {} --model <qwen3.5-35b-a3b-or-qwen3.6-35b-a3b> --gguf-file /path/to/model.gguf`, \
             or allow the release bake download.",
            int4_dir.display(),
            q4km_gptq_dir.display(),
            model_dir.display(),
            model_dir.display()
        ));
    } else if bf16_dir.exists() {
        (bf16_dir, Qwen36WeightMode::Bf16)
    } else {
        return Err(anyhow!(
            "decode requires a baked package — neither FP8-native ({}), \
             INT4-GPTQ ({}) nor BF16 ({}) exists. Create one with the standard bake pipeline \
             or re-run with --dry-run for analytic accounting only.",
            fp8_dir.display(),
            int4_dir.display(),
            bf16_dir.display()
        ));
    };

    Ok(DecodeBakeSelection {
        bake_dir,
        weight_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "supersonic-qwen36-bake-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp model dir");
        dir
    }

    #[test]
    fn select_decode_bake_rejects_bf16_when_int4_required() {
        let model_dir = temp_model_dir("int4-required");
        fs::create_dir_all(model_store::bake_dir(&model_dir)).expect("create bf16 bake dir");

        let err = match select_decode_bake(&model_dir, QuantProfile::Bf16, true) {
            Ok(_) => panic!("int4 requirement must not fall back to bf16"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("an INT4-compatible bake is required"));
        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn select_decode_bake_rejects_raw_q4km_for_moe_decode() {
        let model_dir = temp_model_dir("raw-q4km");
        let q4km_dir = model_store::bake_dir_q4km(&model_dir);
        fs::create_dir_all(&q4km_dir).expect("create q4km bake dir");

        let err = match select_decode_bake(&model_dir, QuantProfile::Q4Km, true) {
            Ok(_) => panic!("raw q4km lacks native INT4 sidecars for MoE dense projections"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("Use --q4km-gptq instead"));
        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn select_decode_bake_uses_q4km_gptq_as_int4_compatible_bake() {
        let model_dir = temp_model_dir("q4km-gptq");
        let q4km_gptq_dir = model_store::bake_dir_q4km_gptq(&model_dir);
        fs::create_dir_all(&q4km_gptq_dir).expect("create q4km-gptq bake dir");

        let selected = select_decode_bake(&model_dir, QuantProfile::Q4KmGptq, true)
            .expect("q4km-gptq is a native INT4-compatible MoE bake");

        assert_eq!(selected.bake_dir, q4km_gptq_dir);
        assert_eq!(selected.weight_mode, Qwen36WeightMode::Int4);
        let _ = fs::remove_dir_all(model_dir);
    }
}
