use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

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
    let variant = if cli.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        model_store::fetch::BakeVariant::Int4Gptq
    };
    let bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    // `should_fetch_exact_bake` honors --download-bake (force) and refuses to
    // fetch when an up-to-date bake is already present.
    let force_download = cli.download_bake;
    if !cli.no_download
        && crate::should_fetch_exact_bake(force_download, model_store::version_ok(&bake_dir))
    {
        let canonical_model = entry.model.to_string();
        match crate::try_download_bake(cli, variant, &canonical_model, &bake_dir) {
            Ok(true) => eprintln!(
                "[fetch] installed qwen3.6-MoE {} bake at {}",
                if cli.fp8_runtime { "FP8" } else { "INT4" },
                bake_dir.display()
            ),
            Ok(false) => {}
            Err(e) => eprintln!(
                "[fetch] qwen3.6-MoE {} bake fetch failed: {e}",
                if cli.fp8_runtime { "FP8" } else { "INT4" }
            ),
        }
    }
    Ok(())
}

pub(crate) fn select_decode_bake(
    model_dir: &Path,
    fp8_runtime: bool,
    int4_runtime: bool,
) -> Result<DecodeBakeSelection> {
    // INT4 remains the default small-VRAM path; explicit --fp8-runtime
    // selects the native FP8 bake.
    let fp8_dir = model_store::bake_dir_fp8(model_dir);
    let int4_dir = model_store::bake_dir_int4(model_dir);
    let bf16_dir = model_store::bake_dir(model_dir);
    let (bake_dir, weight_mode) = if fp8_runtime {
        if !fp8_dir.exists() {
            return Err(anyhow!(
                "--fp8-runtime requested but no FP8-native bake exists at {}. \
                 Create one with `python3 oracle/bake_fp8.py --model-dir {}`.",
                fp8_dir.display(),
                model_dir.display()
            ));
        }
        (fp8_dir, Qwen36WeightMode::Fp8)
    } else if int4_dir.exists() {
        (int4_dir, Qwen36WeightMode::Int4)
    } else if int4_runtime {
        return Err(anyhow!(
            "--int4 requested but no INT4-GPTQ bake exists at {}. \
             Create one with `python3 oracle/bake_int4.py --model-dir {}` \
             or allow the release bake download.",
            int4_dir.display(),
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
