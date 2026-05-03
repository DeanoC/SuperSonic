use anyhow::{anyhow, Result};

use crate::registry::RegistryEntry;

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
