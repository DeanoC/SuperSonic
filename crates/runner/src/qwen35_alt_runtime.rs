use anyhow::Result;

use crate::bakes::ensure_hf_metadata_present;
use crate::registry::{ModelVariant, RegistryEntry};
use crate::{
    qwen35_dflash_engine, should_fetch_exact_bake, specprefill_engine, try_download_bake, Cli,
};

pub(crate) fn run_qwen35_alt_runtime_if_requested(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<bool> {
    if cli.dflash {
        run_qwen35_dflash(cli, model_variant, entry, ordinal, total_vram)?;
        return Ok(true);
    }

    // --specprefill-* dispatch. Validation already ran in
    // validate_specprefill_flags; the presence of --specprefill-draft-dir
    // is the gate that switches to the SpecPrefill orchestrator.
    if cli.specprefill_draft_dir.is_some() {
        specprefill_engine::run_specprefill(cli, model_variant, entry, ordinal, total_vram)?;
        return Ok(true);
    }

    Ok(false)
}

fn run_qwen35_dflash(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    // DFlash needs the target's HF metadata (config.json + tokenizer.json)
    // and the INT4 bake. Reuse the same download hooks as the regular
    // Qwen35 path so the dflash dispatch is self-contained on a fresh
    // machine: ensure_hf_metadata_present fetches HF metadata from the
    // bake tarball if config.json is missing, then we verify or download
    // the INT4 bake itself.
    ensure_hf_metadata_present(cli, model_variant)?;
    if !cli.no_bake {
        let variant = model_store::fetch::BakeVariant::Int4Gptq;
        let bake_dir = variant.bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
        if should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir)) {
            let canonical_model = model_variant.to_string();
            match try_download_bake(cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => {
                    eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                }
                Ok(false) => {
                    anyhow::bail!(
                        "no INT4 bake at {} and --no-download set.\n\
                         Run:\n  python oracle/bake_int4.py --model-dir {}",
                        bake_dir.display(),
                        cli.model_dir.display(),
                    );
                }
                Err(e) => {
                    anyhow::bail!(
                        "could not obtain INT4 bake for --dflash: {e}\n\n\
                         INT4 baking requires a GPTQ calibration pass in Python. \
                         Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                        cli.model_dir.display(),
                    );
                }
            }
        }
    }

    qwen35_dflash_engine::run_qwen35_dflash(cli, model_variant, entry, ordinal, total_vram)
}
