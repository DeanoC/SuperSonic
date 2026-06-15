use anyhow::Result;

use crate::bakes::{cli_variant, ensure_hf_metadata_present};
use crate::policy::q4km_like;
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
        let variant = cli_variant(cli)?;
        let bake_dir = variant.bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
        if should_fetch_exact_bake(
            cli.download_bake,
            model_store::fetch::version_ok_for_variant(variant, &bake_dir),
        ) {
            let canonical_model = model_variant.to_string();
            match try_download_bake(cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => {
                    eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                }
                Ok(false) => {
                    if q4km_like(cli) {
                        anyhow::bail!(
                            "no {variant} bake at {} and --no-download set.\n\
                             Rerun with --gguf-file /path/to/model.gguf to create a local raw GGML q4km bake, \
                             or provide/download a q4km-gptq bake.",
                            bake_dir.display(),
                        );
                    } else {
                        anyhow::bail!(
                            "no {variant} bake at {} and --no-download set.\n\
                             Run:\n  python oracle/bake_int4.py --model-dir {}",
                            bake_dir.display(),
                            cli.model_dir.display(),
                        );
                    }
                }
                Err(e) => {
                    if q4km_like(cli) {
                        anyhow::bail!(
                            "could not obtain {variant} bake for --dflash: {e}\n\n\
                             Rerun with --gguf-file /path/to/model.gguf to create a local raw GGML q4km bake, \
                             or provide/download a q4km-gptq bake.",
                        );
                    } else {
                        anyhow::bail!(
                            "could not obtain {variant} bake for --dflash: {e}\n\n\
                             INT4 baking requires a GPTQ calibration pass in Python. \
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                            cli.model_dir.display(),
                        );
                    }
                }
            }
        }
    }

    qwen35_dflash_engine::run_qwen35_dflash(cli, model_variant, entry, ordinal, total_vram)
}
