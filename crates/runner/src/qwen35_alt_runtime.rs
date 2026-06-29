use anyhow::Result;

use crate::bakes::{cli_variant, effective_flm_source, ensure_hf_metadata_present};
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
    reject_flm_alt_runtime_if_requested(cli)?;

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

pub(crate) fn reject_flm_alt_runtime_if_requested(cli: &Cli) -> Result<()> {
    let Some(flm_source) = effective_flm_source(cli) else {
        return Ok(());
    };
    if cli.dflash {
        anyhow::bail!(
            "FLM source {} is not supported with --dflash because the DFlash runtime still reads config.json/tokenizer.json from --model-dir",
            flm_source.display()
        );
    }
    if cli.specprefill_draft_dir.is_some() {
        anyhow::bail!(
            "FLM source {} is not supported with --specprefill-draft-dir because the SpecPrefill runtime is not FLM-aware",
            flm_source.display()
        );
    }
    Ok(())
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
            let local_bake_ok = dflash_local_bake_ok(cli, variant);
            let canonical_model = model_variant.to_string();
            match try_download_bake(cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => {
                    eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
                }
                Ok(false) => {
                    if local_bake_ok {
                        eprintln!(
                            "[fetch] no {variant} bake downloaded; falling back to local bake"
                        );
                    } else if q4km_like(cli) {
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
                    if local_bake_ok {
                        eprintln!("[fetch] {e}; falling back to local bake");
                    } else if q4km_like(cli) {
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

fn dflash_local_bake_ok(cli: &Cli, variant: model_store::fetch::BakeVariant) -> bool {
    variant == model_store::fetch::BakeVariant::Q4Km && cli.gguf_file.is_some()
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use model_store::fetch::BakeVariant;

    use super::{dflash_local_bake_ok, reject_flm_alt_runtime_if_requested};
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
    fn rejects_flm_source_for_dflash_before_dispatch() {
        let err = reject_flm_alt_runtime_if_requested(&cli_with_model_dir(
            "/tmp/model.flm",
            &["--dflash", "--dflash-draft-dir", "/tmp/draft"],
        ))
        .unwrap_err()
        .to_string();

        assert!(err.contains("FLM"), "{err}");
        assert!(err.contains("--dflash"), "{err}");
    }

    #[test]
    fn rejects_flm_source_for_specprefill_before_dispatch() {
        let err = reject_flm_alt_runtime_if_requested(&cli_with_model_dir(
            "/tmp/model.flm",
            &["--specprefill-draft-dir", "/tmp/draft"],
        ))
        .unwrap_err()
        .to_string();

        assert!(err.contains("FLM"), "{err}");
        assert!(err.contains("--specprefill-draft-dir"), "{err}");
    }

    #[test]
    fn dflash_allows_local_raw_q4km_bake_from_gguf() {
        assert!(dflash_local_bake_ok(
            &cli(&["--q4km", "--gguf-file", "/tmp/model.gguf"]),
            BakeVariant::Q4Km
        ));
    }

    #[test]
    fn dflash_q4km_download_preflight_still_bails_without_gguf_source() {
        assert!(!dflash_local_bake_ok(&cli(&["--q4km"]), BakeVariant::Q4Km));
    }

    #[test]
    fn dflash_q4km_gptq_is_not_locally_baked_from_raw_gguf() {
        assert!(!dflash_local_bake_ok(
            &cli(&["--q4km-gptq"]),
            BakeVariant::Q4KmGptq
        ));
    }
}
