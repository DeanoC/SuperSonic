use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;

use crate::registry::ModelVariant;
use crate::Cli;

fn resolve_release_source(cli: &Cli) -> Result<model_store::fetch::ReleaseSource> {
    let raw = cli
        .bake_release
        .clone()
        .or_else(|| env::var("SUPERSONIC_BAKE_RELEASE").ok());
    match raw {
        Some(s) if !s.is_empty() => model_store::fetch::ReleaseSource::from_override(&s)
            .map_err(|e| anyhow::anyhow!("invalid --bake-release: {e}")),
        _ => Ok(model_store::fetch::ReleaseSource::default_for_format_version()),
    }
}

fn log_fetch_progress() -> impl Fn(model_store::fetch::FetchProgress) {
    use std::cell::Cell;
    let last_pct = Cell::new(i32::MIN);
    let last_part = Cell::new(u32::MAX);
    move |p| {
        use model_store::fetch::FetchProgress::*;
        match p {
            ResolvingIndex => eprintln!("[fetch] resolving release index..."),
            Downloading {
                part,
                total_parts,
                bytes_done,
                bytes_total,
            } => {
                let pct = if bytes_total > 0 {
                    (bytes_done * 100 / bytes_total) as i32
                } else {
                    0
                };
                if part != last_part.get() {
                    last_part.set(part);
                    last_pct.set(i32::MIN);
                    eprintln!(
                        "[fetch] downloading part {part}/{total_parts} ({} MiB)",
                        bytes_total / (1024 * 1024)
                    );
                }
                if pct / 5 != last_pct.get() / 5 {
                    last_pct.set(pct);
                    eprintln!(
                        "[fetch]   {pct}% ({} / {} MiB)",
                        bytes_done / (1024 * 1024),
                        bytes_total / (1024 * 1024)
                    );
                }
            }
            Verifying => eprintln!("[fetch] verifying SHA-256..."),
            Extracting => eprintln!("[fetch] extracting tarball..."),
            Done => eprintln!("[fetch] done"),
        }
    }
}

pub(crate) fn try_download_bake(
    cli: &Cli,
    variant: model_store::fetch::BakeVariant,
    model_cli_name: &str,
    target: &Path,
) -> Result<bool> {
    if cli.no_download {
        return Ok(false);
    }
    let source = resolve_release_source(cli)?;
    eprintln!(
        "[fetch] downloading {model_cli_name} {variant} from {}/{}",
        source.repo_slug, source.tag
    );
    let progress = log_fetch_progress();
    let req = model_store::fetch::FetchRequest {
        source: &source,
        model_cli_name,
        variant,
        target_bake_dir: target,
        target_model_dir: &cli.model_dir,
        progress: &progress,
    };
    model_store::fetch::fetch_bake(req).map_err(|e| anyhow::anyhow!("fetch bake: {e}"))?;
    Ok(true)
}

/// Pick the variant the CLI flags imply, using the same priority order as
/// the rest of the runner.
pub(crate) fn cli_variant(cli: &Cli) -> model_store::fetch::BakeVariant {
    model_store::fetch::variant_from_flags(cli.q4km_gptq, cli.q4km, cli.int4, cli.fp8_runtime)
}

pub(crate) fn variant_version_ok(
    variant: model_store::fetch::BakeVariant,
    bake_dir: &Path,
) -> bool {
    model_store::fetch::version_ok_for_variant(variant, bake_dir)
}

pub(crate) fn should_fetch_bake(
    download_bake: bool,
    bootstrap_downloaded: bool,
    local_version_ok: bool,
) -> bool {
    (download_bake && !bootstrap_downloaded) || !local_version_ok
}

pub(crate) fn should_fetch_exact_bake(download_bake: bool, local_version_ok: bool) -> bool {
    download_bake || !local_version_ok
}

pub(crate) fn load_qwen35_weights(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    ordinal: usize,
    weight_prefix: &str,
    bootstrap_downloaded: bool,
    q4km_like: bool,
) -> Result<qwen35::weights::Qwen35Weights> {
    if cli.no_bake {
        eprintln!("[weights] loading from safetensors (--no-bake)...");
        return qwen35::weights::Qwen35Weights::load(
            &cli.model_dir,
            text_config,
            ordinal,
            weight_prefix,
        )
        .map_err(|e| anyhow::anyhow!("load weights: {e}"));
    }

    let variant = if cli.int4 {
        model_store::fetch::BakeVariant::Int4Gptq
    } else if cli.fp8_runtime {
        model_store::fetch::BakeVariant::Fp8Native
    } else {
        cli_variant(cli)
    };
    let mut bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;

    if should_fetch_bake(
        cli.download_bake,
        bootstrap_downloaded,
        variant_version_ok(variant, &bake_dir),
    ) {
        let local_bake_ok = matches!(
            variant,
            model_store::fetch::BakeVariant::Bf16 | model_store::fetch::BakeVariant::Fp8Native
        ) || (variant == model_store::fetch::BakeVariant::Q4Km
            && cli.gguf_file.is_some());
        let canonical_model = model_variant.to_string();
        match try_download_bake(cli, variant, &canonical_model, &bake_dir) {
            Ok(true) => {
                eprintln!("[fetch] installed {variant} bake at {}", bake_dir.display());
            }
            Ok(false) => {
                if !local_bake_ok {
                    if q4km_like {
                        anyhow::bail!(
                            "no {variant} bake at {} and --no-download set.\n\
                             Rerun with --gguf-file /path/to/model.gguf to create a local raw GGML q4km bake, \
                             or provide/download a q4km-gptq bake.",
                            bake_dir.display(),
                        );
                    } else {
                        anyhow::bail!(
                            "no {variant} bake at {} and --no-download set.\n\
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}",
                            bake_dir.display(),
                            cli.model_dir.display(),
                        );
                    }
                }
            }
            Err(e) => {
                if local_bake_ok {
                    eprintln!("[fetch] {e}; falling back to local bake");
                } else if q4km_like {
                    anyhow::bail!(
                        "could not obtain {variant} bake: {e}\n\n\
                         Rerun with --gguf-file /path/to/model.gguf to create a local raw GGML q4km bake, \
                         or provide/download a q4km-gptq bake.",
                    );
                } else {
                    anyhow::bail!(
                        "could not obtain {variant} bake: {e}\n\n\
                         INT4 baking requires a GPTQ calibration pass in Python. \
                         Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {}\n\
                         then `python oracle/upload_bake.py --model {} --int4 --model-dir {}` to publish.",
                        cli.model_dir.display(),
                        cli.model,
                        cli.model_dir.display(),
                    );
                }
            }
        }
        if !variant_version_ok(variant, &bake_dir) && local_bake_ok {
            let bake_start = Instant::now();
            if cli.q4km {
                bake_dir = model_store::bake_dir_q4km(&cli.model_dir);
                if !model_store::version_ok_q4km(&bake_dir) {
                    run_q4km_baker(cli, &bake_dir)?;
                }
            } else {
                let mode_str = if cli.fp8_runtime { " (FP8 native)" } else { "" };
                eprintln!("[bake] no baked package found — baking weights{mode_str} (one-time)...");
                let layer_is_full: Vec<bool> = (0..text_config.num_hidden_layers)
                    .map(|i| text_config.is_full_attention(i))
                    .collect();
                model_store::bake_qwen35(
                    &cli.model_dir,
                    weight_prefix,
                    text_config.num_hidden_layers,
                    &layer_is_full,
                    cli.fp8_runtime,
                    &|msg| eprintln!("{msg}"),
                )
                .map_err(|e| anyhow::anyhow!("bake weights: {e}"))?;
            }
            eprintln!("[bake] done in {:.1}s", bake_start.elapsed().as_secs_f64());
        }
    }
    if variant_version_ok(variant, &bake_dir) {
        eprintln!("[weights] found baked package at {}", bake_dir.display());
    }
    let store = model_store::BakedStore::open(&bake_dir)
        .map_err(|e| anyhow::anyhow!("open baked store: {e}"))?;
    qwen35::weights::Qwen35Weights::load_baked(&store, text_config, ordinal, weight_prefix)
        .map_err(|e| anyhow::anyhow!("load baked weights: {e}"))
}

pub(crate) fn ensure_gemma4_int4_bake(
    cli: &Cli,
    model_variant: &ModelVariant,
    bootstrap_downloaded: bool,
) -> Result<()> {
    let target = crate::gemma4_int4_engine::int4_bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
    if should_fetch_bake(
        cli.download_bake,
        bootstrap_downloaded,
        crate::gemma4_int4_engine::int4_bake_ok(&cli.model_dir),
    ) {
        let canonical_model = model_variant.to_string();
        match try_download_bake(
            cli,
            model_store::fetch::BakeVariant::Int4Gptq,
            &canonical_model,
            &target,
        ) {
            Ok(true) => eprintln!(
                "[fetch] installed Gemma 4 INT4 bake at {}",
                target.display()
            ),
            Ok(false) => {
                anyhow::bail!(
                    "No INT4 bake at {} and --no-download set.\nRun on a bigger machine:\n  python oracle/bake_int4_gemma4.py --model-dir {}",
                    target.display(),
                    cli.model_dir.display(),
                );
            }
            Err(e) => {
                anyhow::bail!(
                    "could not obtain Gemma 4 INT4 bake: {e}\n\nRun on a bigger machine:\n  python oracle/bake_int4_gemma4.py --model-dir {}\nthen `python oracle/upload_bake.py --model {} --int4 --model-dir {}` to publish.",
                    cli.model_dir.display(),
                    cli.model,
                    cli.model_dir.display(),
                );
            }
        }
    }
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("could not resolve repository root from CARGO_MANIFEST_DIR"))
}

pub(crate) fn run_q4km_baker(cli: &Cli, bake_dir: &Path) -> Result<()> {
    let gguf_file = cli
        .gguf_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--q4km local bake requires --gguf-file"))?;
    let script = repo_root()?.join("oracle/bake_q4km.py");
    let python = env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    eprintln!(
        "[bake] translating GGUF {} into native q4km bake at {}",
        gguf_file.display(),
        bake_dir.display()
    );
    let status = std::process::Command::new(&python)
        .arg(&script)
        .arg("--model-dir")
        .arg(&cli.model_dir)
        .arg("--model")
        .arg(&cli.model)
        .arg("--gguf-file")
        .arg(gguf_file)
        .arg("--out-dir")
        .arg(bake_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("launch q4km baker {script:?}: {e}"))?;
    if !status.success() {
        anyhow::bail!("q4km baker failed with status {status}");
    }
    Ok(())
}

/// When `--model-dir` has no `config.json`, fetch the bake first. The
/// tarball bundles HF metadata under `hf/`, which the downloader extracts
/// into `--model-dir` before anything else reads from it.
pub(crate) fn ensure_hf_metadata_present(cli: &Cli, model_variant: &ModelVariant) -> Result<bool> {
    if cli.no_bake || cli.no_download {
        return Ok(false);
    }
    if cli.model_dir.join("config.json").exists() {
        return Ok(false);
    }
    let variant = cli_variant(cli);
    let bake_dir = variant.bake_dir(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("acquire bake lock: {e}"))?;
    // Race: another process might have populated config between our check
    // above and the lock acquisition.
    if cli.model_dir.join("config.json").exists() {
        return Ok(false);
    }
    let canonical_model = model_variant.to_string();
    eprintln!(
        "[fetch] --model-dir has no config.json; downloading bake to populate \
         HF metadata and weights in one pass"
    );
    try_download_bake(cli, variant, &canonical_model, &bake_dir)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{should_fetch_bake, should_fetch_exact_bake};

    #[test]
    fn bootstrap_download_satisfies_forced_bake_download() {
        assert!(!should_fetch_bake(true, true, true));
    }

    #[test]
    fn forced_bake_download_still_fetches_without_bootstrap() {
        assert!(should_fetch_bake(true, false, true));
    }

    #[test]
    fn invalid_local_bake_fetches_even_after_bootstrap_attempt() {
        assert!(should_fetch_bake(false, true, false));
    }

    #[test]
    fn forced_exact_bake_fetch_ignores_metadata_bootstrap() {
        assert!(should_fetch_exact_bake(true, true));
    }
}
