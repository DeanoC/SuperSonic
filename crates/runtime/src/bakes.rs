use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use crate::state::LoaderConfig;

/// When `--model-dir` has no `config.json`, trigger a bake download so the
/// tarball can populate HF metadata before we attempt to read it. Mirrors the
/// CLI's preflight.
pub(crate) fn ensure_hf_metadata_present(cfg: &LoaderConfig) -> Result<()> {
    if cfg.no_download {
        return Ok(());
    }
    if cfg.model_dir.join("config.json").exists() {
        return Ok(());
    }
    let bake_variant = selected_bake_variant(cfg);
    let bake_dir = bake_variant.bake_dir(&cfg.model_dir);
    let _lock = model_store::BakeLock::acquire(&cfg.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    if cfg.model_dir.join("config.json").exists() {
        return Ok(());
    }
    tracing::info!(
        "--model-dir has no config.json; fetching {} bake to populate HF metadata + weights",
        bake_variant
    );
    let _ = try_download_bake(cfg, bake_variant, &bake_dir)?;
    Ok(())
}

pub(crate) fn selected_bake_variant(cfg: &LoaderConfig) -> model_store::fetch::BakeVariant {
    model_store::fetch::variant_from_flags(cfg.q4km_gptq, cfg.q4km, cfg.int4, cfg.fp8_runtime)
}

pub(crate) fn ensure_qwen35_bake_available(
    cfg: &LoaderConfig,
    variant: model_store::fetch::BakeVariant,
    bake_dir: &Path,
    weight_prefix: &str,
    text_config: &qwen35::config::TextConfig,
) -> Result<()> {
    let _lock = model_store::BakeLock::acquire(&cfg.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;

    let bake_ok = || model_store::fetch::version_ok_for_variant(variant, bake_dir);
    if bake_ok() {
        return Ok(());
    }

    let local_bake_ok = matches!(
        variant,
        model_store::fetch::BakeVariant::Bf16 | model_store::fetch::BakeVariant::Fp8Native
    );
    // Network failures are recoverable when we can bake locally from the HF
    // safetensors; for low-bit variants they are fatal because calibration
    // must happen offline.
    let downloaded = match try_download_bake(cfg, variant, bake_dir) {
        Ok(v) => v,
        Err(e) if local_bake_ok => {
            tracing::warn!(
                "fetch {} bake failed: {}; falling back to local bake from safetensors",
                variant,
                e
            );
            false
        }
        Err(e) => return Err(e),
    };
    if !downloaded && !local_bake_ok {
        bail!(
            "no {variant} bake at {} and download unavailable. Low-bit baking \
             must happen offline — run `python oracle/bake_q4km.py --model-dir {} --gguf-file /path/to/model.gguf --out-dir {}` \
             for q4km or `python oracle/bake_int4.py --model-dir {}` for GPTQ INT4, \
             then rerun without --no-download to fetch from the GitHub bakes release.",
            bake_dir.display(),
            cfg.model_dir.display(),
            bake_dir.display(),
            cfg.model_dir.display(),
        );
    }
    if bake_ok() || !local_bake_ok {
        return Ok(());
    }

    tracing::info!("baking Qwen3.5 {} weights (one-time)...", variant);
    let bake_start = Instant::now();
    let layer_is_full: Vec<bool> = (0..text_config.num_hidden_layers)
        .map(|i| text_config.is_full_attention(i))
        .collect();
    model_store::bake_qwen35(
        &cfg.model_dir,
        weight_prefix,
        text_config.num_hidden_layers,
        &layer_is_full,
        cfg.fp8_runtime,
        &|m| tracing::info!("{m}"),
    )
    .map_err(|e| anyhow!("bake weights: {e}"))?;
    tracing::info!("bake done in {:.1}s", bake_start.elapsed().as_secs_f64());
    Ok(())
}

/// Attempt to fetch the requested bake from the GitHub `bakes-v{FORMAT_VERSION}`
/// release. Returns `Ok(true)` on success, `Ok(false)` when download is
/// disabled via `--no-download`, `Err(_)` when the fetch itself failed.
pub(crate) fn try_download_bake(
    cfg: &LoaderConfig,
    variant: model_store::fetch::BakeVariant,
    target_bake_dir: &std::path::Path,
) -> Result<bool> {
    if cfg.no_download {
        return Ok(false);
    }
    let source = model_store::fetch::ReleaseSource::default_for_format_version();
    tracing::info!(
        "downloading {} bake for {} from bakes-v{} release...",
        variant,
        cfg.model,
        model_store::manifest::FORMAT_VERSION
    );
    let req = model_store::fetch::FetchRequest {
        source: &source,
        model_cli_name: &cfg.model,
        variant,
        target_bake_dir,
        target_model_dir: &cfg.model_dir,
        progress: &fetch_progress_logger(),
    };
    model_store::fetch::fetch_bake(req).map_err(|e| anyhow!("fetch {} bake: {}", variant, e))?;
    Ok(true)
}

fn fetch_progress_logger() -> impl Fn(model_store::fetch::FetchProgress) {
    use std::cell::Cell;
    let last_pct = Cell::new(i32::MIN);
    let last_part = Cell::new(u32::MAX);
    move |p| {
        use model_store::fetch::FetchProgress::*;
        match p {
            ResolvingIndex => tracing::info!("[fetch] resolving release index..."),
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
                    tracing::info!(
                        "[fetch] part {}/{} — {} MiB",
                        part + 1,
                        total_parts,
                        bytes_total / (1024 * 1024)
                    );
                }
                if pct >= last_pct.get() + 10 {
                    last_pct.set(pct);
                    tracing::info!(
                        "[fetch]   {}% ({} / {} MiB)",
                        pct,
                        bytes_done / (1024 * 1024),
                        bytes_total / (1024 * 1024)
                    );
                }
            }
            Verifying => tracing::info!("[fetch] verifying checksums..."),
            Extracting => tracing::info!("[fetch] extracting archive..."),
            Done => tracing::info!("[fetch] done"),
        }
    }
}
