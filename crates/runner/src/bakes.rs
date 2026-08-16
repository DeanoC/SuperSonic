use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use model_store::manifest::QuantProfile;

use crate::flm_model_source::{is_flm_model_path, FlmModelSource, FlmModelSourceOptions};
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
pub(crate) fn effective_quant_profile(cli: &Cli) -> Result<QuantProfile> {
    let mut selected: Option<QuantProfile> = None;
    let mut set = |profile: QuantProfile, source: &str| -> Result<()> {
        if let Some(prev) = selected {
            if prev != profile {
                anyhow::bail!(
                    "{source} selects {profile}, but another quant flag already selected {prev}"
                );
            }
        } else {
            selected = Some(profile);
        }
        Ok(())
    };
    if let Some(raw) = cli.weight_quant.as_deref() {
        let profile = raw
            .parse::<QuantProfile>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        set(profile, "--weight-quant")?;
    }
    if cli.int4 {
        set(QuantProfile::Int4Gptq, "--int4")?;
    }
    if cli.fp8_runtime {
        set(QuantProfile::Fp8Native, "--fp8-runtime")?;
    }
    if cli.q4km {
        set(QuantProfile::Q4Km, "--q4km")?;
    }
    if cli.q4km_gptq {
        set(QuantProfile::Q4KmGptq, "--q4km-gptq")?;
    }
    Ok(selected.unwrap_or(QuantProfile::Bf16))
}

pub(crate) fn cli_variant(cli: &Cli) -> Result<model_store::fetch::BakeVariant> {
    Ok(model_store::fetch::variant_from_quant_profile(
        effective_quant_profile(cli)?,
    ))
}

pub(crate) fn variant_version_ok(
    variant: model_store::fetch::BakeVariant,
    bake_dir: &Path,
) -> bool {
    model_store::fetch::version_ok_for_variant(variant, bake_dir)
}

fn quant_bake_method_note(profile: QuantProfile) -> &'static str {
    match profile {
        QuantProfile::Int4Hqq => "INT4-HQQ baking is data-free and runs in Python.",
        QuantProfile::Int4Gptq => "INT4-GPTQ baking requires a calibration pass in Python.",
        QuantProfile::Int4Awq => {
            "INT4-AWQ baking requires activation statistics and runs in Python."
        }
        QuantProfile::Int4Autoround => {
            "INT4-AutoRound baking requires activation samples and a Python rounding-optimization pass."
        }
        _ => "This quant profile requires an external Python bake.",
    }
}

fn upload_bake_args(profile: QuantProfile) -> String {
    match profile {
        QuantProfile::Int4Gptq => "--int4".to_string(),
        other => format!("--weight-quant {other}"),
    }
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

pub(crate) fn effective_flm_source(cli: &Cli) -> Option<&Path> {
    cli.flm_file
        .as_deref()
        .or_else(|| is_flm_model_path(&cli.model_dir).then_some(cli.model_dir.as_path()))
}

pub(crate) fn cli_args_include_model<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    args.into_iter().any(|arg| {
        let arg = arg.as_ref();
        arg == OsStr::new("--model")
            || arg
                .to_str()
                .map(|arg| arg.starts_with("--model="))
                .unwrap_or(false)
    })
}

fn parse_cli_model_variant(model: &str) -> Result<ModelVariant> {
    ModelVariant::from_cli_str(model).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown model '{}'. Supported models: {}",
            model,
            crate::registry::supported_models_list().join(", ")
        )
    })
}

pub(crate) fn model_variant_from_flm_identity(
    identity: model_store::FlmRuntimeIdentity,
) -> Option<ModelVariant> {
    match (identity.architecture_id, identity.model_id) {
        (model_store::flm::ARCH_QWEN3_6_DENSE, model_store::flm::MODEL_QWEN3_6_DENSE_V1) => {
            Some(ModelVariant::Qwen3_6_27B)
        }
        (model_store::flm::ARCH_QWEN3_6_MOE, model_store::flm::MODEL_QWEN3_6_MOE_V1) => {
            Some(ModelVariant::Qwen3_6_35B_A3B)
        }
        _ => None,
    }
}

pub(crate) fn resolve_model_variant(cli: &Cli, model_arg_present: bool) -> Result<ModelVariant> {
    if model_arg_present {
        return parse_cli_model_variant(&cli.model);
    }

    if let Some(flm_source) = effective_flm_source(cli) {
        let identity = model_store::read_flm_runtime_identity(flm_source)
            .map_err(|e| anyhow::anyhow!("read FLM runtime identity: {e}"))?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "FLM source {} has no runtime descriptor; pass --model explicitly",
                    flm_source.display()
                )
            })?;
        let model_variant = model_variant_from_flm_identity(identity).ok_or_else(|| {
            anyhow::anyhow!(
                "FLM source {} has unsupported runtime identity architecture_id={} model_id={}",
                flm_source.display(),
                identity.architecture_id,
                identity.model_id
            )
        })?;
        eprintln!("[flm] inferred model {model_variant} from runtime descriptor");
        return Ok(model_variant);
    }

    parse_cli_model_variant(&cli.model)
}

pub(crate) fn flm_source_is_authoritative_for_model(
    cli: &Cli,
    model_variant: &ModelVariant,
) -> bool {
    matches!(
        model_variant,
        ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_6_35B_A3B
    ) && effective_flm_source(cli).is_some()
}

pub(crate) fn validate_effective_flm_source_model(
    cli: &Cli,
    model_variant: &ModelVariant,
) -> Result<()> {
    let Some(flm_source) = effective_flm_source(cli) else {
        return Ok(());
    };
    if matches!(
        model_variant,
        ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_6_35B_A3B
    ) {
        return Ok(());
    }
    let source_flag = if cli.flm_file.is_some() {
        "--flm-file"
    } else {
        "--model-dir"
    };
    anyhow::bail!(
        "FLM source from {source_flag} {} currently supports only --model qwen3.6-27b or qwen3.6-35b-a3b; got --model {}",
        flm_source.display(),
        model_variant
    );
}

pub(crate) fn validate_flm_weight_source_options(cli: &Cli, q4km_like: bool) -> Result<()> {
    if effective_flm_source(cli).is_none() {
        return Ok(());
    }
    if cli.no_bake {
        anyhow::bail!("FLM sources and --no-bake are mutually exclusive");
    }
    if q4km_like {
        anyhow::bail!("FLM sources are not wired for --q4km/--q4km-gptq bakes");
    }
    if cli.int8 {
        anyhow::bail!("FLM sources are not wired for --int8 bakes");
    }
    Ok(())
}

pub(crate) fn flm_source_open_options(cli: &Cli) -> Result<FlmModelSourceOptions> {
    let profile = effective_quant_profile(cli)?;
    Ok(FlmModelSourceOptions {
        int4_runtime: effective_flm_source(cli).is_some() || profile.is_native_int4_runtime(),
        verify_block_hashes: cli.verify_flm_hashes,
    })
}

pub(crate) fn load_qwen35_weights_from_flm_source(
    cli: &Cli,
    model_variant: &ModelVariant,
    text_config: &qwen35::config::TextConfig,
    ordinal: usize,
    weight_prefix: &str,
    q4km_like: bool,
    source: &FlmModelSource,
) -> Result<qwen35::weights::Qwen35Weights> {
    validate_effective_flm_source_model(cli, model_variant)?;
    validate_flm_weight_source_options(cli, q4km_like)?;

    eprintln!(
        "[weights] loading FLM weights from already-open source at {}",
        source.path.display()
    );
    qwen35::weights::Qwen35Weights::load_baked(&source.store, text_config, ordinal, weight_prefix)
        .map_err(|e| anyhow::anyhow!("load FLM weights: {e}"))
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
    validate_effective_flm_source_model(cli, model_variant)?;
    validate_flm_weight_source_options(cli, q4km_like)?;

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

    if let Some(flm_file) = effective_flm_source(cli) {
        let options = flm_source_open_options(cli)?;
        eprintln!(
            "[flm] opening model source at {}{}{}",
            flm_file.display(),
            if options.int4_runtime {
                " (FLM logical INT4 aliases enabled)"
            } else {
                ""
            },
            if options.verify_block_hashes {
                " (BLAKE3 hash verification enabled)"
            } else {
                ""
            }
        );
        let source = FlmModelSource::open_with_options(flm_file, options)
            .map_err(|e| anyhow::anyhow!("open FLM store: {e}"))?;
        return load_qwen35_weights_from_flm_source(
            cli,
            model_variant,
            text_config,
            ordinal,
            weight_prefix,
            q4km_like,
            &source,
        );
    }

    let profile = effective_quant_profile(cli)?;
    let variant = model_store::fetch::variant_from_quant_profile(profile);
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
                             {}\n\
                             Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {} --profile {}",
                            bake_dir.display(),
                            quant_bake_method_note(profile),
                            cli.model_dir.display(),
                            profile,
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
                         {} \
                         Run on a bigger machine:\n  python oracle/bake_int4.py --model-dir {} --profile {}\n\
                         then `python oracle/upload_bake.py --model {} {} --model-dir {}` to publish.",
                        quant_bake_method_note(profile),
                        cli.model_dir.display(),
                        profile,
                        cli.model,
                        upload_bake_args(profile),
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
    if flm_source_is_authoritative_for_model(cli, model_variant) {
        return Ok(false);
    }
    if cli.no_bake || cli.no_download {
        return Ok(false);
    }
    if cli.model_dir.join("config.json").exists() {
        return Ok(false);
    }
    let variant = cli_variant(cli)?;
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
    use std::path::Path;

    use clap::Parser;
    use model_store::manifest::QuantProfile;

    use super::{
        cli_args_include_model, effective_flm_source, effective_quant_profile,
        ensure_hf_metadata_present, flm_source_is_authoritative_for_model, flm_source_open_options,
        model_variant_from_flm_identity, should_fetch_bake, should_fetch_exact_bake,
        validate_effective_flm_source_model, validate_flm_weight_source_options,
    };
    use crate::registry::ModelVariant;
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
    fn cli_args_include_model_detects_separate_and_equals_forms() {
        assert!(cli_args_include_model([
            "supersonic",
            "--model",
            "qwen3.6-27b"
        ]));
        assert!(cli_args_include_model([
            "supersonic",
            "--model=qwen3.6-35b-a3b"
        ]));
        assert!(!cli_args_include_model([
            "supersonic",
            "--model-dir",
            "/tmp/model.flm"
        ]));
    }

    #[test]
    fn flm_runtime_identity_selects_model_variant() {
        assert_eq!(
            model_variant_from_flm_identity(model_store::FlmRuntimeIdentity {
                architecture_id: model_store::flm::ARCH_QWEN3_6_DENSE,
                model_id: model_store::flm::MODEL_QWEN3_6_DENSE_V1,
            }),
            Some(ModelVariant::Qwen3_6_27B)
        );
        assert_eq!(
            model_variant_from_flm_identity(model_store::FlmRuntimeIdentity {
                architecture_id: model_store::flm::ARCH_QWEN3_6_MOE,
                model_id: model_store::flm::MODEL_QWEN3_6_MOE_V1,
            }),
            Some(ModelVariant::Qwen3_6_35B_A3B)
        );
    }

    #[test]
    fn flm_runtime_identity_rejects_mismatched_model_id() {
        assert_eq!(
            model_variant_from_flm_identity(model_store::FlmRuntimeIdentity {
                architecture_id: model_store::flm::ARCH_QWEN3_6_MOE,
                model_id: model_store::flm::MODEL_QWEN3_6_DENSE_V1,
            }),
            None
        );
    }

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

    #[test]
    fn flm_file_is_authoritative_for_hf_metadata_bootstrap() {
        assert!(flm_source_is_authoritative_for_model(
            &cli(&["--flm-file", "/tmp/model.flm"]),
            &ModelVariant::Qwen3_6_27B
        ));
    }

    #[test]
    fn flm_model_dir_is_authoritative_for_hf_metadata_bootstrap() {
        assert!(flm_source_is_authoritative_for_model(
            &cli_with_model_dir("/tmp/model.flm", &[]),
            &ModelVariant::Qwen3_6_27B
        ));
    }

    #[test]
    fn qwen36_flm_model_dir_skips_hf_metadata_bootstrap_even_without_config() {
        let cli = cli_with_model_dir("/tmp/qwen36-27b-no-hf.flm", &["--verify-flm-hashes"]);

        let downloaded = ensure_hf_metadata_present(&cli, &ModelVariant::Qwen3_6_27B)
            .expect("authoritative FLM source should bypass HF metadata bootstrap");

        assert!(!downloaded);
    }

    #[test]
    fn flm_model_dir_is_authoritative_for_qwen36_moe_hf_metadata_bootstrap() {
        assert!(flm_source_is_authoritative_for_model(
            &cli_with_model_dir("/tmp/qwen36-35b-a3b.flm", &[]),
            &ModelVariant::Qwen3_6_35B_A3B
        ));
    }

    #[test]
    fn qwen36_moe_flm_model_dir_skips_hf_metadata_bootstrap_even_without_config() {
        let cli = cli_with_model_dir("/tmp/qwen36-35b-a3b-no-hf.flm", &["--verify-flm-hashes"]);

        let downloaded = ensure_hf_metadata_present(&cli, &ModelVariant::Qwen3_6_35B_A3B)
            .expect("authoritative MoE FLM source should bypass HF metadata bootstrap");

        assert!(!downloaded);
    }

    #[test]
    fn effective_flm_source_accepts_qwen36_moe_model_variant() {
        validate_effective_flm_source_model(
            &cli_with_model_dir("/tmp/qwen36-35b-a3b.flm", &[]),
            &ModelVariant::Qwen3_6_35B_A3B,
        )
        .expect("qwen3.6-35b-a3b FLM should be accepted");
    }

    #[test]
    fn flm_file_is_not_authoritative_for_gemma_hf_metadata_bootstrap() {
        assert!(!flm_source_is_authoritative_for_model(
            &cli(&["--flm-file", "/tmp/model.flm"]),
            &ModelVariant::Gemma4_E2B
        ));
    }

    #[test]
    fn flm_model_dir_is_not_authoritative_for_gemma_hf_metadata_bootstrap() {
        assert!(!flm_source_is_authoritative_for_model(
            &cli_with_model_dir("/tmp/model.flm", &[]),
            &ModelVariant::Gemma4_E2B
        ));
    }

    #[test]
    fn effective_flm_source_prefers_explicit_flm_file() {
        let cli = cli_with_model_dir("/tmp/model-dir.flm", &["--flm-file", "/tmp/explicit.flm"]);

        assert_eq!(
            effective_flm_source(&cli),
            Some(Path::new("/tmp/explicit.flm"))
        );
    }

    #[test]
    fn effective_flm_source_uses_flm_model_dir_without_explicit_flm_file() {
        let cli = cli_with_model_dir("/tmp/model-dir.flm", &[]);

        assert_eq!(
            effective_flm_source(&cli),
            Some(Path::new("/tmp/model-dir.flm"))
        );
    }

    #[test]
    fn flm_source_open_options_enable_hash_verification_for_single_source() {
        let cli = cli_with_model_dir("/tmp/model.flm", &["--int4", "--verify-flm-hashes"]);

        let options = flm_source_open_options(&cli).expect("valid FLM source options");

        assert!(options.int4_runtime);
        assert!(options.verify_block_hashes);
    }

    #[test]
    fn flm_source_open_options_keep_hash_verification_opt_in() {
        let cli = cli_with_model_dir("/tmp/model.flm", &[]);

        let options = flm_source_open_options(&cli).expect("valid FLM source options");

        assert!(options.int4_runtime);
        assert!(!options.verify_block_hashes);
    }

    #[test]
    fn effective_flm_source_requires_qwen36_27b_model_variant() {
        let err = validate_effective_flm_source_model(
            &cli(&["--flm-file", "/tmp/model.flm"]),
            &ModelVariant::Qwen3_5_0_8B,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("--flm-file"), "{err}");
        assert!(err.contains("qwen3.6-27b"), "{err}");
        assert!(err.contains("qwen3.5-0.8b"), "{err}");
    }

    #[test]
    fn flm_model_dir_no_bake_is_rejected_by_weight_source_options() {
        let err = validate_flm_weight_source_options(
            &cli_with_model_dir("/tmp/model.flm", &["--no-bake"]),
            false,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("FLM"), "{err}");
        assert!(err.contains("--no-bake"), "{err}");
    }

    #[test]
    fn weight_quant_selects_canonical_profile() {
        assert_eq!(
            effective_quant_profile(&cli(&["--weight-quant", "hqq"])).unwrap(),
            QuantProfile::Int4Hqq
        );
    }

    #[test]
    fn legacy_int4_alias_matches_gptq_profile() {
        assert_eq!(
            effective_quant_profile(&cli(&["--int4"])).unwrap(),
            QuantProfile::Int4Gptq
        );
    }

    #[test]
    fn conflicting_quant_flags_are_rejected() {
        let err = effective_quant_profile(&cli(&["--weight-quant", "int4-awq", "--int4"]))
            .expect_err("conflicting quant flags should fail")
            .to_string();
        assert!(err.contains("--int4 selects int4-gptq"));
    }
}
