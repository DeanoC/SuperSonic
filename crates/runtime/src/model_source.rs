use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use model_store::FlmRuntimeIdentity;
use supersonic_core::registry::{self, ModelVariant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    Directory(PathBuf),
    Flm(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelSource {
    pub source: ModelSource,
    pub model: ModelVariant,
}

pub fn resolve_model_source(
    flm_file: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    explicit_model: Option<&str>,
) -> Result<ResolvedModelSource> {
    resolve_model_source_with(
        flm_file,
        model_dir,
        explicit_model,
        model_store::read_flm_runtime_identity,
    )
}

fn resolve_model_source_with(
    flm_file: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    explicit_model: Option<&str>,
    read_identity: impl FnOnce(&Path) -> Result<Option<FlmRuntimeIdentity>, model_store::Error>,
) -> Result<ResolvedModelSource> {
    if flm_file.is_some() && model_dir.is_some() {
        bail!("--flm-file and --model-dir are mutually exclusive");
    }

    let source = match (flm_file, model_dir) {
        (Some(path), None) => ModelSource::Flm(path),
        (None, Some(path)) if crate::flm_model_source::is_flm_model_path(&path) => {
            ModelSource::Flm(path)
        }
        (None, Some(path)) => ModelSource::Directory(path),
        (None, None) => bail!("one of --flm-file or --model-dir is required"),
        (Some(_), Some(_)) => unreachable!("mutual exclusion checked above"),
    };

    match source {
        ModelSource::Directory(path) => {
            let model = explicit_model
                .ok_or_else(|| anyhow!("--model is required with a directory --model-dir"))
                .and_then(parse_model_variant)?;
            Ok(ResolvedModelSource {
                source: ModelSource::Directory(path),
                model,
            })
        }
        ModelSource::Flm(path) => {
            let identity = read_identity(&path)
                .with_context(|| format!("read FLM runtime descriptor from {}", path.display()))?
                .ok_or_else(|| {
                    anyhow!("FLM source {} has no runtime descriptor", path.display())
                })?;
            let model = model_variant_from_flm_identity(identity).ok_or_else(|| {
                anyhow!(
                    "FLM source {} has unsupported runtime descriptor architecture_id={} model_id={}",
                    path.display(),
                    identity.architecture_id,
                    identity.model_id
                )
            })?;
            if let Some(explicit_model) = explicit_model {
                let explicit = parse_model_variant(explicit_model)?;
                if explicit != model {
                    if same_dense_hybrid_geometry(explicit, model) {
                        // GGUF→FLM converters emit the dense-hybrid architecture id.
                        // `--model qwen3.8-27b` selects the 3.8 registry row.
                        return Ok(ResolvedModelSource {
                            source: ModelSource::Flm(path),
                            model: explicit,
                        });
                    }
                    bail!(
                        "--model {} does not match FLM runtime descriptor model {}",
                        explicit,
                        model
                    );
                }
            }
            Ok(ResolvedModelSource {
                source: ModelSource::Flm(path),
                model,
            })
        }
    }
}

fn parse_model_variant(model: &str) -> Result<ModelVariant> {
    ModelVariant::from_cli_str(model).ok_or_else(|| {
        anyhow!(
            "unknown --model '{}' (supported: {})",
            model,
            registry::supported_models_list().join(", ")
        )
    })
}

fn model_variant_from_flm_identity(identity: FlmRuntimeIdentity) -> Option<ModelVariant> {
    match (identity.architecture_id, identity.model_id) {
        (model_store::flm::ARCH_QWEN3_6_MOE, model_store::flm::MODEL_QWEN3_6_MOE_V1) => {
            Some(ModelVariant::Qwen3_6_35B_A3B)
        }
        (model_store::flm::ARCH_QWEN3_6_DENSE, model_store::flm::MODEL_QWEN3_8_DENSE_V1) => {
            Some(ModelVariant::Qwen3_8_27B)
        }
        (model_store::flm::ARCH_QWEN3_6_DENSE, _) => Some(ModelVariant::Qwen3_6_27B),
        _ => None,
    }
}

fn same_dense_hybrid_geometry(a: ModelVariant, b: ModelVariant) -> bool {
    matches!(
        (a, b),
        (
            ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_8_27B,
            ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_8_27B
        )
    )
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
    use model_store::FlmRuntimeIdentity;
    use supersonic_core::registry::ModelVariant;

    use super::{resolve_model_source_with, ModelSource};

    fn qwen36_moe_identity() -> FlmRuntimeIdentity {
        FlmRuntimeIdentity {
            architecture_id: ARCH_QWEN3_6_MOE,
            model_id: MODEL_QWEN3_6_MOE_V1,
        }
    }

    #[test]
    fn model_source_selects_explicit_flm_file_and_infers_model() {
        let resolved = resolve_model_source_with(
            Some(PathBuf::from("/models/qwen36.flm")),
            None,
            None,
            |path| {
                assert_eq!(path, Path::new("/models/qwen36.flm"));
                Ok(Some(qwen36_moe_identity()))
            },
        )
        .expect("descriptor-backed FLM source");

        assert_eq!(
            resolved.source,
            ModelSource::Flm(PathBuf::from("/models/qwen36.flm"))
        );
        assert_eq!(resolved.model, ModelVariant::Qwen3_6_35B_A3B);
    }

    #[test]
    fn model_source_normalizes_file_valued_model_dir() {
        let resolved = resolve_model_source_with(
            None,
            Some(PathBuf::from("/models/compat.FLM")),
            None,
            |_| Ok(Some(qwen36_moe_identity())),
        )
        .expect("file-valued --model-dir");

        assert_eq!(
            resolved.source,
            ModelSource::Flm(PathBuf::from("/models/compat.FLM"))
        );
        assert_eq!(resolved.model, ModelVariant::Qwen3_6_35B_A3B);
    }

    #[test]
    fn model_source_rejects_multiple_source_arguments() {
        let err = resolve_model_source_with(
            Some(PathBuf::from("/models/explicit.flm")),
            Some(PathBuf::from("/models/directory")),
            None,
            |_| Ok(Some(qwen36_moe_identity())),
        )
        .expect_err("two source arguments must fail")
        .to_string();

        assert!(err.contains("--flm-file"), "{err}");
        assert!(err.contains("--model-dir"), "{err}");
        assert!(err.contains("mutually exclusive"), "{err}");
    }

    #[test]
    fn model_source_directory_requires_explicit_model() {
        let err =
            resolve_model_source_with(None, Some(PathBuf::from("/models/directory")), None, |_| {
                panic!("directory startup must not inspect an FLM descriptor")
            })
            .expect_err("directory startup without --model must fail")
            .to_string();

        assert!(err.contains("--model"), "{err}");
        assert!(err.contains("directory"), "{err}");
    }

    #[test]
    fn model_source_directory_preserves_legacy_selection() {
        let resolved = resolve_model_source_with(
            None,
            Some(PathBuf::from("/models/directory")),
            Some("qwen3.5-0.8b"),
            |_| panic!("directory startup must not inspect an FLM descriptor"),
        )
        .expect("legacy directory source");

        assert_eq!(
            resolved.source,
            ModelSource::Directory(PathBuf::from("/models/directory"))
        );
        assert_eq!(resolved.model, ModelVariant::Qwen3_5_0_8B);
    }

    #[test]
    fn model_source_accepts_model_matching_flm_descriptor() {
        let resolved = resolve_model_source_with(
            Some(PathBuf::from("/models/qwen36.flm")),
            None,
            Some("qwen3.6-35b-a3b"),
            |_| Ok(Some(qwen36_moe_identity())),
        )
        .expect("matching --model");

        assert_eq!(resolved.model, ModelVariant::Qwen3_6_35B_A3B);
    }

    #[test]
    fn model_source_rejects_model_mismatching_flm_descriptor() {
        let err = resolve_model_source_with(
            Some(PathBuf::from("/models/qwen36.flm")),
            None,
            Some("qwen3.5-0.8b"),
            |_| Ok(Some(qwen36_moe_identity())),
        )
        .expect_err("mismatching --model must fail")
        .to_string();

        assert!(err.contains("qwen3.5-0.8b"), "{err}");
        assert!(err.contains("qwen3.6-35b-a3b"), "{err}");
        assert!(err.contains("descriptor"), "{err}");
    }

    #[test]
    fn model_source_rejects_flm_without_runtime_descriptor() {
        let err = resolve_model_source_with(
            Some(PathBuf::from("/models/no-runtime.flm")),
            None,
            None,
            |_| Ok(None),
        )
        .expect_err("missing runtime descriptor must fail")
        .to_string();

        assert!(err.contains("runtime descriptor"), "{err}");
        assert!(err.contains("no-runtime.flm"), "{err}");
    }
}
