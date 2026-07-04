use anyhow::{anyhow, Result};
use model_store::manifest::LayoutTag;
use model_store::BakedStore;

use crate::bakes::{
    effective_flm_source, flm_source_open_options, validate_effective_flm_source_model,
};
use crate::flm_model_source::FlmModelSource;
use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
use crate::registry::ModelVariant;
use crate::Cli;

const QWEN36_MOE_WEIGHT_MODE_PROBE: &str =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";

pub(crate) struct Qwen36MoeFlmSource {
    pub(crate) source: FlmModelSource,
    pub(crate) config: qwen36_moe::config::Config,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) weight_mode: Qwen36WeightMode,
    pub(crate) weight_mode_label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen36MoeFlmWeightSelection {
    mode: Qwen36WeightMode,
    label: &'static str,
}

#[cfg(test)]
fn qwen36_moe_flm_weight_mode(probe: Option<(&LayoutTag, &str)>) -> Result<Qwen36WeightMode> {
    Ok(qwen36_moe_flm_weight_selection_from_probe(probe)?.mode)
}

fn qwen36_moe_flm_weight_selection_for_store(
    store: &BakedStore,
) -> Result<Qwen36MoeFlmWeightSelection> {
    let probe = store
        .meta(QWEN36_MOE_WEIGHT_MODE_PROBE)
        .map(|meta| (&meta.layout, meta.dtype.as_str()));
    qwen36_moe_flm_weight_selection_from_probe(probe)
}

fn qwen36_moe_flm_weight_selection_from_probe(
    probe: Option<(&LayoutTag, &str)>,
) -> Result<Qwen36MoeFlmWeightSelection> {
    match probe {
        Some((LayoutTag::Raw, "bf16")) => Ok(Qwen36MoeFlmWeightSelection {
            mode: Qwen36WeightMode::Bf16,
            label: "BF16",
        }),
        Some((LayoutTag::Int4Quantized, "u8")) | None => Ok(Qwen36MoeFlmWeightSelection {
            mode: Qwen36WeightMode::Int4,
            label: "INT4 native FLM",
        }),
        other => Err(anyhow!(
            "Qwen3.6 MoE FLM source requires a native INT4 or BF16 fallback probe; got {other:?}"
        )),
    }
}

pub(crate) fn open_qwen36_moe_flm_source(cli: &Cli) -> Result<Option<Qwen36MoeFlmSource>> {
    let Some(path) = effective_flm_source(cli) else {
        return Ok(None);
    };
    validate_effective_flm_source_model(cli, &ModelVariant::Qwen3_6_35B_A3B)?;
    let options = flm_source_open_options(cli)?;
    eprintln!(
        "[flm] opening model source at {}{}{}",
        path.display(),
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
    let source = FlmModelSource::open_with_options(path, options)
        .map_err(|e| anyhow!("opening Qwen3.6 MoE FLM source {}: {e}", path.display()))?;
    eprintln!("[qwen36-moe] loading config from FLM runtime descriptor");
    let config = source.qwen_moe_config()?;
    eprintln!("[qwen36-moe] loading tokenizer from FLM assets");
    let tokenizer = source.qwen_tokenizer()?;
    let weight_selection = qwen36_moe_flm_weight_selection_for_store(&source.store)?;
    eprintln!("[qwen36-moe] FLM weight mode: {}", weight_selection.label);
    Ok(Some(Qwen36MoeFlmSource {
        source,
        config,
        tokenizer,
        weight_mode: weight_selection.mode,
        weight_mode_label: weight_selection.label,
    }))
}

#[cfg(test)]
mod tests {
    use model_store::manifest::LayoutTag;

    use super::*;

    #[test]
    fn maps_moe_mixed_lowbit_flm_to_int4_weight_mode() {
        let mode =
            qwen36_moe_flm_weight_mode(None).expect("missing probe fixtures default to INT4");

        assert_eq!(mode, Qwen36WeightMode::Int4);
    }

    #[test]
    fn maps_ct_int4_bf16_fallback_probe_to_bf16_weight_mode() {
        let selection = qwen36_moe_flm_weight_selection_from_probe(Some((&LayoutTag::Raw, "bf16")))
            .expect("CT INT4 fallback is supported through BF16 load");

        assert_eq!(selection.mode, Qwen36WeightMode::Bf16);
        assert_eq!(selection.label, "BF16");
    }

    #[test]
    fn keeps_native_int4_probe_on_int4_weight_mode() {
        let selection =
            qwen36_moe_flm_weight_selection_from_probe(Some((&LayoutTag::Int4Quantized, "u8")))
                .expect("native INT4 FLM payload is supported");

        assert_eq!(selection.mode, Qwen36WeightMode::Int4);
        assert_eq!(selection.label, "INT4 native FLM");
    }

    #[test]
    fn native_int4_probe_overrides_default_bf16_cli_profile() {
        let selection =
            qwen36_moe_flm_weight_selection_from_probe(Some((&LayoutTag::Int4Quantized, "u8")))
                .expect("native INT4 FLM payload should select native mode without --int4");

        assert_eq!(selection.mode, Qwen36WeightMode::Int4);
        assert_eq!(selection.label, "INT4 native FLM");
    }

    #[test]
    fn raw_bf16_probe_selects_fallback_without_int4_cli_profile() {
        let selection = qwen36_moe_flm_weight_selection_from_probe(Some((&LayoutTag::Raw, "bf16")))
            .expect("BF16 FLM fallback should be file-driven without --int4");

        assert_eq!(selection.mode, Qwen36WeightMode::Bf16);
        assert_eq!(selection.label, "BF16");
    }

    #[test]
    fn rejects_unknown_flm_weight_mode_probe_for_this_stage() {
        let err = qwen36_moe_flm_weight_mode(Some((&LayoutTag::Raw, "u8")))
            .unwrap_err()
            .to_string();

        assert!(err.contains("Qwen3.6 MoE FLM"), "{err}");
        assert!(err.contains("INT4"), "{err}");
        assert!(err.contains("Raw"), "{err}");
    }
}
