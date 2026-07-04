use anyhow::{anyhow, Result};
use model_store::manifest::{LayoutTag, QuantProfile};
use model_store::BakedStore;

use crate::bakes::{
    effective_flm_source, effective_quant_profile, flm_source_open_options,
    validate_effective_flm_source_model,
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
}

#[cfg(test)]
fn qwen36_moe_flm_weight_mode(profile: QuantProfile) -> Result<Qwen36WeightMode> {
    qwen36_moe_flm_weight_mode_from_probe(profile, None)
}

fn qwen36_moe_flm_weight_mode_for_store(
    store: &BakedStore,
    profile: QuantProfile,
) -> Result<Qwen36WeightMode> {
    let probe = store
        .meta(QWEN36_MOE_WEIGHT_MODE_PROBE)
        .map(|meta| (&meta.layout, meta.dtype.as_str()));
    qwen36_moe_flm_weight_mode_from_probe(profile, probe)
}

fn qwen36_moe_flm_weight_mode_from_probe(
    profile: QuantProfile,
    probe: Option<(&LayoutTag, &str)>,
) -> Result<Qwen36WeightMode> {
    match profile {
        QuantProfile::Int4Gptq
        | QuantProfile::Int4Awq
        | QuantProfile::Int4Autoround
        | QuantProfile::Int4Hqq => {
            if matches!(probe, Some((LayoutTag::Raw, "bf16"))) {
                Ok(Qwen36WeightMode::Bf16)
            } else {
                Ok(Qwen36WeightMode::Int4)
            }
        }
        other => Err(anyhow!(
            "Qwen3.6 MoE FLM main path currently supports only INT4-compatible profiles; got {other}"
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
    let weight_mode =
        qwen36_moe_flm_weight_mode_for_store(&source.store, effective_quant_profile(cli)?)?;
    eprintln!(
        "[qwen36-moe] FLM weight mode: {}",
        weight_mode.display_name()
    );
    Ok(Some(Qwen36MoeFlmSource {
        source,
        config,
        tokenizer,
        weight_mode,
    }))
}

#[cfg(test)]
mod tests {
    use model_store::manifest::{LayoutTag, QuantProfile};

    use super::*;

    #[test]
    fn maps_moe_mixed_lowbit_flm_to_int4_weight_mode() {
        let mode =
            qwen36_moe_flm_weight_mode(QuantProfile::Int4Gptq).expect("INT4 profile is supported");

        assert_eq!(mode, Qwen36WeightMode::Int4);
    }

    #[test]
    fn maps_ct_int4_bf16_fallback_probe_to_bf16_weight_mode() {
        let mode = qwen36_moe_flm_weight_mode_from_probe(
            QuantProfile::Int4Gptq,
            Some((&LayoutTag::Raw, "bf16")),
        )
        .expect("CT INT4 fallback is supported through BF16 load");

        assert_eq!(mode, Qwen36WeightMode::Bf16);
    }

    #[test]
    fn keeps_native_int4_probe_on_int4_weight_mode() {
        let mode = qwen36_moe_flm_weight_mode_from_probe(
            QuantProfile::Int4Gptq,
            Some((&LayoutTag::Raw, "u8")),
        )
        .expect("native INT4 FLM payload is supported");

        assert_eq!(mode, Qwen36WeightMode::Int4);
    }

    #[test]
    fn rejects_q4km_flm_weight_mode_for_this_stage() {
        let err = qwen36_moe_flm_weight_mode(QuantProfile::Q4Km)
            .unwrap_err()
            .to_string();

        assert!(err.contains("Qwen3.6 MoE FLM"), "{err}");
        assert!(err.contains("INT4"), "{err}");
    }
}
