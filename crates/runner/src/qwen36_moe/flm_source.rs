use anyhow::{anyhow, Result};
use model_store::flm::FlmStage3DirectWeightKind;
#[cfg(test)]
use model_store::manifest::LayoutTag;
use model_store::BakedStore;
use qwen36_moe::weights::{
    expected_tensor_specs, TensorRole, DEFAULT_PREFIX as QWEN36_MOE_WEIGHT_PREFIX,
};

use crate::bakes::{
    effective_flm_source, flm_source_open_options, validate_effective_flm_source_model,
    validate_flm_weight_source_options,
};
use crate::flm_model_source::FlmModelSource;
use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
use crate::registry::ModelVariant;
use crate::Cli;

#[cfg(test)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequiredFlmDirectWeightKind {
    RawDense,
    QuantizedProjection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequiredFlmDirectWeight {
    name: String,
    expected: RequiredFlmDirectWeightKind,
}

#[cfg(test)]
fn qwen36_moe_flm_weight_mode(probe: Option<(&LayoutTag, &str)>) -> Result<Qwen36WeightMode> {
    Ok(qwen36_moe_flm_weight_selection_from_probe(probe)?.mode)
}

fn qwen36_moe_flm_weight_selection_for_store(
    store: &BakedStore,
    text_config: &qwen36_moe::config::TextConfig,
) -> Result<Qwen36MoeFlmWeightSelection> {
    let runtime = store.flm_runtime().ok_or_else(|| {
        anyhow!(
            "Qwen3.6 MoE FLM source requires runtime direct weight plans for SuperSonic loading"
        )
    })?;
    let required = qwen36_moe_required_flm_direct_weights(text_config, QWEN36_MOE_WEIGHT_PREFIX);
    let mut plan_kinds = Vec::with_capacity(required.len());
    for required in required {
        let kind = runtime
            .stage3_direct_weight_kind(&required.name)
            .map_err(|err| {
                anyhow!(
                    "Qwen3.6 MoE FLM direct weight plan for {} is unsupported: {err}",
                    required.name
                )
            })?;
        plan_kinds.push((required.name, required.expected, kind));
    }
    qwen36_moe_flm_weight_selection_from_required_plan_kinds(plan_kinds)
}

fn qwen36_moe_required_flm_direct_weights(
    text_config: &qwen36_moe::config::TextConfig,
    weight_prefix: &str,
) -> Vec<RequiredFlmDirectWeight> {
    expected_tensor_specs(text_config, weight_prefix)
        .into_iter()
        .map(|spec| RequiredFlmDirectWeight {
            name: spec.name,
            expected: qwen36_moe_required_flm_direct_weight_kind(spec.role),
        })
        .collect()
}

fn qwen36_moe_required_flm_direct_weight_kind(role: TensorRole) -> RequiredFlmDirectWeightKind {
    match role {
        TensorRole::FullQProj
        | TensorRole::FullKProj
        | TensorRole::FullVProj
        | TensorRole::FullOProj
        | TensorRole::LinearInQkvProj
        | TensorRole::LinearInZProj
        | TensorRole::LinearOutProj
        | TensorRole::SharedExpertGateProj
        | TensorRole::SharedExpertUpProj
        | TensorRole::SharedExpertDownProj
        | TensorRole::ExpertGateUpProj
        | TensorRole::ExpertDownProj => RequiredFlmDirectWeightKind::QuantizedProjection,
        TensorRole::Embed
        | TensorRole::Norm
        | TensorRole::LmHead
        | TensorRole::LayerInputNorm
        | TensorRole::LayerPostAttnNorm
        | TensorRole::FullQNorm
        | TensorRole::FullKNorm
        | TensorRole::LinearInBProj
        | TensorRole::LinearInAProj
        | TensorRole::LinearConv1d
        | TensorRole::LinearDtBias
        | TensorRole::LinearALog
        | TensorRole::LinearNorm
        | TensorRole::Router
        | TensorRole::SharedExpertGate => RequiredFlmDirectWeightKind::RawDense,
    }
}

fn qwen36_moe_flm_weight_selection_from_required_plan_kinds<I, S>(
    plans: I,
) -> Result<Qwen36MoeFlmWeightSelection>
where
    I: IntoIterator<
        Item = (
            S,
            RequiredFlmDirectWeightKind,
            Option<FlmStage3DirectWeightKind>,
        ),
    >,
    S: Into<String>,
{
    let mut native_int4_name: Option<String> = None;
    let mut fallback_name: Option<String> = None;

    for (name, expected, kind) in plans {
        let name = name.into();
        match (expected, kind) {
            (RequiredFlmDirectWeightKind::RawDense, Some(FlmStage3DirectWeightKind::RawDense)) => {}
            (RequiredFlmDirectWeightKind::RawDense, Some(other)) => {
                return Err(anyhow!(
                    "Qwen3.6 MoE FLM expected raw dense direct plan for {name}, got {other:?}"
                ));
            }
            (RequiredFlmDirectWeightKind::RawDense, None) => {
                return Err(anyhow!("missing direct weight plan for {name}"));
            }
            (
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::NativeInt4),
            ) => {
                native_int4_name.get_or_insert(name);
            }
            (
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(
                    FlmStage3DirectWeightKind::CtInt4Bf16Fallback
                    | FlmStage3DirectWeightKind::RawDense,
                ),
            ) => {
                fallback_name.get_or_insert(name);
            }
            (RequiredFlmDirectWeightKind::QuantizedProjection, None) => {
                return Err(anyhow!("missing direct weight plan for {name}"));
            }
        }
    }

    if let (Some(native), Some(fallback)) = (native_int4_name.as_ref(), fallback_name.as_ref()) {
        return Err(anyhow!(
            "mixed FLM direct projection modes: native INT4 example {native}; BF16 fallback example {fallback}"
        ));
    }

    if native_int4_name.is_some() {
        Ok(Qwen36MoeFlmWeightSelection {
            mode: Qwen36WeightMode::Int4,
            label: "INT4 native FLM",
        })
    } else {
        Ok(Qwen36MoeFlmWeightSelection {
            mode: Qwen36WeightMode::Bf16,
            label: "BF16",
        })
    }
}

#[cfg(test)]
fn qwen36_moe_flm_weight_selection_from_stage3_kind(
    kind: Option<FlmStage3DirectWeightKind>,
) -> Result<Qwen36MoeFlmWeightSelection> {
    match kind {
        Some(FlmStage3DirectWeightKind::NativeInt4) => Ok(Qwen36MoeFlmWeightSelection {
            mode: Qwen36WeightMode::Int4,
            label: "INT4 native FLM",
        }),
        Some(
            FlmStage3DirectWeightKind::CtInt4Bf16Fallback | FlmStage3DirectWeightKind::RawDense,
        ) => Ok(Qwen36MoeFlmWeightSelection {
            mode: Qwen36WeightMode::Bf16,
            label: "BF16",
        }),
        None => Err(anyhow!(
            "Qwen3.6 MoE FLM source requires a compatible direct weight plan for {}",
            QWEN36_MOE_WEIGHT_MODE_PROBE
        )),
    }
}

#[cfg(test)]
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
    validate_flm_weight_source_options(cli, crate::policy::q4km_like(cli))?;
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
    let weight_selection =
        qwen36_moe_flm_weight_selection_for_store(&source.store, &config.text_config)?;
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
    use clap::Parser;
    use model_store::flm::FlmStage3DirectWeightKind;
    use model_store::manifest::LayoutTag;
    use qwen36_moe::config::{Activation, TextConfig};

    use super::*;

    fn cli(extra: &[&str]) -> Cli {
        let mut args = vec![
            "supersonic",
            "--model",
            "qwen3.6-35b-a3b",
            "--model-dir",
            "/tmp/qwen36-moe-missing.flm",
            "--dry-run",
        ];
        args.extend_from_slice(extra);
        Cli::parse_from(args)
    }

    fn small_text_config() -> TextConfig {
        TextConfig {
            vocab_size: 1024,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 64,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            layer_types: Vec::new(),
            rope_parameters: None,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 64,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        }
        .normalized()
    }

    fn has_required(
        required: &[RequiredFlmDirectWeight],
        name: &str,
        expected: RequiredFlmDirectWeightKind,
    ) -> bool {
        required
            .iter()
            .any(|item| item.name == name && item.expected == expected)
    }

    #[test]
    fn required_direct_plan_names_cover_qwen36_moe_runtime_path() {
        let required =
            qwen36_moe_required_flm_direct_weights(&small_text_config(), "model.language_model");

        assert_eq!(required.len(), 72);
        assert!(has_required(
            &required,
            "model.language_model.embed_tokens.weight",
            RequiredFlmDirectWeightKind::RawDense
        ));
        assert!(has_required(
            &required,
            "lm_head.weight",
            RequiredFlmDirectWeightKind::RawDense
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            RequiredFlmDirectWeightKind::RawDense
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            RequiredFlmDirectWeightKind::RawDense
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.linear_attn.A_log",
            RequiredFlmDirectWeightKind::RawDense
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            RequiredFlmDirectWeightKind::QuantizedProjection
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.3.self_attn.q_proj.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection
        ));
        assert!(has_required(
            &required,
            "model.language_model.layers.3.self_attn.q_norm.weight",
            RequiredFlmDirectWeightKind::RawDense
        ));
        assert!(!required
            .iter()
            .any(|item| item.name == "model.language_model.layers.3.linear_attn.A_log"));
    }

    #[test]
    fn required_direct_plan_selection_uses_complete_native_int4_coverage() {
        let selection = qwen36_moe_flm_weight_selection_from_required_plan_kinds([
            (
                "model.language_model.embed_tokens.weight",
                RequiredFlmDirectWeightKind::RawDense,
                Some(FlmStage3DirectWeightKind::RawDense),
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::NativeInt4),
            ),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::NativeInt4),
            ),
        ])
        .expect("complete native INT4 direct coverage should select native mode");

        assert_eq!(selection.mode, Qwen36WeightMode::Int4);
        assert_eq!(selection.label, "INT4 native FLM");
    }

    #[test]
    fn required_direct_plan_selection_rejects_missing_required_tensor() {
        let err = qwen36_moe_flm_weight_selection_from_required_plan_kinds([(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection,
            None,
        )])
        .unwrap_err()
        .to_string();

        assert!(err.contains("missing direct weight plan"), "{err}");
        assert!(
            err.contains("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            "{err}"
        );
    }

    #[test]
    fn required_direct_plan_selection_rejects_mixed_projection_modes() {
        let err = qwen36_moe_flm_weight_selection_from_required_plan_kinds([
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::NativeInt4),
            ),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::RawDense),
            ),
        ])
        .unwrap_err()
        .to_string();

        assert!(err.contains("mixed FLM direct projection modes"), "{err}");
    }

    #[test]
    fn maps_native_stage3_direct_plan_to_int4_weight_mode() {
        let selection = qwen36_moe_flm_weight_selection_from_stage3_kind(Some(
            FlmStage3DirectWeightKind::NativeInt4,
        ))
        .expect("native Stage 3 direct plan should select native INT4");

        assert_eq!(selection.mode, Qwen36WeightMode::Int4);
        assert_eq!(selection.label, "INT4 native FLM");
    }

    #[test]
    fn maps_ct_stage3_direct_plan_to_bf16_fallback_weight_mode() {
        let selection = qwen36_moe_flm_weight_selection_from_stage3_kind(Some(
            FlmStage3DirectWeightKind::CtInt4Bf16Fallback,
        ))
        .expect("CT Stage 3 direct plan should select BF16 fallback");

        assert_eq!(selection.mode, Qwen36WeightMode::Bf16);
        assert_eq!(selection.label, "BF16");
    }

    #[test]
    fn maps_raw_dense_stage3_direct_plan_to_bf16_weight_mode() {
        let selection = qwen36_moe_flm_weight_selection_from_stage3_kind(Some(
            FlmStage3DirectWeightKind::RawDense,
        ))
        .expect("raw dense Stage 3 direct plan should select BF16");

        assert_eq!(selection.mode, Qwen36WeightMode::Bf16);
        assert_eq!(selection.label, "BF16");
    }

    #[test]
    fn rejects_missing_stage3_direct_weight_plan() {
        let err = qwen36_moe_flm_weight_selection_from_stage3_kind(None)
            .unwrap_err()
            .to_string();

        assert!(err.contains("direct weight plan"), "{err}");
        assert!(err.contains(QWEN36_MOE_WEIGHT_MODE_PROBE), "{err}");
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

    #[test]
    fn rejects_incompatible_quant_flags_before_opening_flm_source() {
        for flag in ["--q4km", "--q4km-gptq", "--int8"] {
            let err = match open_qwen36_moe_flm_source(&cli(&[flag])) {
                Ok(_) => panic!("incompatible FLM quant flag {flag} should fail before file open"),
                Err(err) => err.to_string(),
            };

            assert!(err.contains("FLM"), "{err}");
            assert!(
                err.contains(flag) || err.contains("--q4km/--q4km-gptq"),
                "error should identify {flag}: {err}"
            );
            assert!(
                !err.contains("opening Qwen3.6 MoE FLM source"),
                "validation should run before source open: {err}"
            );
        }
    }
}
