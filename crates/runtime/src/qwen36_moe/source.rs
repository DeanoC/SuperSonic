use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use model_store::flm::FlmStage3DirectWeightKind;
use model_store::BakedStore;
use qwen36_moe::weights::{
    expected_tensor_specs, TensorRole, DEFAULT_PREFIX as QWEN36_MOE_WEIGHT_PREFIX,
};

use crate::flm_model_source::{FlmModelSource, FlmModelSourceOptions};

struct Qwen36MoeSourceOpenObserverState {
    path: PathBuf,
    count: AtomicU64,
}

pub(crate) struct Qwen36MoeSourceOpenObserver {
    state: Arc<Qwen36MoeSourceOpenObserverState>,
}

static SOURCE_OPEN_OBSERVERS: OnceLock<Mutex<Vec<Weak<Qwen36MoeSourceOpenObserverState>>>> =
    OnceLock::new();

impl Qwen36MoeSourceOpenObserver {
    pub(crate) fn for_path(path: &Path) -> Self {
        let state = Arc::new(Qwen36MoeSourceOpenObserverState {
            path: path.to_owned(),
            count: AtomicU64::new(0),
        });
        source_open_observers()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Arc::downgrade(&state));
        Self { state }
    }

    pub(crate) fn observed_count(&self) -> u64 {
        self.state.count.load(Ordering::SeqCst)
    }
}

fn source_open_observers() -> &'static Mutex<Vec<Weak<Qwen36MoeSourceOpenObserverState>>> {
    SOURCE_OPEN_OBSERVERS.get_or_init(|| Mutex::new(Vec::new()))
}

fn record_source_open(path: &Path) {
    source_open_observers()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .retain(|observer| {
            let Some(observer) = observer.upgrade() else {
                return false;
            };
            if observer.path == path {
                observer.count.fetch_add(1, Ordering::SeqCst);
            }
            true
        });
}

pub struct Qwen36MoeSource {
    pub source: FlmModelSource,
    pub config: qwen36_moe::config::Config,
    pub weight_mode: Qwen36WeightMode,
    pub direct_profile: Qwen36MoeDirectProfile,
    pub timings: Qwen36MoeSourceOpenTimings,
}

impl Qwen36MoeSource {
    pub fn open(path: &Path, options: FlmModelSourceOptions) -> Result<Self> {
        record_source_open(path);
        let mut timings = Qwen36MoeSourceOpenTimings::default();
        let store_open_start = Instant::now();
        let source = FlmModelSource::open_with_options(path, options)
            .map_err(|err| anyhow!("opening Qwen3.6 MoE FLM source {}: {err}", path.display()))?;
        timings.store_open = store_open_start.elapsed();

        let config_start = Instant::now();
        let config = source.qwen_moe_config()?;
        timings.config = config_start.elapsed();

        let direct_plan_start = Instant::now();
        let selection =
            qwen36_moe_flm_weight_selection_for_store(&source.store, &config.text_config)?;
        timings.direct_plan = direct_plan_start.elapsed();

        Ok(Self {
            source,
            config,
            weight_mode: selection.mode,
            direct_profile: selection.direct_profile,
            timings,
        })
    }

    pub fn load_tokenizer_timed(&self) -> Result<crate::flm_tokenizer::QwenBpeTokenizerLoad> {
        self.source.qwen_tokenizer_timed()
    }

    pub fn chat_template_source(&self) -> Result<&str> {
        self.source.chat_template_source()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen36MoeSourceOpenTimings {
    pub store_open: Duration,
    pub config: Duration,
    pub tokenizer: Duration,
    pub tokenizer_assets: Duration,
    pub tokenizer_parse: Duration,
    pub tokenizer_build: Duration,
    pub direct_plan: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36WeightMode {
    Bf16,
    Int4,
}

impl Qwen36WeightMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Int4 => "INT4 native FLM",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen36MoeDirectProfile {
    /// SuperSonic runtime-required logical weights covered by direct plans.
    /// This is intentionally smaller than the full FLM tensor table, which
    /// also contains storage sidecars and compatibility/runtime assets.
    pub required_tensors: usize,
    pub raw_dense: usize,
    /// Aggregate direct INT4 coverage across row-group and retained tile-v1 codecs.
    pub native_int4: usize,
    pub row_group_int4: usize,
    pub tile_int4_v1: usize,
    pub bf16_fallback: usize,
}

impl fmt::Display for Qwen36MoeDirectProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "required={} raw_dense={} native_int4={} row_group_int4={} tile_int4_v1={} bf16_fallback={}",
            self.required_tensors,
            self.raw_dense,
            self.native_int4,
            self.row_group_int4,
            self.tile_int4_v1,
            self.bf16_fallback
        )
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen36MoeFlmWeightSelection {
    mode: Qwen36WeightMode,
    direct_profile: Qwen36MoeDirectProfile,
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
    let direct_profile = qwen36_moe_flm_direct_profile_from_required_plan_kinds(plans)?;
    if direct_profile.native_int4 != direct_profile.row_group_int4 + direct_profile.tile_int4_v1 {
        anyhow::bail!(
            "Qwen3.6 MoE FLM direct profile native INT4 aggregate does not match codec-specific coverage"
        );
    }
    if direct_profile.row_group_int4 == 0
        || direct_profile.tile_int4_v1 != 0
        || direct_profile.bf16_fallback != 0
    {
        anyhow::bail!(
            "Qwen3.6 MoE first-class row-group INT4 direct plans require row_group_int4 > 0, tile_int4_v1 == 0, and bf16_fallback == 0; got {direct_profile}"
        );
    }

    Ok(Qwen36MoeFlmWeightSelection {
        mode: Qwen36WeightMode::Int4,
        direct_profile,
    })
}

fn qwen36_moe_flm_direct_profile_from_required_plan_kinds<I, S>(
    plans: I,
) -> Result<Qwen36MoeDirectProfile>
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
    let mut direct_profile = Qwen36MoeDirectProfile::default();

    for (name, expected, kind) in plans {
        let name = name.into();
        direct_profile.required_tensors += 1;
        match (expected, kind) {
            (RequiredFlmDirectWeightKind::RawDense, Some(FlmStage3DirectWeightKind::RawDense)) => {
                direct_profile.raw_dense += 1;
            }
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
                Some(FlmStage3DirectWeightKind::RowGroupInt4),
            ) => {
                direct_profile.native_int4 += 1;
                direct_profile.row_group_int4 += 1;
            }
            (
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::NativeInt4),
            ) => {
                direct_profile.native_int4 += 1;
                direct_profile.tile_int4_v1 += 1;
            }
            (
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(
                    FlmStage3DirectWeightKind::CtInt4Bf16Fallback
                    | FlmStage3DirectWeightKind::RawDense,
                ),
            ) => {
                direct_profile.bf16_fallback += 1;
            }
            (RequiredFlmDirectWeightKind::QuantizedProjection, None) => {
                return Err(anyhow!("missing direct weight plan for {name}"));
            }
        }
    }

    assert_eq!(
        direct_profile.native_int4,
        direct_profile.row_group_int4 + direct_profile.tile_int4_v1,
        "native INT4 aggregate must equal codec-specific direct coverage"
    );
    Ok(direct_profile)
}

#[cfg(test)]
mod tests {
    use model_store::flm::FlmStage3DirectWeightKind;
    use qwen36_moe::config::{Activation, TextConfig};

    use super::*;

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

    fn config_35b_a3b() -> TextConfig {
        let layer_types = (0..40)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();
        TextConfig {
            vocab_size: 248_320,
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 256,
            full_attention_interval: 4,
            attn_output_gate: true,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            layer_types,
            rope_parameters: None,
            num_experts: 256,
            num_experts_per_tok: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        }
    }

    #[test]
    fn direct_profile_separates_row_group_and_legacy_tile_int4() {
        let profile = qwen36_moe_flm_direct_profile_from_required_plan_kinds([
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::RowGroupInt4),
            ),
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                RequiredFlmDirectWeightKind::QuantizedProjection,
                Some(FlmStage3DirectWeightKind::NativeInt4),
            ),
            (
                "model.language_model.embed_tokens.weight",
                RequiredFlmDirectWeightKind::RawDense,
                Some(FlmStage3DirectWeightKind::RawDense),
            ),
        ])
        .expect("diagnostic plan accounting must accept both supported INT4 codecs");

        assert_eq!(profile.required_tensors, 3);
        assert_eq!(profile.raw_dense, 1);
        assert_eq!(profile.native_int4, 2);
        assert_eq!(profile.row_group_int4, 1);
        assert_eq!(profile.tile_int4_v1, 1);
        assert_eq!(profile.bf16_fallback, 0);
        assert_eq!(
            profile.native_int4,
            profile.row_group_int4 + profile.tile_int4_v1
        );
        assert_eq!(
            profile.to_string(),
            "required=3 raw_dense=1 native_int4=2 row_group_int4=1 tile_int4_v1=1 bf16_fallback=0"
        );
    }

    #[test]
    fn required_direct_plan_selection_accepts_only_pure_row_group_int4() {
        let row_group = qwen36_moe_flm_weight_selection_from_required_plan_kinds([(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection,
            Some(FlmStage3DirectWeightKind::RowGroupInt4),
        )])
        .expect("row-group-only direct plans must select the first-class INT4 path");
        assert_eq!(row_group.mode, Qwen36WeightMode::Int4);
        assert_eq!(row_group.direct_profile.row_group_int4, 1);
        assert_eq!(row_group.direct_profile.tile_int4_v1, 0);
        assert_eq!(row_group.direct_profile.bf16_fallback, 0);

        let err = qwen36_moe_flm_weight_selection_from_required_plan_kinds([(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection,
            Some(FlmStage3DirectWeightKind::NativeInt4),
        )])
        .expect_err("legacy tile INT4 must remain diagnostic-only");
        assert!(err.to_string().contains("tile_int4_v1"), "{err}");
    }

    #[test]
    fn required_direct_plan_selection_reports_native_35b_a3b_profile() {
        let plans = qwen36_moe_required_flm_direct_weights(
            &config_35b_a3b(),
            qwen36_moe::weights::DEFAULT_PREFIX,
        )
        .into_iter()
        .map(|required| {
            let kind = match required.expected {
                RequiredFlmDirectWeightKind::RawDense => FlmStage3DirectWeightKind::RawDense,
                RequiredFlmDirectWeightKind::QuantizedProjection => {
                    FlmStage3DirectWeightKind::RowGroupInt4
                }
            };
            (required.name, required.expected, Some(kind))
        });

        let selection = qwen36_moe_flm_weight_selection_from_required_plan_kinds(plans)
            .expect("complete native direct plans should select the Qwen3.6 runtime source");
        assert_eq!(selection.mode, Qwen36WeightMode::Int4);
        let profile = selection.direct_profile;

        assert_eq!(profile.native_int4, 330);
        assert_eq!(profile.row_group_int4, 330);
        assert_eq!(profile.tile_int4_v1, 0);
        assert_eq!(profile.bf16_fallback, 0);
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
    }

    #[test]
    fn required_direct_plan_selection_rejects_tile_and_fallback_evidence() {
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

        assert!(err.contains("row_group_int4"), "{err}");
    }

    #[test]
    fn required_direct_plan_selection_rejects_bf16_fallback_coverage() {
        let err = qwen36_moe_flm_weight_selection_from_required_plan_kinds([(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            RequiredFlmDirectWeightKind::QuantizedProjection,
            Some(FlmStage3DirectWeightKind::CtInt4Bf16Fallback),
        )])
        .expect_err("BF16 fallback plans cannot select the first-class row-group path")
        .to_string();

        assert!(err.contains("bf16_fallback"), "{err}");
    }

    #[test]
    fn required_direct_plan_selection_rejects_wrong_kind_for_raw_weight() {
        let err = qwen36_moe_flm_weight_selection_from_required_plan_kinds([(
            "model.language_model.embed_tokens.weight",
            RequiredFlmDirectWeightKind::RawDense,
            Some(FlmStage3DirectWeightKind::NativeInt4),
        )])
        .unwrap_err()
        .to_string();

        assert!(err.contains("expected raw dense direct plan"), "{err}");
    }

    #[test]
    fn required_direct_plan_names_cover_small_qwen36_moe_runtime_path() {
        let required =
            qwen36_moe_required_flm_direct_weights(&small_text_config(), "model.language_model");

        assert_eq!(required.len(), 72);
        assert!(required.iter().any(|item| {
            item.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
                && item.expected == RequiredFlmDirectWeightKind::QuantizedProjection
        }));
        assert!(required.iter().any(|item| {
            item.name == "model.language_model.layers.3.self_attn.q_norm.weight"
                && item.expected == RequiredFlmDirectWeightKind::RawDense
        }));
    }
}
