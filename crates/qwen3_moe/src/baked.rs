use std::path::{Path, PathBuf};

use model_store::{
    manifest::{LayoutTag, TensorMeta},
    BakedStore,
};
use thiserror::Error;

use crate::{
    config::TextConfig,
    weights::{baked_tensor_contract, BakedLayout, BakedTensorSpec},
};

pub const DEFAULT_INT4_GROUP_SIZE: usize = 128;

#[derive(Debug, Error)]
pub enum BakedIndexError {
    #[error("Qwen3 INT4 bake does not match runtime contract")]
    Contract(BakedContractReport),
    #[error("required baked tensor disappeared after contract validation: {0}")]
    MissingTensor(String),
}

#[derive(Debug)]
pub enum Int4BakeInspection {
    MissingOrOutdated {
        bake_dir: PathBuf,
    },
    Present {
        bake_dir: PathBuf,
        report: BakedContractReport,
    },
}

impl Int4BakeInspection {
    pub fn bake_dir(&self) -> &Path {
        match self {
            Self::MissingOrOutdated { bake_dir } | Self::Present { bake_dir, .. } => bake_dir,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BakedContractReport {
    pub present: usize,
    pub missing: usize,
    pub shape_mismatches: usize,
    pub dtype_mismatches: usize,
    pub layout_mismatches: usize,
    pub int4_layouts: usize,
    pub raw_layouts: usize,
    pub other_layouts: usize,
    pub bytes: u64,
    pub missing_examples: Vec<String>,
    pub shape_examples: Vec<String>,
    pub dtype_examples: Vec<String>,
    pub layout_examples: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakedTensorRef {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_len: u64,
}

impl BakedTensorRef {
    fn from_meta(meta: &TensorMeta) -> Self {
        Self {
            name: meta.name.clone(),
            dtype: meta.dtype.clone(),
            shape: meta.shape.clone(),
            byte_len: meta.byte_len,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeInt4BakedIndex {
    pub embed_tokens: BakedTensorRef,
    pub final_norm: BakedTensorRef,
    pub lm_head: Option<Qwen3MoeInt4Tensor>,
    pub layers: Vec<Qwen3MoeInt4LayerIndex>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeInt4LayerIndex {
    pub input_norm: BakedTensorRef,
    pub post_attn_norm: BakedTensorRef,
    pub q_proj: Qwen3MoeInt4Tensor,
    pub k_proj: Qwen3MoeInt4Tensor,
    pub v_proj: Qwen3MoeInt4Tensor,
    pub o_proj: Qwen3MoeInt4Tensor,
    pub q_norm: BakedTensorRef,
    pub k_norm: BakedTensorRef,
    pub router: BakedTensorRef,
    pub experts_gate_up: Qwen3MoeInt4Tensor,
    pub experts_down: Qwen3MoeInt4Tensor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3MoeInt4Tensor {
    pub weight: BakedTensorRef,
    pub scale: BakedTensorRef,
    pub zero: BakedTensorRef,
}

impl BakedContractReport {
    pub fn is_complete(&self) -> bool {
        self.missing == 0
            && self.shape_mismatches == 0
            && self.dtype_mismatches == 0
            && self.layout_mismatches == 0
    }
}

pub fn int4_contract(config: &TextConfig, prefix: &str) -> Vec<BakedTensorSpec> {
    baked_tensor_contract(config, prefix, DEFAULT_INT4_GROUP_SIZE)
}

pub fn inspect_int4_bake(
    model_dir: &Path,
    config: &TextConfig,
    prefix: &str,
) -> Result<Int4BakeInspection, model_store::Error> {
    let bake_dir = model_store::bake_dir_for_quant_profile(
        model_dir,
        model_store::manifest::QuantProfile::Int4Gptq,
    );
    if !model_store::version_ok_for_quant_profile(
        &bake_dir,
        model_store::manifest::QuantProfile::Int4Gptq,
    ) {
        return Ok(Int4BakeInspection::MissingOrOutdated { bake_dir });
    }

    let store = BakedStore::open(&bake_dir)?;
    let report = inspect_int4_store(&store, config, prefix);
    Ok(Int4BakeInspection::Present { bake_dir, report })
}

pub fn inspect_int4_store(
    store: &BakedStore,
    config: &TextConfig,
    prefix: &str,
) -> BakedContractReport {
    inspect_contract(store, &int4_contract(config, prefix))
}

pub fn build_int4_baked_index(
    store: &BakedStore,
    config: &TextConfig,
    prefix: &str,
) -> Result<Qwen3MoeInt4BakedIndex, BakedIndexError> {
    let report = inspect_int4_store(store, config, prefix);
    if !report.is_complete() {
        return Err(BakedIndexError::Contract(report));
    }

    let embed_tokens = required_ref(store, &format!("{prefix}.embed_tokens.weight"))?;
    let final_norm = required_ref(store, &format!("{prefix}.norm.weight"))?;
    let lm_head = if config.tie_word_embeddings {
        None
    } else {
        Some(required_int4_tensor(store, "lm_head.weight")?)
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let lp = format!("{prefix}.layers.{layer}");
        let ap = format!("{lp}.self_attn");
        let mp = format!("{lp}.mlp");
        layers.push(Qwen3MoeInt4LayerIndex {
            input_norm: required_ref(store, &format!("{lp}.input_layernorm.weight"))?,
            post_attn_norm: required_ref(store, &format!("{lp}.post_attention_layernorm.weight"))?,
            q_proj: required_int4_tensor(store, &format!("{ap}.q_proj.weight"))?,
            k_proj: required_int4_tensor(store, &format!("{ap}.k_proj.weight"))?,
            v_proj: required_int4_tensor(store, &format!("{ap}.v_proj.weight"))?,
            o_proj: required_int4_tensor(store, &format!("{ap}.o_proj.weight"))?,
            q_norm: required_ref(store, &format!("{ap}.q_norm.weight"))?,
            k_norm: required_ref(store, &format!("{ap}.k_norm.weight"))?,
            router: required_ref(store, &format!("{mp}.gate.weight"))?,
            experts_gate_up: required_int4_tensor(store, &format!("{mp}.experts.gate_up_proj"))?,
            experts_down: required_int4_tensor(store, &format!("{mp}.experts.down_proj"))?,
        });
    }

    let mut total_bytes = embed_tokens.byte_len + final_norm.byte_len;
    if let Some(lm_head) = &lm_head {
        total_bytes = total_bytes.saturating_add(lm_head.total_bytes());
    }
    for layer in &layers {
        total_bytes = total_bytes.saturating_add(layer.total_bytes());
    }

    Ok(Qwen3MoeInt4BakedIndex {
        embed_tokens,
        final_norm,
        lm_head,
        layers,
        total_bytes,
    })
}

impl Qwen3MoeInt4LayerIndex {
    pub fn total_bytes(&self) -> u64 {
        self.input_norm
            .byte_len
            .saturating_add(self.post_attn_norm.byte_len)
            .saturating_add(self.q_proj.total_bytes())
            .saturating_add(self.k_proj.total_bytes())
            .saturating_add(self.v_proj.total_bytes())
            .saturating_add(self.o_proj.total_bytes())
            .saturating_add(self.q_norm.byte_len)
            .saturating_add(self.k_norm.byte_len)
            .saturating_add(self.router.byte_len)
            .saturating_add(self.experts_gate_up.total_bytes())
            .saturating_add(self.experts_down.total_bytes())
    }
}

impl Qwen3MoeInt4Tensor {
    pub fn total_bytes(&self) -> u64 {
        self.weight
            .byte_len
            .saturating_add(self.scale.byte_len)
            .saturating_add(self.zero.byte_len)
    }
}

pub fn inspect_contract(store: &BakedStore, specs: &[BakedTensorSpec]) -> BakedContractReport {
    let mut report = BakedContractReport::default();
    for spec in specs {
        let Some(meta) = store.meta(&spec.name) else {
            report.missing += 1;
            push_example(&mut report.missing_examples, spec.name.clone());
            continue;
        };
        report.present += 1;
        report.bytes = report.bytes.saturating_add(meta.byte_len);

        let got_layout = store.layout(&spec.name);
        match got_layout {
            Some(LayoutTag::Int4Quantized) => report.int4_layouts += 1,
            Some(LayoutTag::Raw) => report.raw_layouts += 1,
            Some(_) => report.other_layouts += 1,
            None => {}
        }
        if !layout_matches(got_layout, spec.layout) {
            report.layout_mismatches += 1;
            push_example(
                &mut report.layout_examples,
                format!("{} has layout {:?}", spec.name, got_layout),
            );
        }
        if meta.dtype != spec.dtype {
            report.dtype_mismatches += 1;
            push_example(
                &mut report.dtype_examples,
                format!(
                    "{} has dtype {}, expected {}",
                    spec.name, meta.dtype, spec.dtype
                ),
            );
        }
        if meta.shape != spec.shape {
            report.shape_mismatches += 1;
            push_example(
                &mut report.shape_examples,
                format!(
                    "{} has {:?}, expected {:?}",
                    spec.name, meta.shape, spec.shape
                ),
            );
        }
    }
    report
}

fn required_int4_tensor(
    store: &BakedStore,
    name: &str,
) -> Result<Qwen3MoeInt4Tensor, BakedIndexError> {
    Ok(Qwen3MoeInt4Tensor {
        weight: required_ref(store, name)?,
        scale: required_ref(store, &format!("{name}_int4_scale"))?,
        zero: required_ref(store, &format!("{name}_int4_zero"))?,
    })
}

fn required_ref(store: &BakedStore, name: &str) -> Result<BakedTensorRef, BakedIndexError> {
    store
        .meta(name)
        .map(BakedTensorRef::from_meta)
        .ok_or_else(|| BakedIndexError::MissingTensor(name.to_string()))
}

fn layout_matches(got: Option<&LayoutTag>, expected: BakedLayout) -> bool {
    matches!(
        (got, expected),
        (Some(LayoutTag::Raw), BakedLayout::Raw)
            | (Some(LayoutTag::Int4Quantized), BakedLayout::Int4Quantized)
    )
}

fn push_example(examples: &mut Vec<String>, value: String) {
    if examples.len() < 3 {
        examples.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Activation, TextConfig};

    fn config_30b_a3b() -> TextConfig {
        TextConfig {
            vocab_size: 151_936,
            hidden_size: 2048,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            max_position_embeddings: 40_960,
            rms_norm_eps: 1e-6,
            intermediate_size: 6144,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 768,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
        }
    }

    #[test]
    fn int4_contract_uses_qwen3_group_size() {
        let contract = int4_contract(&config_30b_a3b(), crate::weights::DEFAULT_PREFIX);
        let q0_scale = contract
            .iter()
            .find(|s| s.name == "model.layers.0.self_attn.q_proj.weight_int4_scale")
            .unwrap();
        assert_eq!(q0_scale.shape, vec![32, 16]);
    }

    #[test]
    fn complete_report_has_no_contract_errors() {
        let report = BakedContractReport {
            present: 1,
            ..BakedContractReport::default()
        };
        assert!(report.is_complete());

        let report = BakedContractReport {
            missing: 1,
            ..BakedContractReport::default()
        };
        assert!(!report.is_complete());
    }

    #[test]
    fn qwen3_int4_index_byte_accounting_matches_contract_projection() {
        let cfg = config_30b_a3b();
        let contract = int4_contract(&cfg, crate::weights::DEFAULT_PREFIX);
        let expected_bytes: u64 = contract
            .iter()
            .map(|spec| {
                let dtype_size = match spec.dtype {
                    "bf16" => 2,
                    "u8" => 1,
                    other => panic!("unexpected dtype {other}"),
                };
                spec.shape.iter().product::<usize>() as u64 * dtype_size
            })
            .sum();
        let projected = crate::weights::CheckpointAccount::from_config(&cfg)
            .project_int4_total_bytes(&cfg, DEFAULT_INT4_GROUP_SIZE as u64);
        assert_eq!(expected_bytes, projected);
    }
}
