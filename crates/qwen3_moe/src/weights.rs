use crate::config::TextConfig;

pub const DEFAULT_PREFIX: &str = "model";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    Embed,
    Norm,
    LmHead,
    LayerInputNorm,
    LayerPostAttnNorm,
    QProj,
    KProj,
    VProj,
    OProj,
    QNorm,
    KNorm,
    Router,
    ExpertGateProj,
    ExpertUpProj,
    ExpertDownProj,
    FusedExpertGateUpProj,
    FusedExpertDownProj,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckpointDtype {
    Bf16,
    F32,
}

impl CheckpointDtype {
    pub const fn size(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

pub fn checkpoint_dtype_for(_role: TensorRole) -> CheckpointDtype {
    CheckpointDtype::Bf16
}

pub fn checkpoint_dtype_acceptable(role: TensorRole, got: CheckpointDtype) -> bool {
    match role {
        TensorRole::Norm
        | TensorRole::LayerInputNorm
        | TensorRole::LayerPostAttnNorm
        | TensorRole::QNorm
        | TensorRole::KNorm => matches!(got, CheckpointDtype::Bf16 | CheckpointDtype::F32),
        _ => got == CheckpointDtype::Bf16,
    }
}

pub fn checkpoint_elems_for(config: &TextConfig, role: TensorRole) -> u64 {
    let hidden = config.hidden_size as u64;
    let head_dim = config.head_dim as u64;
    let q_dim = config.num_attention_heads as u64 * head_dim;
    let kv_dim = config.num_key_value_heads as u64 * head_dim;
    let moe = config.moe_intermediate_size as u64;

    match role {
        TensorRole::Embed | TensorRole::LmHead => hidden * config.vocab_size as u64,
        TensorRole::Norm | TensorRole::LayerInputNorm | TensorRole::LayerPostAttnNorm => hidden,
        TensorRole::QProj => q_dim * hidden,
        TensorRole::KProj | TensorRole::VProj => kv_dim * hidden,
        TensorRole::OProj => hidden * q_dim,
        TensorRole::QNorm | TensorRole::KNorm => head_dim,
        TensorRole::Router => config.num_experts as u64 * hidden,
        TensorRole::ExpertGateProj | TensorRole::ExpertUpProj => moe * hidden,
        TensorRole::ExpertDownProj => hidden * moe,
        TensorRole::FusedExpertGateUpProj => config.num_experts as u64 * 2 * moe * hidden,
        TensorRole::FusedExpertDownProj => config.num_experts as u64 * hidden * moe,
    }
}

pub fn checkpoint_bytes_for(config: &TextConfig, role: TensorRole) -> u64 {
    checkpoint_elems_for(config, role) * checkpoint_dtype_for(role).size()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub name: String,
    pub role: TensorRole,
    pub layer_idx: Option<usize>,
    pub expert_idx: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BakedLayout {
    Raw,
    Int4Quantized,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BakedTensorSpec {
    pub name: String,
    pub role: TensorRole,
    pub layer_idx: Option<usize>,
    pub layout: BakedLayout,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    fn new(role: TensorRole, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            role,
            layer_idx: None,
            expert_idx: None,
        }
    }

    fn for_layer(role: TensorRole, name: impl Into<String>, layer: usize) -> Self {
        Self {
            name: name.into(),
            role,
            layer_idx: Some(layer),
            expert_idx: None,
        }
    }

    fn for_expert(role: TensorRole, name: impl Into<String>, layer: usize, expert: usize) -> Self {
        Self {
            name: name.into(),
            role,
            layer_idx: Some(layer),
            expert_idx: Some(expert),
        }
    }
}

pub fn expected_tensor_specs(config: &TextConfig, prefix: &str) -> Vec<TensorSpec> {
    let mut out = Vec::new();
    out.push(TensorSpec::new(
        TensorRole::Embed,
        format!("{prefix}.embed_tokens.weight"),
    ));
    out.push(TensorSpec::new(
        TensorRole::Norm,
        format!("{prefix}.norm.weight"),
    ));
    if !config.tie_word_embeddings {
        out.push(TensorSpec::new(TensorRole::LmHead, "lm_head.weight"));
    }

    for layer in 0..config.num_hidden_layers {
        let lp = format!("{prefix}.layers.{layer}");
        out.push(TensorSpec::for_layer(
            TensorRole::LayerInputNorm,
            format!("{lp}.input_layernorm.weight"),
            layer,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::LayerPostAttnNorm,
            format!("{lp}.post_attention_layernorm.weight"),
            layer,
        ));

        let ap = format!("{lp}.self_attn");
        for (role, suffix) in [
            (TensorRole::QProj, "q_proj.weight"),
            (TensorRole::KProj, "k_proj.weight"),
            (TensorRole::VProj, "v_proj.weight"),
            (TensorRole::OProj, "o_proj.weight"),
            (TensorRole::QNorm, "q_norm.weight"),
            (TensorRole::KNorm, "k_norm.weight"),
        ] {
            out.push(TensorSpec::for_layer(role, format!("{ap}.{suffix}"), layer));
        }

        let mp = format!("{lp}.mlp");
        out.push(TensorSpec::for_layer(
            TensorRole::Router,
            format!("{mp}.gate.weight"),
            layer,
        ));
        for expert in 0..config.num_experts {
            let ep = format!("{mp}.experts.{expert}");
            out.push(TensorSpec::for_expert(
                TensorRole::ExpertGateProj,
                format!("{ep}.gate_proj.weight"),
                layer,
                expert,
            ));
            out.push(TensorSpec::for_expert(
                TensorRole::ExpertUpProj,
                format!("{ep}.up_proj.weight"),
                layer,
                expert,
            ));
            out.push(TensorSpec::for_expert(
                TensorRole::ExpertDownProj,
                format!("{ep}.down_proj.weight"),
                layer,
                expert,
            ));
        }
    }
    out
}

pub fn baked_tensor_specs(config: &TextConfig, prefix: &str) -> Vec<TensorSpec> {
    let mut out = expected_tensor_specs(config, prefix)
        .into_iter()
        .filter(|s| {
            !matches!(
                s.role,
                TensorRole::ExpertGateProj | TensorRole::ExpertUpProj | TensorRole::ExpertDownProj
            )
        })
        .collect::<Vec<_>>();
    for layer in 0..config.num_hidden_layers {
        let mp = format!("{prefix}.layers.{layer}.mlp.experts");
        out.push(TensorSpec::for_layer(
            TensorRole::FusedExpertGateUpProj,
            format!("{mp}.gate_up_proj"),
            layer,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::FusedExpertDownProj,
            format!("{mp}.down_proj"),
            layer,
        ));
    }
    out
}

pub fn baked_tensor_contract(
    config: &TextConfig,
    prefix: &str,
    group_size: usize,
) -> Vec<BakedTensorSpec> {
    let mut out = Vec::new();
    for spec in baked_tensor_specs(config, prefix) {
        let layout = baked_layout_for(spec.role);
        let dtype = match layout {
            BakedLayout::Raw => "bf16",
            BakedLayout::Int4Quantized => "u8",
        };
        let shape = baked_shape_for(config, spec.role, layout, group_size);
        out.push(BakedTensorSpec {
            name: spec.name.clone(),
            role: spec.role,
            layer_idx: spec.layer_idx,
            layout,
            dtype,
            shape,
        });
        if layout == BakedLayout::Int4Quantized {
            let sidecar_shape = int4_sidecar_shape_for(config, spec.role, group_size);
            for suffix in ["_int4_scale", "_int4_zero"] {
                out.push(BakedTensorSpec {
                    name: format!("{}{}", spec.name, suffix),
                    role: spec.role,
                    layer_idx: spec.layer_idx,
                    layout: BakedLayout::Raw,
                    dtype: "bf16",
                    shape: sidecar_shape.clone(),
                });
            }
        }
    }
    out
}

pub fn baked_layout_for(role: TensorRole) -> BakedLayout {
    match role {
        TensorRole::LmHead
        | TensorRole::QProj
        | TensorRole::KProj
        | TensorRole::VProj
        | TensorRole::OProj
        | TensorRole::FusedExpertGateUpProj
        | TensorRole::FusedExpertDownProj => BakedLayout::Int4Quantized,
        _ => BakedLayout::Raw,
    }
}

pub fn baked_shape_for(
    config: &TextConfig,
    role: TensorRole,
    layout: BakedLayout,
    _group_size: usize,
) -> Vec<usize> {
    let hidden = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let moe = config.moe_intermediate_size;
    let pack_cols = |cols: usize| match layout {
        BakedLayout::Raw => cols,
        BakedLayout::Int4Quantized => cols.div_ceil(2),
    };

    match role {
        TensorRole::Embed | TensorRole::LmHead => vec![config.vocab_size, pack_cols(hidden)],
        TensorRole::Norm | TensorRole::LayerInputNorm | TensorRole::LayerPostAttnNorm => {
            vec![hidden]
        }
        TensorRole::QProj => vec![q_dim, pack_cols(hidden)],
        TensorRole::KProj | TensorRole::VProj => vec![kv_dim, pack_cols(hidden)],
        TensorRole::OProj => vec![hidden, pack_cols(q_dim)],
        TensorRole::QNorm | TensorRole::KNorm => vec![config.head_dim],
        TensorRole::Router => vec![config.num_experts, hidden],
        TensorRole::ExpertGateProj | TensorRole::ExpertUpProj => vec![moe, hidden],
        TensorRole::ExpertDownProj => vec![hidden, moe],
        TensorRole::FusedExpertGateUpProj => {
            vec![config.num_experts, 2 * moe, pack_cols(hidden)]
        }
        TensorRole::FusedExpertDownProj => vec![config.num_experts, hidden, pack_cols(moe)],
    }
}

pub fn int4_sidecar_shape_for(
    config: &TextConfig,
    role: TensorRole,
    group_size: usize,
) -> Vec<usize> {
    let g = group_size.max(1);
    let hidden = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let moe = config.moe_intermediate_size;

    match role {
        TensorRole::LmHead => vec![config.vocab_size.div_ceil(g), hidden.div_ceil(g)],
        TensorRole::QProj => vec![q_dim.div_ceil(g), hidden.div_ceil(g)],
        TensorRole::KProj | TensorRole::VProj => vec![kv_dim.div_ceil(g), hidden.div_ceil(g)],
        TensorRole::OProj => vec![hidden.div_ceil(g), q_dim.div_ceil(g)],
        TensorRole::FusedExpertGateUpProj => {
            vec![
                config.num_experts,
                (2 * moe).div_ceil(g),
                hidden.div_ceil(g),
            ]
        }
        TensorRole::FusedExpertDownProj => {
            vec![config.num_experts, hidden.div_ceil(g), moe.div_ceil(g)]
        }
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CheckpointAccount {
    pub embed_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
    pub per_layer_norm_bytes: u64,
    pub attn_bytes_per_layer: u64,
    pub router_bytes_per_layer: u64,
    pub experts_bytes_per_layer: u64,
    pub total_bytes: u64,
}

impl CheckpointAccount {
    pub fn from_config(config: &TextConfig) -> Self {
        let cb = |role| checkpoint_bytes_for(config, role);
        let embed_bytes = cb(TensorRole::Embed);
        let final_norm_bytes = cb(TensorRole::Norm);
        let lm_head_bytes = if config.tie_word_embeddings {
            0
        } else {
            cb(TensorRole::LmHead)
        };
        let per_layer_norm_bytes =
            cb(TensorRole::LayerInputNorm) + cb(TensorRole::LayerPostAttnNorm);
        let attn_bytes_per_layer = cb(TensorRole::QProj)
            + cb(TensorRole::KProj)
            + cb(TensorRole::VProj)
            + cb(TensorRole::OProj)
            + cb(TensorRole::QNorm)
            + cb(TensorRole::KNorm);
        let router_bytes_per_layer = cb(TensorRole::Router);
        let experts_bytes_per_layer = config.num_experts as u64
            * (cb(TensorRole::ExpertGateProj)
                + cb(TensorRole::ExpertUpProj)
                + cb(TensorRole::ExpertDownProj));
        let layers = config.num_hidden_layers as u64;
        let total_bytes = embed_bytes
            + final_norm_bytes
            + lm_head_bytes
            + layers
                * (per_layer_norm_bytes
                    + attn_bytes_per_layer
                    + router_bytes_per_layer
                    + experts_bytes_per_layer);
        Self {
            embed_bytes,
            final_norm_bytes,
            lm_head_bytes,
            per_layer_norm_bytes,
            attn_bytes_per_layer,
            router_bytes_per_layer,
            experts_bytes_per_layer,
            total_bytes,
        }
    }

    pub fn project_int4_total_bytes(&self, config: &TextConfig, group_size: u64) -> u64 {
        let hidden = config.hidden_size as u64;
        let vocab = config.vocab_size as u64;
        let head_dim = config.head_dim as u64;
        let q_dim = config.num_attention_heads as u64 * head_dim;
        let kv_dim = config.num_key_value_heads as u64 * head_dim;
        let moe = config.moe_intermediate_size as u64;

        let mut total = self.embed_bytes + self.final_norm_bytes;
        if !config.tie_word_embeddings {
            total += int4_bytes(vocab * hidden, vocab, hidden, group_size);
        }

        let per_layer_norms = self.per_layer_norm_bytes;
        let attn = int4_bytes(q_dim * hidden, q_dim, hidden, group_size)
            + int4_bytes(kv_dim * hidden, kv_dim, hidden, group_size)
            + int4_bytes(kv_dim * hidden, kv_dim, hidden, group_size)
            + int4_bytes(hidden * q_dim, hidden, q_dim, group_size)
            + 2 * 2 * head_dim;
        let router = self.router_bytes_per_layer;
        let per_expert = int4_bytes(moe * hidden, moe, hidden, group_size)
            + int4_bytes(moe * hidden, moe, hidden, group_size)
            + int4_bytes(hidden * moe, hidden, moe, group_size);
        total += config.num_hidden_layers as u64
            * (per_layer_norms + attn + router + config.num_experts as u64 * per_expert);
        total
    }
}

fn int4_bytes(elems: u64, out_dim: u64, in_dim: u64, group_size: u64) -> u64 {
    let packed = elems / 2;
    let tiles = out_dim.div_ceil(group_size).max(1) * in_dim.div_ceil(group_size).max(1);
    packed + 2 * 2 * tiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Activation;

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
    fn enumeration_counts_match_qwen3_30b_a3b() {
        let cfg = config_30b_a3b();
        let specs = expected_tensor_specs(&cfg, DEFAULT_PREFIX);
        assert_eq!(specs.len(), 3 + 48 * 393);
        assert_eq!(baked_tensor_specs(&cfg, DEFAULT_PREFIX).len(), 3 + 48 * 11);
    }

    #[test]
    fn baked_contract_counts_and_shapes_match_qwen3_30b_a3b() {
        let cfg = config_30b_a3b();
        let contract = baked_tensor_contract(&cfg, DEFAULT_PREFIX, 128);
        let quantized = contract
            .iter()
            .filter(|s| s.layout == BakedLayout::Int4Quantized)
            .count();
        assert_eq!(quantized, 1 + 48 * 6);
        assert_eq!(contract.len(), (3 + 48 * 11) + 2 * quantized);

        let q0 = contract
            .iter()
            .find(|s| s.name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q0.shape, vec![4096, 1024]);
        let q0_scale = contract
            .iter()
            .find(|s| s.name == "model.layers.0.self_attn.q_proj.weight_int4_scale")
            .unwrap();
        assert_eq!(q0_scale.shape, vec![32, 16]);
        let gate_up = contract
            .iter()
            .find(|s| s.name == "model.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        assert_eq!(gate_up.shape, vec![128, 1536, 1024]);
        let gate_up_scale = contract
            .iter()
            .find(|s| s.name == "model.layers.0.mlp.experts.gate_up_proj_int4_scale")
            .unwrap();
        assert_eq!(gate_up_scale.shape, vec![128, 12, 16]);
    }

    #[test]
    fn account_matches_hf_index_total() {
        let cfg = config_30b_a3b();
        let acct = CheckpointAccount::from_config(&cfg);
        assert_eq!(acct.total_bytes, 61_064_245_248);
    }
}
