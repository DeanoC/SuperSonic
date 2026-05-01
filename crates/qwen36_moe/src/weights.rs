use crate::config::TextConfig;

pub const DEFAULT_PREFIX: &str = "model.language_model";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Linear,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    Embed,
    Norm,
    LmHead,
    LayerInputNorm,
    LayerPostAttnNorm,
    /// `self_attn.q_proj.weight`. Real shape `[2 * num_heads * head_dim, hidden]`
    /// when `attn_output_gate=true` (Qwen3-Next fuses Q with the output-gate
    /// projection); otherwise `[num_heads * head_dim, hidden]`.
    FullQProj,
    FullKProj,
    FullVProj,
    FullOProj,
    FullQNorm,
    FullKNorm,
    LinearInQkvProj,
    LinearInZProj,
    LinearInBProj,
    LinearInAProj,
    LinearOutProj,
    LinearConv1d,
    LinearDtBias,
    LinearALog,
    LinearNorm,
    Router,
    SharedExpertGate,
    SharedExpertGateProj,
    SharedExpertUpProj,
    SharedExpertDownProj,
    /// Fused `mlp.experts.gate_up_proj` (no `.weight` suffix) — single tensor
    /// of shape `[num_experts, 2 * moe_intermediate_size, hidden]` carrying
    /// gate and up projections for all experts in one slab.
    ExpertGateUpProj,
    /// `mlp.experts.down_proj` (no `.weight` suffix) — single tensor of shape
    /// `[num_experts, hidden, moe_intermediate_size]`.
    ExpertDownProj,
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

/// On-disk dtype for a tensor role in the published safetensors checkpoint.
///
/// Empirically: Qwen3.5-0.8B publishes `linear_attn.A_log` and
/// `linear_attn.norm.weight` as F32. Qwen3.6-35B-A3B publishes the same
/// tensors as BF16. Both load fine — runtime upcasts BF16 norms to F32 on
/// demand (`crates/qwen35/src/weights.rs::ensure_f32_on_gpu`).
///
/// This function returns the **default-expected** dtype for accounting
/// purposes — analytic byte totals use it, but a real checkpoint can store
/// either F32 or BF16 for these two tensors and both are accepted by
/// `dtype_compatible`. We default to BF16 here because that matches the
/// MoE checkpoint we're targeting in v1; the 5 KiB drift on the analytic
/// total when a checkpoint uses F32 instead is below the noise floor.
pub fn checkpoint_dtype_for(role: TensorRole) -> CheckpointDtype {
    let _ = role;
    CheckpointDtype::Bf16
}

/// Whether `got` is an acceptable on-disk dtype for `role`. Lenient by design:
/// norm-like and `A_log` tensors may be either F32 or BF16 depending on which
/// Qwen3-Next variant published the checkpoint, and the runtime handles both.
pub fn checkpoint_dtype_acceptable(role: TensorRole, got: CheckpointDtype) -> bool {
    match role {
        TensorRole::LinearNorm
        | TensorRole::LinearALog
        | TensorRole::Norm
        | TensorRole::LayerInputNorm
        | TensorRole::LayerPostAttnNorm
        | TensorRole::FullQNorm
        | TensorRole::FullKNorm => matches!(got, CheckpointDtype::F32 | CheckpointDtype::Bf16),
        _ => got == CheckpointDtype::Bf16,
    }
}

/// Computed elem count for a tensor role given the model config. Matches the
/// shapes stored in the safetensors checkpoint, not what the runtime ships
/// after bake-time transforms.
pub fn checkpoint_elems_for(config: &TextConfig, role: TensorRole) -> u64 {
    let hidden = config.hidden_size as u64;
    let head_dim = config.head_dim as u64;
    let q_dim_full = config.num_attention_heads as u64 * head_dim;
    let kv_dim_full = config.num_key_value_heads as u64 * head_dim;

    let lin_k_heads = config.linear_num_key_heads as u64;
    let lin_v_heads = config.linear_num_value_heads as u64;
    let lin_k_hd = config.linear_key_head_dim as u64;
    let lin_v_hd = config.linear_value_head_dim as u64;
    let lin_k_dim = lin_k_heads * lin_k_hd;
    let lin_v_dim = lin_v_heads * lin_v_hd;
    // Convention from a real Qwen3.5 0.8B checkpoint: `in_proj_qkv` packs
    // Q, K, V channels with Q_heads == K_heads (no separate Qwen-Next field
    // names them). For 0.8B: 16+16+16 heads × 128 = 6144. For 35B-A3B:
    // 16+16+32 heads × 128 = 8192.
    let lin_q_dim = lin_k_dim;
    let lin_qkv_dim = lin_q_dim + lin_k_dim + lin_v_dim;
    // conv1d packs the same Q+K+V channels along axis 0; shape is
    // [lin_qkv_dim, 1, kernel].
    let conv1d_elems = lin_qkv_dim * config.linear_conv_kernel_dim as u64;

    let q_proj_out = if config.attn_output_gate {
        2 * q_dim_full
    } else {
        q_dim_full
    };

    match role {
        TensorRole::Embed | TensorRole::LmHead => hidden * config.vocab_size as u64,
        TensorRole::Norm | TensorRole::LayerInputNorm | TensorRole::LayerPostAttnNorm => hidden,

        TensorRole::FullQProj => q_proj_out * hidden,
        TensorRole::FullKProj | TensorRole::FullVProj => kv_dim_full * hidden,
        TensorRole::FullOProj => hidden * q_dim_full,
        TensorRole::FullQNorm | TensorRole::FullKNorm => head_dim,

        TensorRole::LinearInQkvProj => lin_qkv_dim * hidden,
        TensorRole::LinearInZProj => lin_v_dim * hidden,
        TensorRole::LinearInBProj | TensorRole::LinearInAProj => lin_v_heads * hidden,
        TensorRole::LinearOutProj => hidden * lin_v_dim,
        TensorRole::LinearConv1d => conv1d_elems,
        TensorRole::LinearDtBias | TensorRole::LinearALog => lin_v_heads,
        // linear_attn.norm.weight applies to per-head value stream; shape
        // [head_dim]. Stored BF16 in 35B-A3B but F32 in some Qwen3.5 variants.
        TensorRole::LinearNorm => lin_k_hd,

        TensorRole::Router => config.num_experts as u64 * hidden,
        // shared_expert_gate is shape [1, hidden] in the checkpoint.
        TensorRole::SharedExpertGate => hidden,
        TensorRole::SharedExpertGateProj | TensorRole::SharedExpertUpProj => {
            config.shared_expert_intermediate_size as u64 * hidden
        }
        TensorRole::SharedExpertDownProj => hidden * config.shared_expert_intermediate_size as u64,

        // Fused expert tensors carry weights for ALL experts in a single slab.
        // `gate_up_proj`: [E, 2*moe_int, hidden] (gate concat'd with up).
        // `down_proj`:    [E, hidden, moe_int].
        TensorRole::ExpertGateUpProj => {
            config.num_experts as u64 * 2 * config.moe_intermediate_size as u64 * hidden
        }
        TensorRole::ExpertDownProj => {
            config.num_experts as u64 * hidden * config.moe_intermediate_size as u64
        }
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

    #[allow(dead_code)]
    fn for_expert(role: TensorRole, name: impl Into<String>, layer: usize, expert: usize) -> Self {
        // Reserved for future per-expert iteration paths if a checkpoint ever
        // ships unfused expert tensors. The current Qwen3.6-MoE bake fuses
        // them per-layer, so this helper is unused at the moment.
        Self {
            name: name.into(),
            role,
            layer_idx: Some(layer),
            expert_idx: Some(expert),
        }
    }
}

/// Build the full list of tensor names this runtime expects in the loaded
/// safetensors checkpoint. Used by the dry-run path to (a) fail fast if the
/// checkpoint is missing tensors, (b) compute the BF16 footprint analytically,
/// and (c) cross-check that footprint against the real on-disk total once the
/// download is local.
///
/// The naming follows HuggingFace's `Qwen3_5MoeForConditionalGeneration` /
/// `Qwen3_5MoeForCausalLM` convention. The text decoder lives under
/// `model.language_model.*`; `lm_head.weight` is at the root (untied).
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

    for idx in 0..config.num_hidden_layers {
        let lp = format!("{prefix}.layers.{idx}");
        out.push(TensorSpec::for_layer(
            TensorRole::LayerInputNorm,
            format!("{lp}.input_layernorm.weight"),
            idx,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::LayerPostAttnNorm,
            format!("{lp}.post_attention_layernorm.weight"),
            idx,
        ));

        if config.is_full_attention(idx) {
            let fa = format!("{lp}.self_attn");
            for (role, suffix) in [
                (TensorRole::FullQProj, "q_proj.weight"),
                (TensorRole::FullKProj, "k_proj.weight"),
                (TensorRole::FullVProj, "v_proj.weight"),
                (TensorRole::FullOProj, "o_proj.weight"),
                (TensorRole::FullQNorm, "q_norm.weight"),
                (TensorRole::FullKNorm, "k_norm.weight"),
            ] {
                out.push(TensorSpec::for_layer(role, format!("{fa}.{suffix}"), idx));
            }
        } else {
            let la = format!("{lp}.linear_attn");
            for (role, suffix) in [
                (TensorRole::LinearInQkvProj, "in_proj_qkv.weight"),
                (TensorRole::LinearInZProj, "in_proj_z.weight"),
                (TensorRole::LinearInBProj, "in_proj_b.weight"),
                (TensorRole::LinearInAProj, "in_proj_a.weight"),
                (TensorRole::LinearOutProj, "out_proj.weight"),
                (TensorRole::LinearConv1d, "conv1d.weight"),
                (TensorRole::LinearDtBias, "dt_bias"),
                (TensorRole::LinearALog, "A_log"),
                (TensorRole::LinearNorm, "norm.weight"),
            ] {
                out.push(TensorSpec::for_layer(role, format!("{la}.{suffix}"), idx));
            }
        }

        // MoE block. v1 assumes every layer uses MoE — `mlp_only_layers` is
        // empty for 35B-A3B per the published config.
        let mp = format!("{lp}.mlp");
        out.push(TensorSpec::for_layer(
            TensorRole::Router,
            format!("{mp}.gate.weight"),
            idx,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::SharedExpertGate,
            format!("{mp}.shared_expert_gate.weight"),
            idx,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::SharedExpertGateProj,
            format!("{mp}.shared_expert.gate_proj.weight"),
            idx,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::SharedExpertUpProj,
            format!("{mp}.shared_expert.up_proj.weight"),
            idx,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::SharedExpertDownProj,
            format!("{mp}.shared_expert.down_proj.weight"),
            idx,
        ));
        // Fused expert tensors: one gate_up_proj + one down_proj per layer,
        // batched across all experts. Note: NO `.weight` suffix in the
        // published Qwen3.6-MoE checkpoint.
        out.push(TensorSpec::for_layer(
            TensorRole::ExpertGateUpProj,
            format!("{mp}.experts.gate_up_proj"),
            idx,
        ));
        out.push(TensorSpec::for_layer(
            TensorRole::ExpertDownProj,
            format!("{mp}.experts.down_proj"),
            idx,
        ));
    }
    out
}

/// Theoretical checkpoint footprint computed analytically from the config —
/// no checkpoint required. Useful to set the registry's `fixed_bytes` budget
/// honestly and to compare against the on-disk total once the download is
/// local. Mixed dtype across roles (F32 norms / A_log; BF16 everything else)
/// matches what the published safetensors actually store.
pub struct CheckpointAccount {
    pub embed_bytes: u64,
    pub final_norm_bytes: u64,
    pub lm_head_bytes: u64,
    pub per_layer_norm_bytes: u64,
    pub full_attn_bytes_per_layer: u64,
    pub linear_attn_bytes_per_layer: u64,
    pub router_bytes_per_layer: u64,
    pub shared_expert_bytes_per_layer: u64,
    pub experts_bytes_per_layer: u64,
    pub num_full_layers: u64,
    pub num_linear_layers: u64,
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
        let per_layer_norm_bytes = cb(TensorRole::LayerInputNorm)
            + cb(TensorRole::LayerPostAttnNorm);

        let full_attn_bytes_per_layer = cb(TensorRole::FullQProj)
            + cb(TensorRole::FullKProj)
            + cb(TensorRole::FullVProj)
            + cb(TensorRole::FullOProj)
            + cb(TensorRole::FullQNorm)
            + cb(TensorRole::FullKNorm);

        let linear_attn_bytes_per_layer = cb(TensorRole::LinearInQkvProj)
            + cb(TensorRole::LinearInZProj)
            + cb(TensorRole::LinearInBProj)
            + cb(TensorRole::LinearInAProj)
            + cb(TensorRole::LinearOutProj)
            + cb(TensorRole::LinearConv1d)
            + cb(TensorRole::LinearDtBias)
            + cb(TensorRole::LinearALog)
            + cb(TensorRole::LinearNorm);

        let router_bytes_per_layer = cb(TensorRole::Router);
        let shared_expert_bytes_per_layer = cb(TensorRole::SharedExpertGate)
            + cb(TensorRole::SharedExpertGateProj)
            + cb(TensorRole::SharedExpertUpProj)
            + cb(TensorRole::SharedExpertDownProj);
        // Fused-expert tensors already include the `* num_experts` factor in
        // their elem count, so we don't multiply again here.
        let experts_bytes_per_layer =
            cb(TensorRole::ExpertGateUpProj) + cb(TensorRole::ExpertDownProj);

        let num_full_layers = config.num_full_attention_layers() as u64;
        let num_linear_layers = config.num_linear_attention_layers() as u64;
        let num_hidden = config.num_hidden_layers as u64;

        let total_bytes = embed_bytes
            + final_norm_bytes
            + lm_head_bytes
            + num_hidden * per_layer_norm_bytes
            + num_full_layers * full_attn_bytes_per_layer
            + num_linear_layers * linear_attn_bytes_per_layer
            + num_hidden
                * (router_bytes_per_layer
                    + shared_expert_bytes_per_layer
                    + experts_bytes_per_layer);

        Self {
            embed_bytes,
            final_norm_bytes,
            lm_head_bytes,
            per_layer_norm_bytes,
            full_attn_bytes_per_layer,
            linear_attn_bytes_per_layer,
            router_bytes_per_layer,
            shared_expert_bytes_per_layer,
            experts_bytes_per_layer,
            num_full_layers,
            num_linear_layers,
            total_bytes,
        }
    }

    /// Same totals projected to an INT4-GPTQ packed footprint. Quantizable
    /// projections (`_proj`, expert MLP, lm_head) shrink to ~4.5 bits/weight
    /// once you fold in the BF16 scale+zero sidecar at group_size=128;
    /// non-quantizable tensors (norms, routers, scalar gates, conv, dt_bias,
    /// A_log) stay at their native size. Approximation, not bit-exact —
    /// useful for VRAM budgeting before the bake exists.
    pub fn project_int4_total_bytes(&self, config: &TextConfig, group_size: u64) -> u64 {
        // Embed and lm_head: lm_head is INT4-quantized by the bake (Qwen3.5
        // already uses this); embed stays BF16 because the runtime reads it
        // for free in the kernel via lookup, not matmul.
        let embed = self.embed_bytes;
        let final_norm = self.final_norm_bytes;
        // 4 bits / weight + 1 BF16 scale + 1 BF16 zero per [gs, gs] tile.
        let lm_head = if config.tie_word_embeddings {
            0
        } else {
            int4_bytes(
                config.vocab_size as u64 * config.hidden_size as u64,
                config.vocab_size as u64,
                config.hidden_size as u64,
                group_size,
            )
        };

        let per_norm = self.per_layer_norm_bytes;
        let mut total = embed + final_norm + lm_head + per_norm * config.num_hidden_layers as u64;

        // Full attention projections (Q,K,V,O) quantize. q_norm/k_norm don't.
        let q_dim = config.num_attention_heads as u64 * config.head_dim as u64;
        let q_proj_out = if config.attn_output_gate { 2 * q_dim } else { q_dim };
        let kv_dim = config.num_key_value_heads as u64 * config.head_dim as u64;
        let hidden = config.hidden_size as u64;
        let head_dim = config.head_dim as u64;
        let full_int4 = int4_bytes(q_proj_out * hidden, q_proj_out, hidden, group_size)
            + int4_bytes(kv_dim * hidden, kv_dim, hidden, group_size)
            + int4_bytes(kv_dim * hidden, kv_dim, hidden, group_size)
            + int4_bytes(hidden * q_dim, hidden, q_dim, group_size)
            + 2 * 2 * head_dim;
        total += self.num_full_layers * full_int4;

        // Linear attention: in_proj_qkv/z/out_proj quantize; in_proj_a/b
        // (per-head, [V_heads, hidden]) often won't align to gs=128 on the
        // V_heads axis, so they stay BF16 — the bake predicate already
        // rejects them on the shape gate. conv/dt_bias/A_log/norm stay too.
        let lin_v_heads = config.linear_num_value_heads as u64;
        let lin_v_hd = config.linear_value_head_dim as u64;
        let lin_k_heads = config.linear_num_key_heads as u64;
        let lin_k_hd = config.linear_key_head_dim as u64;
        let lin_k_dim = lin_k_heads * lin_k_hd;
        let lin_v_dim = lin_v_heads * lin_v_hd;
        let lin_qkv_dim = lin_k_dim + lin_k_dim + lin_v_dim;
        let lin_int4 = int4_bytes(lin_qkv_dim * hidden, lin_qkv_dim, hidden, group_size)
            + int4_bytes(lin_v_dim * hidden, lin_v_dim, hidden, group_size)
            + int4_bytes(hidden * lin_v_dim, hidden, lin_v_dim, group_size);
        let lin_raw_bf16 = 2 * (lin_v_heads * hidden + lin_v_heads * hidden) // a, b
            + 2 * lin_qkv_dim * config.linear_conv_kernel_dim as u64 // conv1d
            + 2 * lin_v_heads // dt_bias
            + 4 * lin_v_heads // A_log f32
            + 4 * lin_k_hd; // norm f32
        total += self.num_linear_layers * (lin_int4 + lin_raw_bf16);

        // Router stays BF16; shared-expert + experts quantize their gate/up/down.
        let router_raw = 2 * config.num_experts as u64 * hidden;
        let scalar_gate_raw = 2 * hidden;
        let moe_int = config.moe_intermediate_size as u64;
        let shared_int = config.shared_expert_intermediate_size as u64;
        let shared_int4 =
            int4_bytes(shared_int * hidden, shared_int, hidden, group_size) * 2 // gate+up
            + int4_bytes(hidden * shared_int, hidden, shared_int, group_size); // down
        // Fused-expert tile dimensions: gate_up_proj is laid out as
        // [E, 2*moe_int, hidden] which we treat as `E` parallel
        // [2*moe_int, hidden] tiles for INT4 packing. Same for down_proj.
        let expert_gate_up = int4_bytes(
            2 * moe_int * hidden,
            2 * moe_int,
            hidden,
            group_size,
        );
        let expert_down = int4_bytes(hidden * moe_int, hidden, moe_int, group_size);
        let per_layer_expert_int4 =
            config.num_experts as u64 * (expert_gate_up + expert_down);
        total += config.num_hidden_layers as u64
            * (router_raw + scalar_gate_raw + shared_int4 + per_layer_expert_int4);
        total
    }
}

fn int4_bytes(elems: u64, out_dim: u64, in_dim: u64, group_size: u64) -> u64 {
    // 2 nibbles per byte
    let packed = elems / 2;
    // BF16 scale+zero per [out/gs, in/gs] tile, two tiles' worth (scale + zero)
    let tiles = (out_dim / group_size).max(1) * (in_dim / group_size).max(1);
    let sidecar = 2 * 2 * tiles;
    packed + sidecar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_35b_a3b() -> TextConfig {
        let layer_types: Vec<String> = (0..40)
            .map(|i| {
                if (i + 1) % 4 == 0 {
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
            hidden_act: crate::config::Activation::Silu,
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
    fn enumeration_counts_match_35b_a3b_geometry() {
        let cfg = config_35b_a3b();
        let specs = expected_tensor_specs(&cfg, DEFAULT_PREFIX);

        let mut count = std::collections::BTreeMap::<&'static str, usize>::new();
        for s in &specs {
            let bucket = match s.role {
                TensorRole::Embed => "embed",
                TensorRole::Norm => "norm",
                TensorRole::LmHead => "lm_head",
                TensorRole::LayerInputNorm | TensorRole::LayerPostAttnNorm => "layer_norm",
                TensorRole::FullQProj
                | TensorRole::FullKProj
                | TensorRole::FullVProj
                | TensorRole::FullOProj
                | TensorRole::FullQNorm
                | TensorRole::FullKNorm => "full",
                TensorRole::LinearInQkvProj
                | TensorRole::LinearInZProj
                | TensorRole::LinearInBProj
                | TensorRole::LinearInAProj
                | TensorRole::LinearOutProj
                | TensorRole::LinearConv1d
                | TensorRole::LinearDtBias
                | TensorRole::LinearALog
                | TensorRole::LinearNorm => "linear",
                TensorRole::Router => "router",
                TensorRole::SharedExpertGate
                | TensorRole::SharedExpertGateProj
                | TensorRole::SharedExpertUpProj
                | TensorRole::SharedExpertDownProj => "shared_expert",
                TensorRole::ExpertGateUpProj | TensorRole::ExpertDownProj => "expert",
            };
            *count.entry(bucket).or_default() += 1;
        }
        assert_eq!(count["embed"], 1);
        assert_eq!(count["norm"], 1);
        assert_eq!(count["lm_head"], 1);
        assert_eq!(count["layer_norm"], 2 * 40);
        assert_eq!(count["full"], 6 * 10);
        assert_eq!(count["linear"], 9 * 30);
        assert_eq!(count["router"], 40);
        assert_eq!(count["shared_expert"], 4 * 40);
        // Fused-expert layout: 2 tensors per layer (gate_up_proj + down_proj)
        // batched across all 256 experts in one slab each.
        assert_eq!(count["expert"], 2 * 40);
    }

    #[test]
    fn account_matches_summed_specs_for_35b_a3b() {
        let cfg = config_35b_a3b();
        let specs = expected_tensor_specs(&cfg, DEFAULT_PREFIX);
        let total_from_specs: u64 = specs
            .iter()
            .map(|s| checkpoint_bytes_for(&cfg, s.role))
            .sum();
        let acct = CheckpointAccount::from_config(&cfg);
        assert_eq!(
            total_from_specs, acct.total_bytes,
            "spec-sum vs analytic-account drift"
        );
        // 35B BF16 disk size sanity: ~65-70 GiB.
        let gib = 1024.0_f64.powi(3);
        let total_gib = acct.total_bytes as f64 / gib;
        assert!(
            (60.0..80.0).contains(&total_gib),
            "expected ~60-80 GiB total, got {total_gib:.3}"
        );
    }

    #[test]
    fn linear_attn_qkv_matches_real_qwen35_0_8b_geometry() {
        // Spot-check against the real 0.8B safetensors: in_proj_qkv = [6144, 1024]
        // when hidden=1024, K=V=16 heads, head_dim=128. The key invariant here
        // is that in_proj_qkv packs Q+K+V channels with Q_heads == K_heads.
        let mut cfg_08b = config_35b_a3b();
        cfg_08b.hidden_size = 1024;
        cfg_08b.linear_num_value_heads = 16; // 0.8B uses 16 V heads
        cfg_08b.num_experts = 1; // unused for this assert
        cfg_08b.moe_intermediate_size = 64;
        cfg_08b.shared_expert_intermediate_size = 64;
        let qkv_elems = checkpoint_elems_for(&cfg_08b, TensorRole::LinearInQkvProj);
        // 16K + 16K + 16V = 48 heads × 128 = 6144 channels × 1024 hidden
        assert_eq!(qkv_elems, 6144 * 1024);
    }

    #[test]
    fn int4_projection_fits_24gib_for_35b_a3b() {
        let cfg = config_35b_a3b();
        let acct = CheckpointAccount::from_config(&cfg);
        let int4 = acct.project_int4_total_bytes(&cfg, 128);
        let gib = 1024.0_f64.powi(3);
        let int4_gib = int4 as f64 / gib;
        // 35B INT4 projection: weights ~17.5 GiB + scales ~0.5 GiB + embed BF16 ~1 GiB.
        // Plan §6 budgets 19 GiB for weights with overhead_factor 1.1.
        assert!(
            (16.0..22.0).contains(&int4_gib),
            "expected ~17-21 GiB INT4 projection, got {int4_gib:.2}"
        );
        // BF16 baseline must be ~3.5x larger than INT4.
        let ratio = acct.total_bytes as f64 / int4 as f64;
        assert!(
            (3.0..4.5).contains(&ratio),
            "BF16/INT4 ratio out of band: {ratio:.2}"
        );
    }
}
