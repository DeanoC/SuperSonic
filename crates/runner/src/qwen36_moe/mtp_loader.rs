use anyhow::{anyhow, Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use model_store::BakedStore;

use crate::qwen36_moe_cli::layers::load_to_gpu;
use crate::qwen36_moe_types::{FullAttnKvCache, MtpLayerBuffers, MultiLayerGeom};

/// Load the Qwen3.6-MoE multi-token-prediction (MTP) head from the bake.
/// Used by Phase 6 self-speculative decode (`oracle/qwen36_moe_mtp_oracle.py`
/// is the bit-exact PyTorch reference).
///
/// Returns `Ok(Some(buffers))` when the bake has all 19 `mtp.*` tensors,
/// `Ok(None)` when the bake is pre-PR-#84 and lacks them (production
/// decode is unaffected; only the speculative-decode path is unavailable),
/// or `Err(...)` when the bake is partially-MTP.
///
/// `kv_max_t > 0` allocates a per-layer KV cache for the MTP block,
/// separate from the base layers' KV caches per the vLLM reference.
pub(crate) fn load_mtp_buffers(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    kv_max_t: usize,
) -> Result<Option<MtpLayerBuffers>> {
    let probe = "mtp.fc.weight";
    if !store.contains(probe) {
        return Ok(None);
    }

    let required = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    ];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|n| !store.contains(n))
        .collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "bake has `{probe}` but is missing {} of the other 18 mtp.* tensors \
             (e.g. {}); refusing to load a partial MTP block. Re-bake against \
             `oracle/bake_int4.py` (see GitHub issue #87 for the producer \
             workflow).",
            missing.len(),
            missing.first().copied().unwrap_or("<none>")
        ));
    }

    let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
    let kv_cache = if kv_max_t > 0 {
        let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
            .context("alloc mtp kv_cache_k")?;
        let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
            .context("alloc mtp kv_cache_v")?;
        Some(FullAttnKvCache {
            k: Some(k),
            v: Some(v),
            kv_max_t: kv_max_t as i32,
            // MTP keeps BF16 KV. KV-FP8 + MTP combo is not validated yet.
            kv_scale_k: None,
            kv_scale_v: None,
            kv_shadow_k: None,
            kv_shadow_v: None,
            kv_shadow_start: -1,
            kv_shadow_window: 0,
            virtual_kv_cache_k: None,
            virtual_kv_cache_v: None,
            virtual_kv_max_t: None,
        })
    } else {
        None
    };

    Ok(Some(MtpLayerBuffers {
        pre_fc_norm_hidden_w: load_to_gpu(store, ordinal, "mtp.pre_fc_norm_hidden.weight")?,
        pre_fc_norm_embedding_w: load_to_gpu(store, ordinal, "mtp.pre_fc_norm_embedding.weight")?,
        fc_w: load_to_gpu(store, ordinal, "mtp.fc.weight")?,
        norm_w: load_to_gpu(store, ordinal, "mtp.norm.weight")?,
        input_norm_w: load_to_gpu(store, ordinal, "mtp.layers.0.input_layernorm.weight")?,
        post_attn_norm_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.post_attention_layernorm.weight",
        )?,
        q_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.q_proj.weight")?,
        k_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.k_proj.weight")?,
        v_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.v_proj.weight")?,
        o_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.o_proj.weight")?,
        q_norm_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.q_norm.weight")?,
        k_norm_w: load_to_gpu(store, ordinal, "mtp.layers.0.self_attn.k_norm.weight")?,
        gate_w: load_to_gpu(store, ordinal, "mtp.layers.0.mlp.gate.weight")?,
        gate_up_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.mlp.experts.gate_up_proj")?,
        down_proj_w: load_to_gpu(store, ordinal, "mtp.layers.0.mlp.experts.down_proj")?,
        shared_gate_proj_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        )?,
        shared_up_proj_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        )?,
        shared_down_proj_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        )?,
        shared_expert_gate_w: load_to_gpu(
            store,
            ordinal,
            "mtp.layers.0.mlp.shared_expert_gate.weight",
        )?,
        kv_cache,
    }))
}
