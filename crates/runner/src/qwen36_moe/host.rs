use anyhow::{anyhow, Context, Result};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_cli::layers::QWEN36_MOE_INT4_GROUP_SIZE;
use crate::qwen36_moe_logits::dequant_int4_to_bf16_bytes;
use crate::qwen36_moe_types::MultiLayerGeom;

const MIB: f64 = (1024 * 1024) as f64;

/// Look up one row of the embedding table on the host. The full
/// embed_tokens tensor is BF16 `[vocab, hidden]`; this slices the requested
/// row from the mmap-backed raw payload to avoid a full GPU upload.
pub(crate) fn lookup_embed_row(
    store: &BakedStore,
    weight_prefix: &str,
    token_id: usize,
    hidden: usize,
) -> Result<Vec<u8>> {
    let name = format!("{weight_prefix}.embed_tokens.weight");
    let bytes = store
        .raw_bytes(&name)
        .ok_or_else(|| anyhow!("missing {name} in bake"))?;
    let row_bytes = hidden * 2;
    let start = token_id * row_bytes;
    let end = start + row_bytes;
    if end > bytes.len() {
        return Err(anyhow!(
            "embed_tokens row {token_id} out of bounds (need {end} bytes, have {})",
            bytes.len()
        ));
    }
    Ok(bytes[start..end].to_vec())
}

/// Pull the host-side bytes for a tensor used by CPU-side setup paths.
pub(crate) fn host_load_bytes(store: &BakedStore, name: &str) -> Result<Vec<u8>> {
    let raw = store
        .raw_bytes(name)
        .ok_or_else(|| anyhow!("missing {name} in bake"))?;
    Ok(raw.to_vec())
}

/// Load lm_head as BF16 host bytes, handling tied embeddings, standalone
/// BF16, and INT4 GPTQ sidecars.
pub(crate) fn load_lm_head_bf16(
    store: &BakedStore,
    text_config: &TextConfig,
    weight_prefix: &str,
    geom: &MultiLayerGeom,
) -> Result<Vec<u8>> {
    let (lm_name, lm_packed) = if text_config.tie_word_embeddings {
        let n = format!("{weight_prefix}.embed_tokens.weight");
        let b = host_load_bytes(store, &n).context("load tied lm_head from embed_tokens")?;
        (n, b)
    } else {
        let n = "lm_head.weight";
        let b = host_load_bytes(store, n).context("load lm_head")?;
        (n.to_string(), b)
    };
    let scale_name = format!("{lm_name}_int4_scale");
    if store.contains(&scale_name) {
        let zero_name = format!("{lm_name}_int4_zero");
        let scale = host_load_bytes(store, &scale_name).context("load lm_head INT4 scale")?;
        let zero = host_load_bytes(store, &zero_name).context("load lm_head INT4 zero")?;
        let vocab = geom.vocab as usize;
        let hidden = geom.hidden as usize;
        println!(
            "  dequantizing lm_head INT4 [{vocab}, {hidden}] (≈{:.1} MiB → {:.1} MiB BF16)…",
            lm_packed.len() as f64 / MIB,
            (vocab * hidden * 2) as f64 / MIB,
        );
        Ok(dequant_int4_to_bf16_bytes(
            &lm_packed,
            &scale,
            &zero,
            vocab,
            hidden,
            QWEN36_MOE_INT4_GROUP_SIZE as usize,
        ))
    } else {
        Ok(lm_packed)
    }
}
