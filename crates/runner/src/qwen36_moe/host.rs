use anyhow::{anyhow, Result};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;
use std::time::{Duration, Instant};

use crate::qwen36_moe_types::MultiLayerGeom;
use supersonic_runtime::qwen36_moe::weights::PreparedLmHeadSource;

const MIB: f64 = (1024 * 1024) as f64;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EmbedLookupTiming {
    pub(crate) raw_bytes: Duration,
    pub(crate) copy: Duration,
}

/// Look up one row of the embedding table on the host. The full
/// embed_tokens tensor is BF16 `[vocab, hidden]`; this slices the requested
/// row from the mmap-backed raw payload to avoid a full GPU upload.
pub(crate) fn lookup_embed_row(
    store: &BakedStore,
    weight_prefix: &str,
    token_id: usize,
    hidden: usize,
) -> Result<Vec<u8>> {
    let (row, _) = lookup_embed_row_timed(store, weight_prefix, token_id, hidden)?;
    Ok(row)
}

pub(crate) fn lookup_embed_row_timed(
    store: &BakedStore,
    weight_prefix: &str,
    token_id: usize,
    hidden: usize,
) -> Result<(Vec<u8>, EmbedLookupTiming)> {
    let name = format!("{weight_prefix}.embed_tokens.weight");
    let t_raw = Instant::now();
    let bytes = store
        .raw_bytes(&name)
        .ok_or_else(|| anyhow!("missing {name} in bake"))?;
    let raw_bytes = t_raw.elapsed();
    let row_bytes = hidden * 2;
    let start = token_id * row_bytes;
    let end = start + row_bytes;
    if end > bytes.len() {
        return Err(anyhow!(
            "embed_tokens row {token_id} out of bounds (need {end} bytes, have {})",
            bytes.len()
        ));
    }
    let t_copy = Instant::now();
    let row = bytes[start..end].to_vec();
    let copy = t_copy.elapsed();
    Ok((row, EmbedLookupTiming { raw_bytes, copy }))
}

/// Pull the host-side bytes for a tensor used by CPU-side setup paths.
pub(crate) fn host_load_bytes(store: &BakedStore, name: &str) -> Result<Vec<u8>> {
    supersonic_runtime::qwen36_moe::weights::host_load_bytes(store, name)
}

/// Load lm_head as BF16 host bytes, handling tied embeddings, standalone
/// BF16, and INT4 GPTQ sidecars.
pub(crate) fn load_lm_head_bf16(
    store: &BakedStore,
    text_config: &TextConfig,
    weight_prefix: &str,
    geom: &MultiLayerGeom,
) -> Result<Vec<u8>> {
    let prepared = supersonic_runtime::qwen36_moe::weights::prepare_lm_head_bf16(
        store,
        text_config,
        weight_prefix,
        geom,
    )?;
    match prepared.source {
        PreparedLmHeadSource::NativeInt4 => {
            let vocab = geom.vocab as usize;
            let hidden = geom.hidden as usize;
            println!(
                "  dequantized lm_head INT4 [{vocab}, {hidden}] to {:.1} MiB BF16",
                prepared.lm_head_bf16.len() as f64 / MIB,
            );
        }
        PreparedLmHeadSource::GgmlKBlock => {
            let vocab = geom.vocab as usize;
            let hidden = geom.hidden as usize;
            println!(
                "  dequantized lm_head GGML K-block [{vocab}, {hidden}] to {:.1} MiB BF16",
                prepared.lm_head_bf16.len() as f64 / MIB,
            );
        }
        PreparedLmHeadSource::TiedBf16 | PreparedLmHeadSource::StandaloneBf16 => {}
    }
    Ok(prepared.lm_head_bf16)
}
