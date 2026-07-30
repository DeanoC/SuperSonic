use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_cli::layers::load_to_gpu;
use crate::qwen36_moe_types::{MtpLayerBuffers, MultiLayerGeom};
use supersonic_runtime::qwen36_moe::layers::LoadedQwen36Layers;
use supersonic_runtime::qwen36_moe::weights::PreparedLmHeadSource;

const MIB: f64 = (1024 * 1024) as f64;

pub(crate) struct Qwen36DecodeSession {
    pub(crate) final_norm_w_buf: GpuBuffer,
    pub(crate) lm_head_w_buf: GpuBuffer,
    pub(crate) logits_buf: GpuBuffer,
    pub(crate) counter_buf: GpuBuffer,
    pub(crate) final_hidden_buf: GpuBuffer,
    pub(crate) mtp_buffers: Option<MtpLayerBuffers>,
    pub(crate) mtp_forward_scratch: Option<crate::qwen36_moe_mtp::MtpForwardScratch>,
    pub(crate) mtp_chain_scratch: Option<crate::qwen36_moe_mtp::MtpChainScratch>,
    pub(crate) embed_w_buf: Option<GpuBuffer>,
    pub(crate) linear_attn_snapshot: Option<crate::qwen36_moe_state::LinearAttnSnapshot>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_decode_session(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    kv_max_t: usize,
    speculative_decode: bool,
    batched_spec_verify: bool,
    persistent_decode: bool,
    max_speculative_tokens: usize,
    loaded_layers: &mut LoadedQwen36Layers,
) -> Result<Qwen36DecodeSession> {
    let mtp_buffers_opt = if speculative_decode {
        match crate::qwen36_moe_cli::mtp_loader::load_mtp_buffers(store, ordinal, geom, kv_max_t)
            .context("load MTP head from bake")?
        {
            Some(mtp) => {
                let verify_mode = if batched_spec_verify {
                    "batched verify"
                } else {
                    "sequential verify"
                };
                println!(
                    "  MTP head: loaded 19 mtp.* tensors (~1.6 GiB BF16) — \
                     speculative draft + {verify_mode} path active."
                );
                Some(mtp)
            }
            None => {
                anyhow::bail!(
                    "--speculative-decode requested but the bake doesn't \
                     include mtp.* tensors. Re-bake against the post-#84 \
                     `oracle/bake_int4.py`, or pull the new release tarball \
                     once the producer workflow at GitHub issue #87 lands."
                );
            }
        }
    } else {
        None
    };

    let prepared_lm_head = supersonic_runtime::qwen36_moe::weights::prepare_lm_head_bf16(
        store,
        text_config,
        weight_prefix,
        geom,
    )
    .context("prepare lm_head BF16 buffer")?;
    match prepared_lm_head.source {
        PreparedLmHeadSource::NativeInt4 => println!(
            "  dequantized lm_head INT4 [{}, {}] to {:.1} MiB BF16",
            geom.vocab,
            geom.hidden,
            prepared_lm_head.lm_head_bf16.len() as f64 / MIB,
        ),
        PreparedLmHeadSource::GgmlKBlock => println!(
            "  dequantized lm_head GGML K-block [{}, {}] to {:.1} MiB BF16",
            geom.vocab,
            geom.hidden,
            prepared_lm_head.lm_head_bf16.len() as f64 / MIB,
        ),
        PreparedLmHeadSource::TiedBf16 | PreparedLmHeadSource::StandaloneBf16 => {}
    }
    let final_norm_bytes = prepared_lm_head.final_norm_bf16;
    let lm_head_bf16_bytes = prepared_lm_head.lm_head_bf16;
    println!(
        "  uploading lm_head BF16 ({:.1} MiB) and final norm ({:.1} KiB) to GPU…",
        lm_head_bf16_bytes.len() as f64 / MIB,
        final_norm_bytes.len() as f64 / 1024.0,
    );
    let final_norm_w_buf = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[geom.hidden as usize],
        &final_norm_bytes,
    )
    .context("upload final_norm_w to GPU")?;
    let lm_head_w_buf = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[geom.vocab as usize, geom.hidden as usize],
        &lm_head_bf16_bytes,
    )
    .context("upload lm_head BF16 to GPU")?;
    let logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.vocab as usize])
        .context("alloc logits_buf on GPU")?;
    let counter_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
        .context("alloc lm_head counter_buf on GPU")?;
    let final_hidden_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.hidden as usize])
        .context("alloc final_hidden_buf on GPU")?;
    drop(lm_head_bf16_bytes);
    drop(final_norm_bytes);

    let mtp_buffers = mtp_buffers_opt;
    let mtp_forward_scratch = if mtp_buffers.is_some() {
        Some(
            crate::qwen36_moe_mtp::alloc_mtp_forward_scratch(ordinal, geom, kv_max_t)
                .context("alloc MTP forward scratch")?,
        )
    } else {
        None
    };
    let mtp_chain_scratch = if mtp_buffers.is_some() {
        Some(
            crate::qwen36_moe_mtp::alloc_mtp_chain_scratch(ordinal, geom)
                .context("alloc MTP chain scratch")?,
        )
    } else {
        None
    };
    let dense_prefill_token_loop =
        std::env::var_os("SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP").is_some();
    let embed_w_buf = if mtp_buffers.is_some() || (persistent_decode && dense_prefill_token_loop) {
        let embed_name = format!("{weight_prefix}.embed_tokens.weight");
        let embed = load_to_gpu(store, ordinal, &embed_name)
            .with_context(|| format!("upload {embed_name} to GPU"))?;
        if mtp_buffers.is_some() {
            println!(
                "  uploaded embed_tokens ({:.0} MiB BF16) and allocated MTP \
                 scratches (K={} drafts/step)",
                (geom.vocab as f64 * geom.hidden as f64 * 2.0) / MIB,
                max_speculative_tokens,
            );
        } else {
            println!(
                "  uploaded embed_tokens ({:.0} MiB BF16) for device-side dense prefill",
                (geom.vocab as f64 * geom.hidden as f64 * 2.0) / MIB,
            );
        }
        Some(embed)
    } else {
        None
    };

    let linear_attn_snapshot = if speculative_decode && batched_spec_verify {
        Some(
            crate::qwen36_moe_state::save_linear_attn_state(ordinal, loaded_layers.layers())
                .context("alloc linear-attn state snapshot for batched spec verify")?,
        )
    } else {
        None
    };
    if linear_attn_snapshot.is_some() {
        println!(
            "  --batched-spec-verify: linear-attn state snapshot allocated \
             (K+1 chains run per spec iter; restore + replay accepted prefix \
             on partial accept)"
        );
    }

    if persistent_decode {
        loaded_layers
            .enable_persistent(ordinal, geom)
            .context("alloc PersistentScratch for --persistent-decode")?;
        let stats = loaded_layers
            .persistent_scratch_stats()
            .expect("persistent scratch was just enabled");
        println!(
            "  --persistent-decode: megakernel scratch allocated \
             (descs={}KiB, workspace={}KiB, ping/pong={}KiB){}",
            stats.descriptor_bytes / 1024,
            stats.workspace_bytes / 1024,
            stats.hidden_bytes / 1024,
            if speculative_decode {
                " — also routes spec-verify chains through persistent"
            } else {
                ""
            },
        );
    }

    Ok(Qwen36DecodeSession {
        final_norm_w_buf,
        lm_head_w_buf,
        logits_buf,
        counter_buf,
        final_hidden_buf,
        mtp_buffers,
        mtp_forward_scratch,
        mtp_chain_scratch,
        embed_w_buf,
        linear_attn_snapshot,
    })
}
