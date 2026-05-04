use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_host::{host_load_bytes, load_lm_head_bf16};
use crate::qwen36_moe_layers::load_to_gpu;
use crate::qwen36_moe_types::{LayerBuffers, MtpLayerBuffers, MultiLayerGeom};

const MIB: f64 = (1024 * 1024) as f64;
const QWEN36_NUM_SPECULATIVE_TOKENS: usize = 3;

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
    pub(crate) persistent_scratch: Option<crate::qwen36_moe_persistent_decode::PersistentScratch>,
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
    layers: &mut Vec<LayerBuffers>,
) -> Result<Qwen36DecodeSession> {
    let mtp_buffers_opt = if speculative_decode {
        match crate::qwen36_moe_mtp_loader::load_mtp_buffers(store, ordinal, geom, kv_max_t)
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

    let final_norm_bytes = host_load_bytes(store, &format!("{weight_prefix}.norm.weight"))
        .context("load final norm")?;
    let lm_head_bf16_bytes = load_lm_head_bf16(store, text_config, weight_prefix, geom)
        .context("prepare lm_head BF16 buffer")?;
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
    let embed_w_buf = if mtp_buffers.is_some() {
        let embed_name = format!("{weight_prefix}.embed_tokens.weight");
        let embed = load_to_gpu(store, ordinal, &embed_name)
            .with_context(|| format!("upload {embed_name} to GPU"))?;
        println!(
            "  uploaded embed_tokens ({:.0} MiB BF16) and allocated MTP \
             scratches (K={} drafts/step)",
            (geom.vocab as f64 * geom.hidden as f64 * 2.0) / MIB,
            QWEN36_NUM_SPECULATIVE_TOKENS,
        );
        Some(embed)
    } else {
        None
    };

    let linear_attn_snapshot = if speculative_decode && batched_spec_verify {
        Some(
            crate::qwen36_moe_state::save_linear_attn_state(ordinal, layers)
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

    let persistent_scratch = if persistent_decode {
        let scratch =
            crate::qwen36_moe_persistent_decode::PersistentScratch::new(ordinal, geom, layers)
                .context("alloc PersistentScratch for --persistent-decode")?;
        println!(
            "  --persistent-decode: megakernel scratch allocated \
             (descs={}KiB, workspace={}KiB, ping/pong={}KiB){}",
            scratch.layer_descs_dev.len_bytes() / 1024,
            scratch.workspace.len_bytes() / 1024,
            scratch.hidden_ping.len_bytes() / 1024,
            if speculative_decode {
                " — also routes spec-verify chains through persistent"
            } else {
                ""
            },
        );
        Some(scratch)
    } else {
        None
    };

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
        persistent_scratch,
    })
}
