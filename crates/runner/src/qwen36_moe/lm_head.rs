use anyhow::{Context, Result};
use gpu_hal::{Backend, GpuBuffer};

use crate::qwen36_moe_types::MultiLayerGeom;

pub(crate) struct LmHeadBuffers<'a> {
    pub(crate) final_norm_w: &'a GpuBuffer,
    pub(crate) lm_head_w: &'a GpuBuffer,
    pub(crate) final_hidden: &'a mut GpuBuffer,
    pub(crate) logits: &'a mut GpuBuffer,
    pub(crate) counter: &'a mut GpuBuffer,
}

pub(crate) fn launch_lm_head_from_final_hidden_bytes(
    ordinal: usize,
    geom: &MultiLayerGeom,
    final_hidden_bytes: &[u8],
    buffers: LmHeadBuffers<'_>,
) -> Result<Vec<u8>> {
    gpu_hal::copy_h2d(
        ordinal,
        buffers.final_hidden.as_mut_ptr(),
        final_hidden_bytes.as_ptr() as *const _,
        final_hidden_bytes.len(),
    )
    .context("h2d final_hidden -> final_hidden_buf")?;
    kernel_ffi::qwen36_moe::lm_head_launch(
        ordinal,
        geom.hidden,
        geom.vocab,
        geom.rms_norm_eps,
        buffers.final_hidden,
        buffers.final_norm_w,
        buffers.lm_head_w,
        buffers.logits,
        None,
        buffers.counter,
    )
    .context("gpu lm_head launch")?;
    buffers
        .logits
        .to_host_bytes()
        .context("d2h logits from GPU lm_head")
}

pub(crate) fn launch_lm_head_top1_from_final_hidden_bytes(
    ordinal: usize,
    geom: &MultiLayerGeom,
    final_hidden_bytes: &[u8],
    buffers: LmHeadBuffers<'_>,
) -> Result<u32> {
    gpu_hal::copy_h2d(
        ordinal,
        buffers.final_hidden.as_mut_ptr(),
        final_hidden_bytes.as_ptr() as *const _,
        final_hidden_bytes.len(),
    )
    .context("h2d final_hidden -> final_hidden_buf")?;
    kernel_ffi::qwen36_moe::lm_head_launch(
        ordinal,
        geom.hidden,
        geom.vocab,
        geom.rms_norm_eps,
        buffers.final_hidden,
        buffers.final_norm_w,
        buffers.lm_head_w,
        buffers.logits,
        None,
        buffers.counter,
    )
    .context("gpu lm_head launch")?;
    launch_top1_from_logits(ordinal, geom, buffers.logits, buffers.counter)
}

pub(crate) fn launch_top1_from_logits(
    ordinal: usize,
    geom: &MultiLayerGeom,
    logits: &GpuBuffer,
    out_index: &mut GpuBuffer,
) -> Result<u32> {
    match logits.backend() {
        Backend::Metal => {
            kernel_ffi::metal_argmax_bf16_into(logits, out_index, geom.vocab as usize)
                .context("metal argmax over lm_head logits")?;
            if kernel_ffi::prefill_ffi::metal_batch_is_active() {
                kernel_ffi::prefill_ffi::flush_metal_batch().context("flush metal argmax batch")?;
            }
        }
        Backend::Hip => {
            kernel_ffi::prefill_ffi::argmax_bf16_rows(
                ordinal,
                1,
                geom.vocab as usize,
                logits,
                out_index,
            )
            .context("HIP argmax over lm_head logits")?;
        }
        other => anyhow::bail!("GPU argmax over lm_head logits is not available for {other:?}"),
    }
    let bytes = out_index
        .to_host_bytes()
        .context("d2h greedy token from Metal argmax")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
