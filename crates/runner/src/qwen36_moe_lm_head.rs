use anyhow::{Context, Result};
use gpu_hal::GpuBuffer;

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
