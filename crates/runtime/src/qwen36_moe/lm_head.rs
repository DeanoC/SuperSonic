use anyhow::{Context, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};

use crate::qwen36_moe::types::MultiLayerGeom;

pub struct LmHeadBuffers<'a> {
    pub final_norm_w: &'a GpuBuffer,
    pub lm_head_w: &'a GpuBuffer,
    pub final_hidden: &'a mut GpuBuffer,
    pub logits: &'a mut GpuBuffer,
    pub counter: &'a mut GpuBuffer,
}

type BufferContract<'a> = (&'a str, Backend, usize, ScalarType, usize, usize);

#[allow(clippy::too_many_arguments)]
fn validate_buffer_metadata(
    label: &str,
    expected_ordinal: usize,
    expected_backend: Backend,
    expected_dtype: ScalarType,
    expected_len: usize,
    actual_backend: Backend,
    actual_ordinal: usize,
    actual_dtype: ScalarType,
    actual_len: usize,
) -> Result<()> {
    if actual_backend != expected_backend {
        anyhow::bail!(
            "{label} backend mismatch: got {actual_backend:?}, expected {expected_backend:?}"
        );
    }
    if actual_ordinal != expected_ordinal {
        anyhow::bail!(
            "{label} device ordinal mismatch: got {actual_ordinal}, expected {expected_ordinal}"
        );
    }
    if actual_dtype != expected_dtype {
        anyhow::bail!("{label} dtype mismatch: got {actual_dtype:?}, expected {expected_dtype:?}");
    }
    if actual_len < expected_len {
        anyhow::bail!(
            "{label} buffer too small: got {actual_len} bytes, need at least {expected_len}"
        );
    }
    Ok(())
}

fn validate_lm_head_buffer_contract(
    ordinal: usize,
    backend: Backend,
    hidden: usize,
    vocab: usize,
    final_hidden_bytes_len: usize,
    buffers: &[BufferContract<'_>],
) -> Result<()> {
    let hidden_bytes = hidden
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("Qwen3.6 lm_head hidden byte size overflow"))?;
    let logits_bytes = vocab
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("Qwen3.6 lm_head logits byte size overflow"))?;
    let lm_head_bytes = hidden
        .checked_mul(vocab)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("Qwen3.6 lm_head weight byte size overflow"))?;
    if final_hidden_bytes_len != hidden_bytes {
        anyhow::bail!(
            "final_hidden_bytes length {final_hidden_bytes_len} != expected {hidden_bytes}"
        );
    }
    for &(label, actual_backend, actual_ordinal, dtype, actual_len, expected_len) in buffers {
        let (expected_dtype, computed_len) = match label {
            "final_hidden" | "final_norm" => (ScalarType::BF16, hidden_bytes),
            "lm_head" => (ScalarType::BF16, lm_head_bytes),
            "logits" => (ScalarType::BF16, logits_bytes),
            "counter" => (ScalarType::U32, std::mem::size_of::<u32>()),
            _ => (dtype, expected_len),
        };
        validate_buffer_metadata(
            label,
            ordinal,
            backend,
            expected_dtype,
            computed_len.max(expected_len),
            actual_backend,
            actual_ordinal,
            dtype,
            actual_len,
        )?;
    }
    Ok(())
}

fn validate_lm_head_buffers(
    ordinal: usize,
    geom: &MultiLayerGeom,
    final_hidden_bytes: &[u8],
    buffers: &LmHeadBuffers<'_>,
) -> Result<()> {
    if geom.hidden <= 0 || geom.vocab <= 0 {
        anyhow::bail!(
            "invalid Qwen3.6 lm_head geometry: hidden={} vocab={}",
            geom.hidden,
            geom.vocab
        );
    }
    let backend = buffers.final_hidden.backend();
    validate_lm_head_buffer_contract(
        ordinal,
        backend,
        geom.hidden as usize,
        geom.vocab as usize,
        final_hidden_bytes.len(),
        &[
            (
                "final_hidden",
                buffers.final_hidden.backend(),
                buffers.final_hidden.device_ordinal(),
                buffers.final_hidden.dtype(),
                buffers.final_hidden.len_bytes(),
                0,
            ),
            (
                "final_norm",
                buffers.final_norm_w.backend(),
                buffers.final_norm_w.device_ordinal(),
                buffers.final_norm_w.dtype(),
                buffers.final_norm_w.len_bytes(),
                0,
            ),
            (
                "lm_head",
                buffers.lm_head_w.backend(),
                buffers.lm_head_w.device_ordinal(),
                buffers.lm_head_w.dtype(),
                buffers.lm_head_w.len_bytes(),
                0,
            ),
            (
                "logits",
                buffers.logits.backend(),
                buffers.logits.device_ordinal(),
                buffers.logits.dtype(),
                buffers.logits.len_bytes(),
                0,
            ),
            (
                "counter",
                buffers.counter.backend(),
                buffers.counter.device_ordinal(),
                buffers.counter.dtype(),
                buffers.counter.len_bytes(),
                0,
            ),
        ],
    )
}

pub fn launch_lm_head_from_final_hidden_bytes(
    ordinal: usize,
    geom: &MultiLayerGeom,
    final_hidden_bytes: &[u8],
    launch_options: &kernel_ffi::prefill_ffi::PrefillFfiLaunchOptions,
    buffers: LmHeadBuffers<'_>,
) -> Result<Vec<u8>> {
    validate_lm_head_buffers(ordinal, geom, final_hidden_bytes, &buffers)?;
    gpu_hal::copy_h2d(
        ordinal,
        buffers.final_hidden.as_mut_ptr(),
        final_hidden_bytes.as_ptr() as *const _,
        final_hidden_bytes.len(),
    )
    .context("h2d final_hidden -> final_hidden_buf")?;
    kernel_ffi::qwen36_moe::lm_head_launch_with_options(
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
        launch_options,
    )
    .context("gpu lm_head launch")?;
    buffers
        .logits
        .to_host_bytes()
        .context("d2h logits from GPU lm_head")
}

pub fn launch_lm_head_top1_from_final_hidden_bytes(
    ordinal: usize,
    geom: &MultiLayerGeom,
    final_hidden_bytes: &[u8],
    launch_options: &kernel_ffi::prefill_ffi::PrefillFfiLaunchOptions,
    buffers: LmHeadBuffers<'_>,
) -> Result<u32> {
    validate_lm_head_buffers(ordinal, geom, final_hidden_bytes, &buffers)?;
    gpu_hal::copy_h2d(
        ordinal,
        buffers.final_hidden.as_mut_ptr(),
        final_hidden_bytes.as_ptr() as *const _,
        final_hidden_bytes.len(),
    )
    .context("h2d final_hidden -> final_hidden_buf")?;
    kernel_ffi::qwen36_moe::lm_head_launch_with_options(
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
        launch_options,
    )
    .context("gpu lm_head launch")?;
    launch_top1_from_logits(ordinal, geom, buffers.logits, buffers.counter)
}

pub fn launch_top1_from_logits(
    ordinal: usize,
    geom: &MultiLayerGeom,
    logits: &GpuBuffer,
    out_index: &mut GpuBuffer,
) -> Result<u32> {
    if geom.vocab <= 0 {
        anyhow::bail!("invalid Qwen3.6 lm_head vocab size {}", geom.vocab);
    }
    validate_buffer_metadata(
        "logits",
        ordinal,
        logits.backend(),
        ScalarType::BF16,
        geom.vocab as usize * 2,
        logits.backend(),
        logits.device_ordinal(),
        logits.dtype(),
        logits.len_bytes(),
    )?;
    validate_buffer_metadata(
        "top1_out",
        ordinal,
        logits.backend(),
        ScalarType::U32,
        std::mem::size_of::<u32>(),
        out_index.backend(),
        out_index.device_ordinal(),
        out_index.dtype(),
        out_index.len_bytes(),
    )?;
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
        .context("d2h greedy token from GPU argmax")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(crate) fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "BF16 bytes must be even");
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_hal::ScalarType;

    fn tiny_geom() -> MultiLayerGeom {
        MultiLayerGeom {
            hidden: 2,
            vocab: 2,
            num_layers: 0,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_theta: 10_000.0,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_kernel_dim: 2,
            num_experts: 1,
            moe_intermediate: 2,
            shared_intermediate: 2,
            top_k: 1,
        }
    }

    #[test]
    fn lm_head_contract_rejects_wrong_host_hidden_length() {
        let err = validate_lm_head_buffer_contract(
            0,
            Backend::Hip,
            8,
            16,
            18,
            &[
                ("final_hidden", Backend::Hip, 0, ScalarType::BF16, 16, 16),
                ("final_norm", Backend::Hip, 0, ScalarType::BF16, 16, 16),
                ("lm_head", Backend::Hip, 0, ScalarType::BF16, 256, 256),
                ("logits", Backend::Hip, 0, ScalarType::BF16, 32, 32),
                ("counter", Backend::Hip, 0, ScalarType::U32, 4, 4),
            ],
        )
        .expect_err("oversized hidden input must fail before copy");
        assert!(err.to_string().contains("final_hidden_bytes"));
    }

    #[test]
    fn lm_head_contract_rejects_buffer_dtype_device_and_capacity() {
        for (label, backend, device, dtype, actual, expected) in [
            ("final_hidden", Backend::Hip, 0, ScalarType::F32, 16, 16),
            ("logits", Backend::Hip, 1, ScalarType::BF16, 32, 32),
            ("counter", Backend::Metal, 0, ScalarType::U32, 2, 4),
        ] {
            let err = validate_buffer_metadata(
                label,
                0,
                Backend::Hip,
                ScalarType::BF16,
                expected,
                backend,
                device,
                dtype,
                actual,
            )
            .expect_err("malformed buffer metadata must fail");
            assert!(err.to_string().contains(label));
        }
    }

    #[test]
    fn public_lm_head_rejects_bad_metadata_before_copying_hidden() {
        let ordinal = 0;
        let geom = tiny_geom();
        let final_norm_w =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2]).expect("alloc final norm");
        let lm_head_w =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, 2]).expect("alloc lm head");
        let mut final_hidden =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2]).expect("alloc final hidden");
        let mut logits =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[2]).expect("alloc wrong logits");
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counter");
        let nonzero_hidden = [1u8, 2, 3, 4];

        let err = launch_lm_head_from_final_hidden_bytes(
            ordinal,
            &geom,
            &nonzero_hidden,
            &kernel_ffi::prefill_ffi::PrefillFfiLaunchOptions::default(),
            LmHeadBuffers {
                final_norm_w: &final_norm_w,
                lm_head_w: &lm_head_w,
                final_hidden: &mut final_hidden,
                logits: &mut logits,
                counter: &mut counter,
            },
        )
        .expect_err("wrong logits dtype must fail before H2D copy");

        assert!(err.to_string().contains("logits dtype mismatch"));
        assert_eq!(
            final_hidden
                .to_host_bytes()
                .expect("download untouched hidden"),
            vec![0u8; 4]
        );
    }

    #[test]
    fn public_top1_rejects_short_output_before_launch() {
        let ordinal = 0;
        let geom = tiny_geom();
        let logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2]).expect("alloc logits");
        let mut out =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[1]).expect("alloc malformed top1");

        let err = launch_top1_from_logits(ordinal, &geom, &logits, &mut out)
            .expect_err("malformed top1 output must fail before launch");

        assert!(err.to_string().contains("top1_out"));
    }
}
