use anyhow::Result;

use crate::decode_engine::DecodeEngine;
use crate::qwen35_decode_util::token_history_with_next;
use crate::{oracle, prefill_engine, validate};

pub(crate) fn update_qwen35_oracle_decode_delta(
    oracle_output: Option<&oracle::OracleOutput>,
    logits: &[f32],
    step: usize,
    seqlen_offset: usize,
    next_token: u32,
    batch_size: Option<usize>,
    max_delta: &mut f32,
) {
    let Some(oracle) = oracle_output else {
        return;
    };
    if step >= oracle.decode_logits.len() {
        return;
    }

    let oracle_logits = &oracle.decode_logits[step];
    let delta = validate::max_abs_delta(logits, oracle_logits);
    if delta > *max_delta {
        *max_delta = delta;
    }
    if let Some(batch_size) = batch_size {
        eprintln!(
            "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token} batch_size={batch_size}"
        );
    } else {
        eprintln!(
            "[decode] step={step} seq_off={seqlen_offset} delta={delta:.4} token={next_token}"
        );
    }
}

pub(crate) struct Qwen35GpuDecodeValidation<'a> {
    pub(crate) engine: &'a DecodeEngine,
    pub(crate) logits: &'a [f32],
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) generated_ids: &'a [u32],
    pub(crate) next_token: u32,
    pub(crate) native_token: u32,
    pub(crate) step: usize,
    pub(crate) seqlen_offset: usize,
    pub(crate) ordinal: usize,
    pub(crate) kv_chunk_size: usize,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) use_4b_kernel: bool,
}

pub(crate) fn update_qwen35_gpu_decode_validation(
    validation: Qwen35GpuDecodeValidation<'_>,
    gpu_max_delta: &mut f32,
) -> Result<()> {
    let gpu_token_ids = token_history_with_next(
        validation.prompt_ids,
        validation.generated_ids,
        validation.next_token,
    );
    let gpu_logits = prefill_engine::gpu_reference_replay_step(
        &validation.engine.weights(),
        &validation.engine.rotary(),
        &gpu_token_ids,
        validation.ordinal,
        validation.kv_chunk_size,
        validation.prefill_chunk_size,
        validation.use_4b_kernel,
    )?;
    let delta = validate::max_abs_delta(validation.logits, &gpu_logits);
    let gpu_token = DecodeEngine::greedy_sample(&gpu_logits);
    let token_match = if gpu_token == validation.native_token {
        ""
    } else {
        " MISMATCH"
    };
    if delta > *gpu_max_delta {
        *gpu_max_delta = delta;
    }
    let step = validation.step;
    let seqlen_offset = validation.seqlen_offset;
    let native_token = validation.native_token;
    eprintln!(
        "[gpu-validate] step={step} seq_off={seqlen_offset} delta={delta:.4} native_token={native_token} gpu_token={gpu_token}{token_match}"
    );

    Ok(())
}
