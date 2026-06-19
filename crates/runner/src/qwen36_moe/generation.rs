use std::io::Write as _;
use std::time::Duration;

use anyhow::{Context, Result};
use gpu_hal::{Backend, GpuBuffer};

use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::host::EmbedLookupTiming;
use crate::qwen36_moe_cli::lm_head::{
    launch_lm_head_from_final_hidden_bytes, launch_lm_head_top1_from_final_hidden_bytes,
    launch_top1_from_logits, LmHeadBuffers,
};
use crate::qwen36_moe_cli::output::{
    dump_final_hidden_if_requested, dump_logits_if_requested, emit_final_hidden_tap_if_requested,
    emit_logits_tap_if_requested, print_decoded_token,
};
use crate::qwen36_moe_cli::timing::{Qwen36StageTimingTotals, SamplingParams};
use crate::qwen36_moe_logits::{sample_bf16_logits, XorshiftRng};
use crate::qwen36_moe_types::{DecodeOutputs, MultiLayerGeom};

pub(crate) struct Qwen36GenerationStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) step: usize,
    pub(crate) lm_head_folded: bool,
    pub(crate) lm_head_folded_top1: bool,
    pub(crate) dump_last_logits: bool,
    pub(crate) tokenizer: Option<&'a tokenizers::Tokenizer>,
    pub(crate) sampling: SamplingParams,
    pub(crate) t_embed_step: Duration,
    pub(crate) embed_lookup_timing: Option<EmbedLookupTiming>,
    pub(crate) t_chain_step: Duration,
    pub(crate) outputs: &'a DecodeOutputs,
    pub(crate) final_norm_w_buf: &'a GpuBuffer,
    pub(crate) lm_head_w_buf: &'a GpuBuffer,
    pub(crate) final_hidden_buf: &'a mut GpuBuffer,
    pub(crate) logits_buf: &'a mut GpuBuffer,
    pub(crate) counter_buf: &'a mut GpuBuffer,
    pub(crate) loop_state: &'a mut Qwen36DecodeLoopState,
    pub(crate) rng: &'a mut XorshiftRng,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
}

fn gpu_argmax_enabled(
    sampling: SamplingParams,
    dump_last_logits: bool,
    logits_buf: &GpuBuffer,
) -> bool {
    matches!(logits_buf.backend(), Backend::Metal | Backend::Hip)
        && (sampling.temperature <= 0.0 || sampling.top_k == 1)
        && !dump_last_logits
        && std::env::var_os("SUPERSONIC_QWEN36_DUMP_LOGITS").is_none()
        && std::env::var_os("SUPERSONIC_METAL_DISABLE_QWEN36_LM_HEAD_GPU_ARGMAX").is_none()
}

pub(crate) fn run_generation_step(args: Qwen36GenerationStep<'_>) -> Result<u32> {
    let Qwen36GenerationStep {
        ordinal,
        geom,
        step,
        lm_head_folded,
        lm_head_folded_top1,
        dump_last_logits,
        tokenizer,
        sampling,
        t_embed_step,
        embed_lookup_timing,
        t_chain_step,
        outputs,
        final_norm_w_buf,
        lm_head_w_buf,
        final_hidden_buf,
        logits_buf,
        counter_buf,
        loop_state,
        rng,
        stage_timings,
    } = args;

    dump_final_hidden_if_requested(step, loop_state.position, &outputs.final_hidden_bytes)?;
    let gen_index = loop_state.generated_ids.len();
    let tap_path = outputs.path_label;
    emit_final_hidden_tap_if_requested(
        step,
        gen_index,
        loop_state.position,
        tap_path,
        lm_head_folded,
        &outputs.final_hidden_bytes,
    );

    let t2 = std::time::Instant::now();
    let gpu_argmax = gpu_argmax_enabled(sampling, dump_last_logits, logits_buf);
    let (logits, gpu_next_token) = if gpu_argmax && lm_head_folded_top1 {
        let bytes = counter_buf
            .to_host_bytes()
            .context("d2h folded GPU lm_head top1 token")?;
        let token = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        (None, Some(token))
    } else if gpu_argmax && lm_head_folded {
        let token = launch_top1_from_logits(ordinal, geom, logits_buf, counter_buf)
            .context("folded GPU lm_head argmax")?;
        (None, Some(token))
    } else if gpu_argmax {
        let token = launch_lm_head_top1_from_final_hidden_bytes(
            ordinal,
            geom,
            &outputs.final_hidden_bytes,
            LmHeadBuffers {
                final_norm_w: final_norm_w_buf,
                lm_head_w: lm_head_w_buf,
                final_hidden: final_hidden_buf,
                logits: logits_buf,
                counter: counter_buf,
            },
        )
        .context("standalone GPU lm_head Metal argmax")?;
        (None, Some(token))
    } else if lm_head_folded {
        let bytes = logits_buf
            .to_host_bytes()
            .context("d2h logits from folded GPU lm_head")?;
        (Some(bytes), None)
    } else {
        let bytes = launch_lm_head_from_final_hidden_bytes(
            ordinal,
            geom,
            &outputs.final_hidden_bytes,
            LmHeadBuffers {
                final_norm_w: final_norm_w_buf,
                lm_head_w: lm_head_w_buf,
                final_hidden: final_hidden_buf,
                logits: logits_buf,
                counter: counter_buf,
            },
        )
        .context("standalone GPU lm_head")?;
        (Some(bytes), None)
    };

    if dump_last_logits {
        let logits = logits
            .as_ref()
            .expect("gpu argmax is disabled when dump_last_logits is set");
        loop_state.record_last_logits(logits);
    }
    let t_lm_head_step = t2.elapsed();
    if let Some(logits) = logits.as_ref() {
        dump_logits_if_requested(step, logits)?;
        emit_logits_tap_if_requested(
            step,
            gen_index,
            loop_state.position,
            tap_path,
            lm_head_folded,
            logits,
        );
    }

    let t3 = std::time::Instant::now();
    let next_token = if let Some(token) = gpu_next_token {
        token
    } else {
        let logits = logits
            .as_ref()
            .expect("full logits are present when GPU argmax is disabled");
        sample_bf16_logits(
            logits,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            rng,
        )
    };
    let t_sample_step = t3.elapsed();
    loop_state.generated_ids.push(next_token);

    let t4 = std::time::Instant::now();
    print_decoded_token(tokenizer, next_token);
    std::io::stdout().flush().ok();
    let t_detok_step = t4.elapsed();

    stage_timings.record_generation_step(
        t_embed_step,
        t_chain_step,
        t_lm_head_step,
        t_sample_step,
        t_detok_step,
        outputs,
        embed_lookup_timing,
    );

    Ok(next_token)
}
