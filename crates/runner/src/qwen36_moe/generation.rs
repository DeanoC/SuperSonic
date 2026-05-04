use std::io::Write as _;
use std::time::Duration;

use anyhow::{Context, Result};
use gpu_hal::GpuBuffer;

use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::lm_head::{launch_lm_head_from_final_hidden_bytes, LmHeadBuffers};
use crate::qwen36_moe_cli::output::{
    dump_final_hidden_if_requested, dump_logits_if_requested, print_decoded_token,
};
use crate::qwen36_moe_cli::timing::{Qwen36StageTimingTotals, SamplingParams};
use crate::qwen36_moe_logits::{sample_bf16_logits, XorshiftRng};
use crate::qwen36_moe_types::{DecodeOutputs, MultiLayerGeom};

pub(crate) struct Qwen36GenerationStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) step: usize,
    pub(crate) lm_head_folded: bool,
    pub(crate) dump_last_logits: bool,
    pub(crate) tokenizer: Option<&'a tokenizers::Tokenizer>,
    pub(crate) sampling: SamplingParams,
    pub(crate) t_embed_step: Duration,
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

pub(crate) fn run_generation_step(args: Qwen36GenerationStep<'_>) -> Result<u32> {
    let Qwen36GenerationStep {
        ordinal,
        geom,
        step,
        lm_head_folded,
        dump_last_logits,
        tokenizer,
        sampling,
        t_embed_step,
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

    let t2 = std::time::Instant::now();
    let logits = if lm_head_folded {
        logits_buf
            .to_host_bytes()
            .context("d2h logits from folded GPU lm_head")?
    } else {
        launch_lm_head_from_final_hidden_bytes(
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
        .context("standalone GPU lm_head")?
    };
    if dump_last_logits {
        loop_state.record_last_logits(&logits);
    }
    let t_lm_head_step = t2.elapsed();
    dump_logits_if_requested(step, &logits)?;

    let t3 = std::time::Instant::now();
    let next_token = sample_bf16_logits(
        &logits,
        sampling.temperature,
        sampling.top_k,
        sampling.top_p,
        rng,
    );
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
    );

    Ok(next_token)
}
