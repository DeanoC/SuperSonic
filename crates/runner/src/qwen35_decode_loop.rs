use anyhow::Result;
use std::time::Instant;

use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::oracle;
use crate::qwen35_decode_batch::{
    run_qwen35_batch_decode_step, Qwen35BatchDecodeState, Qwen35BatchDecodeStep,
};
use crate::qwen35_decode_modes::Qwen35DecodeModes;
use crate::qwen35_decode_single::{
    run_qwen35_single_decode_step, Qwen35SingleDecodeState, Qwen35SingleDecodeStep,
};
use crate::qwen35_prefill::HostLmHeadRescorer;
use crate::Cli;

pub(crate) struct Qwen35DecodeLoop<'a> {
    pub(crate) cli: &'a Cli,
    pub(crate) engine: &'a mut DecodeEngine,
    pub(crate) decode_modes: &'a Qwen35DecodeModes,
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) eos_ids: &'a [u32],
    pub(crate) next_token: u32,
    pub(crate) oracle_output: Option<&'a oracle::OracleOutput>,
    pub(crate) host_lm_head_rescorer: Option<&'a HostLmHeadRescorer>,
    pub(crate) allow_host_lm_head_rescore: bool,
    pub(crate) trace_kv_cache_enabled: bool,
    pub(crate) gpu_validate_enabled: bool,
    pub(crate) cuda_08b_hero_enabled: bool,
    pub(crate) ordinal: usize,
    pub(crate) kv_chunk_size: usize,
    pub(crate) use_4b_kernel: bool,
}

pub(crate) struct Qwen35DecodeLoopOutput {
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) decode_ms: f64,
    pub(crate) max_delta: f32,
    pub(crate) gpu_max_delta: f32,
    pub(crate) native_decode_timings: DecodeStageTimings,
    pub(crate) native_decode_timing_steps: usize,
}

pub(crate) fn run_qwen35_decode_loop(
    decode: Qwen35DecodeLoop<'_>,
) -> Result<Qwen35DecodeLoopOutput> {
    let seqlen_start = decode.prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut max_delta = 0.0f32;
    let mut gpu_max_delta = 0.0f32;
    let mut native_decode_timings = DecodeStageTimings::default();
    let mut native_decode_timing_steps = 0usize;
    let mut next_token = decode.next_token;
    let mut batch_next_tokens: Vec<u32> = vec![next_token; decode.cli.batch_size];

    let decode_start = Instant::now();
    for step in 0..decode.cli.max_new_tokens {
        if decode.eos_ids.contains(&next_token) {
            break;
        }

        let seqlen_offset = seqlen_start + step;

        if decode.cli.batch_size > 1 {
            next_token = run_qwen35_batch_decode_step(
                Qwen35BatchDecodeStep {
                    cli: decode.cli,
                    engine: decode.engine,
                    decode_modes: decode.decode_modes,
                    prompt_ids: decode.prompt_ids,
                    generated_ids: &mut generated_ids,
                    batch_next_tokens: &mut batch_next_tokens,
                    next_token,
                    step,
                    seqlen_offset,
                    oracle_output: decode.oracle_output,
                    trace_kv_cache_enabled: decode.trace_kv_cache_enabled,
                    ordinal: decode.ordinal,
                    kv_chunk_size: decode.kv_chunk_size,
                    use_4b_kernel: decode.use_4b_kernel,
                },
                Qwen35BatchDecodeState {
                    max_delta: &mut max_delta,
                    native_decode_timings: &mut native_decode_timings,
                    native_decode_timing_steps: &mut native_decode_timing_steps,
                },
            )?;
        } else {
            next_token = run_qwen35_single_decode_step(
                Qwen35SingleDecodeStep {
                    cli: decode.cli,
                    engine: decode.engine,
                    decode_modes: decode.decode_modes,
                    prompt_ids: decode.prompt_ids,
                    generated_ids: &mut generated_ids,
                    next_token,
                    step,
                    seqlen_offset,
                    oracle_output: decode.oracle_output,
                    host_lm_head_rescorer: decode.host_lm_head_rescorer,
                    allow_host_lm_head_rescore: decode.allow_host_lm_head_rescore,
                    trace_kv_cache_enabled: decode.trace_kv_cache_enabled,
                    gpu_validate_enabled: decode.gpu_validate_enabled,
                    cuda_08b_hero_enabled: decode.cuda_08b_hero_enabled,
                    ordinal: decode.ordinal,
                    kv_chunk_size: decode.kv_chunk_size,
                    use_4b_kernel: decode.use_4b_kernel,
                },
                Qwen35SingleDecodeState {
                    max_delta: &mut max_delta,
                    gpu_max_delta: &mut gpu_max_delta,
                    native_decode_timings: &mut native_decode_timings,
                    native_decode_timing_steps: &mut native_decode_timing_steps,
                },
            )?;
        }
    }

    Ok(Qwen35DecodeLoopOutput {
        generated_ids,
        decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
        max_delta,
        gpu_max_delta,
        native_decode_timings,
        native_decode_timing_steps,
    })
}
