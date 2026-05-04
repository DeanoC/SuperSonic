use anyhow::Result;
use std::time::Instant;

use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::oracle;
use crate::qwen35_decode_modes::Qwen35DecodeModes;
use crate::qwen35_decode_traces::{
    qwen35_persistent_decode_trace_enabled, run_qwen35_persistent_decode_traces,
    Qwen35PersistentDecodeTrace,
};
use crate::qwen35_decode_util::{token_history, token_history_with_next};
use crate::qwen35_decode_validation::update_qwen35_oracle_decode_delta;
use crate::qwen35_kv_trace::trace_kv_cache;
use crate::Cli;

pub(crate) struct Qwen35BatchDecodeStep<'a> {
    pub(crate) cli: &'a Cli,
    pub(crate) engine: &'a mut DecodeEngine,
    pub(crate) decode_modes: &'a Qwen35DecodeModes,
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) generated_ids: &'a mut Vec<u32>,
    pub(crate) batch_next_tokens: &'a mut [u32],
    pub(crate) next_token: u32,
    pub(crate) step: usize,
    pub(crate) seqlen_offset: usize,
    pub(crate) oracle_output: Option<&'a oracle::OracleOutput>,
    pub(crate) trace_kv_cache_enabled: bool,
    pub(crate) ordinal: usize,
    pub(crate) kv_chunk_size: usize,
    pub(crate) use_4b_kernel: bool,
}

pub(crate) struct Qwen35BatchDecodeState<'a> {
    pub(crate) max_delta: &'a mut f32,
    pub(crate) native_decode_timings: &'a mut DecodeStageTimings,
    pub(crate) native_decode_timing_steps: &'a mut usize,
}

pub(crate) fn run_qwen35_batch_decode_step(
    step: Qwen35BatchDecodeStep<'_>,
    state: Qwen35BatchDecodeState<'_>,
) -> Result<u32> {
    if qwen35_persistent_decode_trace_enabled(step.cli) {
        let trace_token_ids =
            token_history_with_next(step.prompt_ids, step.generated_ids, step.next_token);
        let trace_tokens = vec![step.next_token; step.cli.batch_size];
        run_qwen35_persistent_decode_traces(
            step.engine,
            Qwen35PersistentDecodeTrace {
                cli: step.cli,
                trace_token_ids: trace_token_ids.as_slice(),
                trace_tokens: trace_tokens.as_slice(),
                seqlen_offset: step.seqlen_offset,
                ordinal: step.ordinal,
                kv_chunk_size: step.kv_chunk_size,
                use_4b_kernel: step.use_4b_kernel,
                batch_mode: true,
            },
        )?;
    }

    let (batch_logits, batch_timings) = if step.decode_modes.replay_kv_fp8_enabled {
        let token_ids =
            token_history_with_next(step.prompt_ids, step.generated_ids, step.next_token);
        let logits = step.engine.rebuild_prefill_state(&token_ids, true)?;
        (vec![logits; step.cli.batch_size], None)
    } else if step.cli.emit_stage_timings {
        let (logits, timings) = step
            .engine
            .decode_step_batch_with_timings(step.batch_next_tokens, step.seqlen_offset)?;
        (logits, Some(timings))
    } else {
        (
            step.engine
                .decode_step_batch(step.batch_next_tokens, step.seqlen_offset)?,
            None,
        )
    };
    if let Some(timings) = batch_timings {
        state.native_decode_timings.add_assign(timings);
        *state.native_decode_timing_steps += 1;
    }

    let logits = &batch_logits[0];
    update_qwen35_oracle_decode_delta(
        step.oracle_output,
        logits,
        step.step,
        step.seqlen_offset,
        step.next_token,
        Some(step.cli.batch_size),
        state.max_delta,
    );

    let sampling_start = Instant::now();
    for (bi, seq_logits) in batch_logits.iter().enumerate() {
        step.batch_next_tokens[bi] = DecodeEngine::greedy_sample(seq_logits);
    }
    if batch_timings.is_some() {
        state.native_decode_timings.host_sampling_ms +=
            sampling_start.elapsed().as_secs_f64() * 1000.0;
    }

    step.generated_ids.push(step.next_token);
    if step.trace_kv_cache_enabled {
        let cache_token_ids = token_history(step.prompt_ids, step.generated_ids);
        trace_kv_cache(
            step.engine,
            &cache_token_ids,
            step.ordinal,
            step.kv_chunk_size,
            step.cli.prefill_chunk_size,
            step.use_4b_kernel,
            step.cli.kv_fp8,
            step.cli.batch_size,
            step.step,
        )?;
    }

    Ok(step.batch_next_tokens[0])
}
