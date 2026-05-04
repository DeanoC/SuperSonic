use anyhow::Result;
use std::time::Instant;

use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::oracle;
use crate::prefill_engine;
use crate::qwen35_component_decode::{run_qwen35_component_single_decode, Qwen35ComponentDecode};
use crate::qwen35_decode_modes::Qwen35DecodeModes;
use crate::qwen35_decode_traces::{
    qwen35_persistent_decode_trace_enabled, run_qwen35_persistent_decode_traces,
    Qwen35PersistentDecodeTrace,
};
use crate::qwen35_decode_util::{token_history, token_history_with_next};
use crate::qwen35_decode_validation::{
    update_qwen35_gpu_decode_validation, update_qwen35_oracle_decode_delta,
    Qwen35GpuDecodeValidation,
};
use crate::qwen35_kv_trace::trace_kv_cache;
use crate::qwen35_prefill::{sample_qwen_logits_with_rescore, HostLmHeadRescorer};
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
            if qwen35_persistent_decode_trace_enabled(decode.cli) {
                let trace_token_ids =
                    token_history_with_next(decode.prompt_ids, &generated_ids, next_token);
                let trace_tokens = vec![next_token; decode.cli.batch_size];
                run_qwen35_persistent_decode_traces(
                    decode.engine,
                    Qwen35PersistentDecodeTrace {
                        cli: decode.cli,
                        trace_token_ids: trace_token_ids.as_slice(),
                        trace_tokens: trace_tokens.as_slice(),
                        seqlen_offset,
                        ordinal: decode.ordinal,
                        kv_chunk_size: decode.kv_chunk_size,
                        use_4b_kernel: decode.use_4b_kernel,
                        batch_mode: true,
                    },
                )?;
            }
            let (batch_logits, batch_timings) = if decode.decode_modes.replay_kv_fp8_enabled {
                let token_ids =
                    token_history_with_next(decode.prompt_ids, &generated_ids, next_token);
                let logits = decode.engine.rebuild_prefill_state(&token_ids, true)?;
                (vec![logits; decode.cli.batch_size], None)
            } else if decode.cli.emit_stage_timings {
                let (logits, timings) = decode
                    .engine
                    .decode_step_batch_with_timings(&batch_next_tokens, seqlen_offset)?;
                (logits, Some(timings))
            } else {
                (
                    decode
                        .engine
                        .decode_step_batch(&batch_next_tokens, seqlen_offset)?,
                    None,
                )
            };
            if let Some(timings) = batch_timings {
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
            }

            let logits = &batch_logits[0];
            update_qwen35_oracle_decode_delta(
                decode.oracle_output,
                logits,
                step,
                seqlen_offset,
                next_token,
                Some(decode.cli.batch_size),
                &mut max_delta,
            );

            let sampling_start = Instant::now();
            for (bi, seq_logits) in batch_logits.iter().enumerate() {
                batch_next_tokens[bi] = DecodeEngine::greedy_sample(seq_logits);
            }
            if batch_timings.is_some() {
                native_decode_timings.host_sampling_ms +=
                    sampling_start.elapsed().as_secs_f64() * 1000.0;
            }

            generated_ids.push(next_token);
            if decode.trace_kv_cache_enabled {
                let cache_token_ids = token_history(decode.prompt_ids, &generated_ids);
                trace_kv_cache(
                    decode.engine,
                    &cache_token_ids,
                    decode.ordinal,
                    decode.kv_chunk_size,
                    decode.cli.prefill_chunk_size,
                    decode.use_4b_kernel,
                    decode.cli.kv_fp8,
                    decode.cli.batch_size,
                    step,
                )?;
            }
            next_token = batch_next_tokens[0];
        } else {
            let mut maybe_fast_token = None;
            let mut can_rescore_with_normed = false;
            let logits = if decode.decode_modes.cuda_fast_greedy_enabled {
                let (token, timings) = decode
                    .engine
                    .decode_step_cuda_fast_greedy(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if decode.decode_modes.metal_fast_greedy_enabled {
                let token = decode
                    .engine
                    .decode_step_metal_fast_greedy(next_token, seqlen_offset)?;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if decode.cuda_08b_hero_enabled {
                let (token, timings) = decode
                    .engine
                    .decode_step_cuda_08b_hero(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                maybe_fast_token = Some(token);
                Vec::new()
            } else if decode.decode_modes.replay_decode_enabled {
                let token_ids =
                    token_history_with_next(decode.prompt_ids, &generated_ids, next_token);
                prefill_engine::gpu_reference_replay_step(
                    &decode.engine.weights(),
                    &decode.engine.rotary(),
                    &token_ids,
                    decode.ordinal,
                    decode.kv_chunk_size,
                    decode.cli.prefill_chunk_size,
                    decode.use_4b_kernel,
                )?
            } else if decode.decode_modes.replay_kv_fp8_enabled {
                let token_ids =
                    token_history_with_next(decode.prompt_ids, &generated_ids, next_token);
                decode.engine.rebuild_prefill_state(&token_ids, false)?
            } else if decode.decode_modes.component_single_decode_enabled {
                run_qwen35_component_single_decode(
                    decode.engine,
                    Qwen35ComponentDecode {
                        cli: decode.cli,
                        prompt_ids: decode.prompt_ids,
                        generated_ids: &generated_ids,
                        next_token,
                        seqlen_offset,
                        ordinal: decode.ordinal,
                        kv_chunk_size: decode.kv_chunk_size,
                        use_4b_kernel: decode.use_4b_kernel,
                    },
                )?
            } else if decode.decode_modes.kernel_single_decode_enabled {
                if qwen35_persistent_decode_trace_enabled(decode.cli) {
                    let trace_token_ids =
                        token_history_with_next(decode.prompt_ids, &generated_ids, next_token);
                    run_qwen35_persistent_decode_traces(
                        decode.engine,
                        Qwen35PersistentDecodeTrace {
                            cli: decode.cli,
                            trace_token_ids: trace_token_ids.as_slice(),
                            trace_tokens: &[next_token],
                            seqlen_offset,
                            ordinal: decode.ordinal,
                            kv_chunk_size: decode.kv_chunk_size,
                            use_4b_kernel: decode.use_4b_kernel,
                            batch_mode: false,
                        },
                    )?;
                }
                if decode.cli.emit_stage_timings {
                    let (logits, timings) = decode
                        .engine
                        .decode_step_4b_single_kernel_with_timings(next_token, seqlen_offset)?;
                    native_decode_timings.add_assign(timings);
                    native_decode_timing_steps += 1;
                    can_rescore_with_normed = true;
                    logits
                } else {
                    can_rescore_with_normed = true;
                    decode
                        .engine
                        .decode_step_batch(&[next_token], seqlen_offset)?
                        .remove(0)
                }
            } else if decode.cli.emit_stage_timings {
                let (logits, timings) = decode
                    .engine
                    .decode_step_with_timings(next_token, seqlen_offset)?;
                native_decode_timings.add_assign(timings);
                native_decode_timing_steps += 1;
                can_rescore_with_normed = true;
                logits
            } else {
                can_rescore_with_normed = true;
                decode.engine.decode_step(next_token, seqlen_offset)?
            };
            let native_token = if let Some(token) = maybe_fast_token {
                token
            } else {
                let normed = if can_rescore_with_normed && decode.allow_host_lm_head_rescore {
                    Some(decode.engine.last_normed_host_f32()?)
                } else {
                    None
                };
                sample_qwen_logits_with_rescore(
                    &logits,
                    normed.as_deref(),
                    decode
                        .host_lm_head_rescorer
                        .filter(|_| decode.allow_host_lm_head_rescore),
                )?
            };

            update_qwen35_oracle_decode_delta(
                decode.oracle_output,
                &logits,
                step,
                seqlen_offset,
                next_token,
                None,
                &mut max_delta,
            );

            if decode.gpu_validate_enabled {
                update_qwen35_gpu_decode_validation(
                    Qwen35GpuDecodeValidation {
                        engine: decode.engine,
                        logits: &logits,
                        prompt_ids: decode.prompt_ids,
                        generated_ids: &generated_ids,
                        next_token,
                        native_token,
                        step,
                        seqlen_offset,
                        ordinal: decode.ordinal,
                        kv_chunk_size: decode.kv_chunk_size,
                        prefill_chunk_size: decode.cli.prefill_chunk_size,
                        use_4b_kernel: decode.use_4b_kernel,
                    },
                    &mut gpu_max_delta,
                )?;
            }

            generated_ids.push(next_token);
            next_token = native_token;

            if decode.trace_kv_cache_enabled {
                let cache_token_ids = token_history(decode.prompt_ids, &generated_ids);
                trace_kv_cache(
                    decode.engine,
                    &cache_token_ids,
                    decode.ordinal,
                    decode.kv_chunk_size,
                    decode.cli.prefill_chunk_size,
                    decode.use_4b_kernel,
                    decode.cli.kv_fp8,
                    decode.cli.batch_size,
                    step,
                )?;
            }
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
