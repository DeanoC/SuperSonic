use anyhow::Result;

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

pub(crate) struct Qwen35SingleDecodeStep<'a> {
    pub(crate) cli: &'a Cli,
    pub(crate) engine: &'a mut DecodeEngine,
    pub(crate) decode_modes: &'a Qwen35DecodeModes,
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) generated_ids: &'a mut Vec<u32>,
    pub(crate) next_token: u32,
    pub(crate) step: usize,
    pub(crate) seqlen_offset: usize,
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

pub(crate) struct Qwen35SingleDecodeState<'a> {
    pub(crate) max_delta: &'a mut f32,
    pub(crate) gpu_max_delta: &'a mut f32,
    pub(crate) native_decode_timings: &'a mut DecodeStageTimings,
    pub(crate) native_decode_timing_steps: &'a mut usize,
}

pub(crate) fn run_qwen35_single_decode_step(
    step: Qwen35SingleDecodeStep<'_>,
    state: Qwen35SingleDecodeState<'_>,
) -> Result<u32> {
    let mut maybe_fast_token = None;
    let mut can_rescore_with_normed = false;
    let logits = if step.decode_modes.hip_fast_greedy_enabled {
        let (token, timings) = step
            .engine
            .decode_step_hip_fast_greedy(step.next_token, step.seqlen_offset)?;
        state.native_decode_timings.add_assign(timings);
        *state.native_decode_timing_steps += 1;
        maybe_fast_token = Some(token);
        Vec::new()
    } else if step.decode_modes.cuda_fast_greedy_enabled {
        let (token, timings) = step
            .engine
            .decode_step_cuda_fast_greedy(step.next_token, step.seqlen_offset)?;
        state.native_decode_timings.add_assign(timings);
        *state.native_decode_timing_steps += 1;
        maybe_fast_token = Some(token);
        Vec::new()
    } else if step.decode_modes.metal_fast_greedy_enabled {
        let token = step
            .engine
            .decode_step_metal_fast_greedy(step.next_token, step.seqlen_offset)?;
        maybe_fast_token = Some(token);
        Vec::new()
    } else if step.cuda_08b_hero_enabled {
        let (token, timings) = step
            .engine
            .decode_step_cuda_08b_hero(step.next_token, step.seqlen_offset)?;
        state.native_decode_timings.add_assign(timings);
        *state.native_decode_timing_steps += 1;
        maybe_fast_token = Some(token);
        Vec::new()
    } else if step.decode_modes.replay_decode_enabled {
        let token_ids =
            token_history_with_next(step.prompt_ids, step.generated_ids, step.next_token);
        prefill_engine::gpu_reference_replay_step(
            &step.engine.weights(),
            &step.engine.rotary(),
            &token_ids,
            step.ordinal,
            step.kv_chunk_size,
            step.cli.prefill_chunk_size,
            step.use_4b_kernel,
        )?
    } else if step.decode_modes.replay_kv_fp8_enabled {
        let token_ids =
            token_history_with_next(step.prompt_ids, step.generated_ids, step.next_token);
        step.engine.rebuild_prefill_state(&token_ids, false)?
    } else if step.decode_modes.component_single_decode_enabled {
        run_qwen35_component_single_decode(
            step.engine,
            Qwen35ComponentDecode {
                cli: step.cli,
                prompt_ids: step.prompt_ids,
                generated_ids: step.generated_ids,
                next_token: step.next_token,
                seqlen_offset: step.seqlen_offset,
                ordinal: step.ordinal,
                kv_chunk_size: step.kv_chunk_size,
                use_4b_kernel: step.use_4b_kernel,
            },
        )?
    } else if step.decode_modes.kernel_single_decode_enabled {
        if qwen35_persistent_decode_trace_enabled(step.cli) {
            let trace_token_ids =
                token_history_with_next(step.prompt_ids, step.generated_ids, step.next_token);
            run_qwen35_persistent_decode_traces(
                step.engine,
                Qwen35PersistentDecodeTrace {
                    cli: step.cli,
                    trace_token_ids: trace_token_ids.as_slice(),
                    trace_tokens: &[step.next_token],
                    seqlen_offset: step.seqlen_offset,
                    ordinal: step.ordinal,
                    kv_chunk_size: step.kv_chunk_size,
                    use_4b_kernel: step.use_4b_kernel,
                    batch_mode: false,
                },
            )?;
        }
        if step.cli.emit_stage_timings {
            let (logits, timings) = step
                .engine
                .decode_step_4b_single_kernel_with_timings(step.next_token, step.seqlen_offset)?;
            state.native_decode_timings.add_assign(timings);
            *state.native_decode_timing_steps += 1;
            can_rescore_with_normed = true;
            logits
        } else {
            can_rescore_with_normed = true;
            step.engine
                .decode_step_batch(&[step.next_token], step.seqlen_offset)?
                .remove(0)
        }
    } else if step.cli.emit_stage_timings {
        let (logits, timings) = step
            .engine
            .decode_step_with_timings(step.next_token, step.seqlen_offset)?;
        state.native_decode_timings.add_assign(timings);
        *state.native_decode_timing_steps += 1;
        can_rescore_with_normed = true;
        logits
    } else {
        can_rescore_with_normed = true;
        step.engine
            .decode_step(step.next_token, step.seqlen_offset)?
    };

    let native_token = if let Some(token) = maybe_fast_token {
        token
    } else {
        let normed = if can_rescore_with_normed && step.allow_host_lm_head_rescore {
            Some(step.engine.last_normed_host_f32()?)
        } else {
            None
        };
        sample_qwen_logits_with_rescore(
            &logits,
            normed.as_deref(),
            step.host_lm_head_rescorer
                .filter(|_| step.allow_host_lm_head_rescore),
        )?
    };

    update_qwen35_oracle_decode_delta(
        step.oracle_output,
        &logits,
        step.step,
        step.seqlen_offset,
        step.next_token,
        None,
        state.max_delta,
    );

    if step.gpu_validate_enabled {
        update_qwen35_gpu_decode_validation(
            Qwen35GpuDecodeValidation {
                engine: step.engine,
                logits: &logits,
                prompt_ids: step.prompt_ids,
                generated_ids: step.generated_ids,
                next_token: step.next_token,
                native_token,
                step: step.step,
                seqlen_offset: step.seqlen_offset,
                ordinal: step.ordinal,
                kv_chunk_size: step.kv_chunk_size,
                prefill_chunk_size: step.cli.prefill_chunk_size,
                use_4b_kernel: step.use_4b_kernel,
            },
            state.gpu_max_delta,
        )?;
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

    Ok(native_token)
}
