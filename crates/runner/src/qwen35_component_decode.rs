use anyhow::Result;

use crate::decode_engine::DecodeEngine;
use crate::qwen35_decode_util::{token_history, token_history_with_next};
use crate::qwen35_trace::{
    trace_component_input_layer, trace_component_layer, trace_component_linear_layer,
    trace_component_linear_state_layer,
};
use crate::Cli;

pub(crate) struct Qwen35ComponentDecode<'a> {
    pub(crate) cli: &'a Cli,
    pub(crate) prompt_ids: &'a [u32],
    pub(crate) generated_ids: &'a [u32],
    pub(crate) next_token: u32,
    pub(crate) seqlen_offset: usize,
    pub(crate) ordinal: usize,
    pub(crate) kv_chunk_size: usize,
    pub(crate) use_4b_kernel: bool,
}

pub(crate) fn run_qwen35_component_single_decode(
    engine: &mut DecodeEngine,
    decode: Qwen35ComponentDecode<'_>,
) -> Result<Vec<f32>> {
    if let Some(trace_layer) = decode.cli.trace_component_linear_state_layer {
        let trace_token_ids = token_history(decode.prompt_ids, decode.generated_ids);
        trace_component_linear_state_layer(
            engine,
            trace_layer,
            trace_token_ids.as_slice(),
            decode.ordinal,
            decode.kv_chunk_size,
            decode.cli.prefill_chunk_size,
            decode.use_4b_kernel,
        )?;
    }
    if let Some(trace_layer) = decode.cli.trace_component_input_layer {
        let (logits, hidden_trace) = engine.component_decode_step_4b_traced(
            decode.next_token,
            decode.seqlen_offset,
            trace_layer,
        )?;
        trace_component_input_layer(
            engine,
            &hidden_trace,
            trace_layer,
            token_history_with_next(decode.prompt_ids, decode.generated_ids, decode.next_token)
                .as_slice(),
            decode.ordinal,
            decode.kv_chunk_size,
            decode.cli.prefill_chunk_size,
            decode.use_4b_kernel,
        )?;
        Ok(logits)
    } else if let Some(trace_layer) = decode.cli.trace_component_layer {
        let (logits, layer_trace) = engine.component_decode_step_4b_trace_layer(
            decode.next_token,
            decode.seqlen_offset,
            trace_layer,
        )?;
        trace_component_layer(
            engine,
            trace_layer,
            &layer_trace,
            token_history_with_next(decode.prompt_ids, decode.generated_ids, decode.next_token)
                .as_slice(),
            decode.ordinal,
            decode.kv_chunk_size,
            decode.cli.prefill_chunk_size,
            decode.use_4b_kernel,
        )?;
        Ok(logits)
    } else if let Some(trace_layer) = decode.cli.trace_component_linear_layer {
        let (logits, linear_trace) = engine.component_decode_step_4b_trace_linear_layer(
            decode.next_token,
            decode.seqlen_offset,
            trace_layer,
        )?;
        trace_component_linear_layer(
            engine,
            trace_layer,
            &linear_trace,
            token_history_with_next(decode.prompt_ids, decode.generated_ids, decode.next_token)
                .as_slice(),
            decode.ordinal,
            decode.kv_chunk_size,
            decode.cli.prefill_chunk_size,
            decode.use_4b_kernel,
        )?;
        Ok(logits)
    } else {
        engine.decode_step(decode.next_token, decode.seqlen_offset)
    }
}
