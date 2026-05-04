use anyhow::Result;

use crate::decode_engine::DecodeEngine;
use crate::qwen35_trace::{
    trace_persistent_full_attn_layer, trace_persistent_input_layer, trace_persistent_linear_layer,
    trace_persistent_linear_state_layer,
};
use crate::Cli;

pub(crate) struct Qwen35PersistentDecodeTrace<'a> {
    pub(crate) cli: &'a Cli,
    pub(crate) trace_token_ids: &'a [u32],
    pub(crate) trace_tokens: &'a [u32],
    pub(crate) seqlen_offset: usize,
    pub(crate) ordinal: usize,
    pub(crate) kv_chunk_size: usize,
    pub(crate) use_4b_kernel: bool,
    pub(crate) batch_mode: bool,
}

pub(crate) fn qwen35_persistent_decode_trace_enabled(cli: &Cli) -> bool {
    cli.trace_persistent_linear_state_layer.is_some()
        || cli.trace_persistent_input_layer.is_some()
        || cli.trace_persistent_full_attn_layer.is_some()
        || cli.trace_persistent_linear_layer.is_some()
}

pub(crate) fn run_qwen35_persistent_decode_traces(
    engine: &mut DecodeEngine,
    trace: Qwen35PersistentDecodeTrace<'_>,
) -> Result<()> {
    if let Some(trace_layer) = trace.cli.trace_persistent_linear_state_layer {
        let _ = engine.decode_step_batch_trace_hidden_after_layers(
            trace.trace_tokens,
            trace.seqlen_offset,
            trace_layer + 1,
            0,
        )?;
        trace_persistent_linear_state_layer(
            engine,
            trace_layer,
            trace.trace_token_ids,
            trace.ordinal,
            trace.kv_chunk_size,
            trace.cli.prefill_chunk_size,
            trace.use_4b_kernel,
        )?;
        engine.rebuild_prefill_state(trace.trace_token_ids, trace.batch_mode)?;
    }
    if let Some(trace_layer) = trace.cli.trace_persistent_input_layer {
        let native_hidden = engine.decode_step_batch_trace_hidden_after_layers(
            trace.trace_tokens,
            trace.seqlen_offset,
            trace_layer,
            0,
        )?;
        trace_persistent_input_layer(
            engine,
            &native_hidden,
            trace_layer,
            trace.trace_token_ids,
            trace.ordinal,
            trace.kv_chunk_size,
            trace.cli.prefill_chunk_size,
            trace.use_4b_kernel,
        )?;
        engine.rebuild_prefill_state(trace.trace_token_ids, trace.batch_mode)?;
    }
    if let Some(trace_layer) = trace.cli.trace_persistent_full_attn_layer {
        trace_persistent_full_attn_layer(
            engine,
            trace_layer,
            trace.trace_token_ids,
            trace.trace_tokens,
            trace.seqlen_offset,
            trace.ordinal,
            trace.kv_chunk_size,
            trace.cli.prefill_chunk_size,
            trace.use_4b_kernel,
        )?;
        engine.rebuild_prefill_state(trace.trace_token_ids, trace.batch_mode)?;
    }
    if let Some(trace_layer) = trace.cli.trace_persistent_linear_layer {
        trace_persistent_linear_layer(
            engine,
            trace_layer,
            trace.trace_token_ids,
            trace.trace_tokens,
            trace.seqlen_offset,
            trace.ordinal,
            trace.kv_chunk_size,
            trace.cli.prefill_chunk_size,
            trace.use_4b_kernel,
        )?;
        engine.rebuild_prefill_state(trace.trace_token_ids, trace.batch_mode)?;
    }

    Ok(())
}
