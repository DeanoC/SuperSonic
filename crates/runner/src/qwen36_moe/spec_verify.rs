use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use model_store::BakedStore;

use crate::qwen36_moe_cli::decode_loop::Qwen36DecodeLoopState;
use crate::qwen36_moe_cli::host::lookup_embed_row;
use crate::qwen36_moe_cli::lm_head::{launch_lm_head_from_final_hidden_bytes, LmHeadBuffers};
use crate::qwen36_moe_cli::timing::Qwen36StageTimingTotals;
use crate::qwen36_moe_decode::{
    run_chained_decode_fast, run_chained_decode_fast_with_cache_pos, Qwen36ExecutionOptions,
};
use crate::qwen36_moe_logits::argmax_bf16_logits;
use crate::qwen36_moe_mtp::{MtpChainScratch, MtpForwardScratch};
use crate::qwen36_moe_persistent_decode::PersistentScratch;
use crate::qwen36_moe_speculative::{
    run_speculative_decode_step, run_speculative_decode_step_batched, SpeculativeStepResult,
};
use crate::qwen36_moe_state::{
    refresh_linear_attn_state, restore_linear_attn_state, LinearAttnSnapshot,
};
use crate::qwen36_moe_types::{
    DecodeOutputs, LayerBuffers, MtpLayerBuffers, MultiLayerGeom, PositionPair,
};

pub(crate) struct Qwen36SpecChainStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    /// `(rope, cache)` for this verify-replay step. In dense MTP
    /// the two agree; in SpecPrefill+MTP the rope is on the
    /// absolute prompt timeline while cache is the compact slot.
    pub(crate) position: PositionPair,
    pub(crate) input: u32,
    pub(crate) emit_stage_timings: bool,
}

pub(crate) fn run_spec_chain_step(args: Qwen36SpecChainStep<'_>) -> Result<DecodeOutputs> {
    let t_embed_start = std::time::Instant::now();
    let initial_hidden = lookup_embed_row(
        args.store,
        args.weight_prefix,
        args.input as usize,
        args.geom.hidden as usize,
    )
    .with_context(|| {
        format!(
            "spec verify embed lookup token {} at rope {} cache {}",
            args.input, args.position.rope, args.position.cache
        )
    })?;
    args.stage_timings.record_embed(t_embed_start.elapsed());

    let rope = args.position.rope;
    let cache = args.position.cache;
    let t_chain_start = std::time::Instant::now();
    let outputs = if let Some(scratch) = args.persistent_scratch {
        // The persistent kernel takes (rope, cache) directly. In dense
        // MTP rope == cache and the kernel produces the same output
        // as the pre-PR-#211 inherit-from-position path; in
        // SpecPrefill+MTP the verify replay writes accepted draft
        // tokens at the compact slot while RoPE rotates absolute.
        scratch.run(args.ordinal, &initial_hidden, rope, cache, None, true)?
    } else if !args.position.is_dense() {
        run_chained_decode_fast_with_cache_pos(
            args.ordinal,
            args.geom,
            args.layers,
            &initial_hidden,
            rope,
            cache,
            args.emit_stage_timings,
            args.execution,
        )?
    } else {
        run_chained_decode_fast(
            args.ordinal,
            args.geom,
            args.layers,
            &initial_hidden,
            rope,
            args.emit_stage_timings,
            args.execution,
        )?
    };
    args.stage_timings
        .record_chain(t_chain_start.elapsed(), &outputs);
    args.stage_timings.count_generation_step();
    Ok(outputs)
}

pub(crate) struct Qwen36SpecReplayAccepted<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) snapshot: &'a LinearAttnSnapshot,
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) replay_inputs: &'a [(PositionPair, u32)],
    pub(crate) emit_stage_timings: bool,
}

pub(crate) fn restore_and_replay_accepted_prefix(
    mut args: Qwen36SpecReplayAccepted<'_>,
) -> Result<()> {
    restore_linear_attn_state(args.ordinal, args.layers, args.snapshot)
        .context("restore linear-attn state after partial-accept")?;
    for &(position, input) in args.replay_inputs {
        run_spec_chain_step(Qwen36SpecChainStep {
            ordinal: args.ordinal,
            geom: args.geom,
            store: args.store,
            weight_prefix: args.weight_prefix,
            layers: args.layers,
            execution: args.execution,
            persistent_scratch: args.persistent_scratch.as_deref_mut(),
            stage_timings: args.stage_timings,
            position,
            input,
            emit_stage_timings: args.emit_stage_timings,
        })?;
    }
    Ok(())
}

pub(crate) struct Qwen36BatchedSpecVerifyInputs<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) final_norm_w: &'a GpuBuffer,
    pub(crate) lm_head_w: &'a GpuBuffer,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) inputs: &'a [(PositionPair, u32)],
    pub(crate) emit_stage_timings: bool,
}

pub(crate) fn run_batched_spec_verify_inputs(
    mut args: Qwen36BatchedSpecVerifyInputs<'_>,
) -> Result<Vec<(u32, Vec<u8>)>> {
    if args.inputs.is_empty() {
        return Ok(Vec::new());
    }

    let mut final_hiddens = Vec::with_capacity(args.inputs.len());
    for &(position, input) in args.inputs {
        let chain_outputs = run_spec_chain_step(Qwen36SpecChainStep {
            ordinal: args.ordinal,
            geom: args.geom,
            store: args.store,
            weight_prefix: args.weight_prefix,
            layers: args.layers,
            execution: args.execution,
            persistent_scratch: args.persistent_scratch.as_deref_mut(),
            stage_timings: args.stage_timings,
            position,
            input,
            emit_stage_timings: args.emit_stage_timings,
        })?;
        final_hiddens.push(chain_outputs.final_hidden_bytes);
    }

    run_batched_lm_head_top1(Qwen36BatchedLmHeadTop1 {
        ordinal: args.ordinal,
        geom: args.geom,
        final_norm_w: args.final_norm_w,
        lm_head_w: args.lm_head_w,
        execution: args.execution,
        stage_timings: args.stage_timings,
        final_hiddens,
    })
}

pub(crate) struct Qwen36SingleLmHeadTop1<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) final_norm_w: &'a GpuBuffer,
    pub(crate) lm_head_w: &'a GpuBuffer,
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) final_hidden: &'a mut GpuBuffer,
    pub(crate) logits: &'a mut GpuBuffer,
    pub(crate) counter: &'a mut GpuBuffer,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) final_hidden_bytes: &'a [u8],
}

pub(crate) fn run_single_lm_head_top1(args: Qwen36SingleLmHeadTop1<'_>) -> Result<u32> {
    let t_lm_head_start = std::time::Instant::now();
    let logits_bytes = launch_lm_head_from_final_hidden_bytes(
        args.ordinal,
        args.geom,
        args.final_hidden_bytes,
        &args.execution.prefill_kernel,
        LmHeadBuffers {
            final_norm_w: args.final_norm_w,
            lm_head_w: args.lm_head_w,
            final_hidden: args.final_hidden,
            logits: args.logits,
            counter: args.counter,
        },
    )
    .context("spec verify GPU lm_head")?;
    args.stage_timings.record_lm_head(t_lm_head_start.elapsed());
    Ok(argmax_bf16_logits(&logits_bytes))
}

pub(crate) struct Qwen36SequentialSpecVerifyInput<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) final_norm_w: &'a GpuBuffer,
    pub(crate) lm_head_w: &'a GpuBuffer,
    pub(crate) final_hidden: &'a mut GpuBuffer,
    pub(crate) logits: &'a mut GpuBuffer,
    pub(crate) counter: &'a mut GpuBuffer,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) position: PositionPair,
    pub(crate) input: u32,
    pub(crate) emit_stage_timings: bool,
}

pub(crate) fn run_sequential_spec_verify_input(
    args: Qwen36SequentialSpecVerifyInput<'_>,
) -> Result<(u32, Vec<u8>)> {
    let outputs = run_spec_chain_step(Qwen36SpecChainStep {
        ordinal: args.ordinal,
        geom: args.geom,
        store: args.store,
        weight_prefix: args.weight_prefix,
        layers: args.layers,
        execution: args.execution,
        persistent_scratch: args.persistent_scratch,
        stage_timings: args.stage_timings,
        position: args.position,
        input: args.input,
        emit_stage_timings: args.emit_stage_timings,
    })?;

    let top1 = run_single_lm_head_top1(Qwen36SingleLmHeadTop1 {
        ordinal: args.ordinal,
        geom: args.geom,
        final_norm_w: args.final_norm_w,
        lm_head_w: args.lm_head_w,
        execution: args.execution,
        final_hidden: args.final_hidden,
        logits: args.logits,
        counter: args.counter,
        stage_timings: args.stage_timings,
        final_hidden_bytes: &outputs.final_hidden_bytes,
    })?;
    Ok((top1, outputs.final_hidden_bytes))
}

pub(crate) struct Qwen36SpeculativeExtension<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) mtp: &'a mut MtpLayerBuffers,
    pub(crate) forward_scratch: &'a mut MtpForwardScratch,
    pub(crate) chain_scratch: &'a mut MtpChainScratch,
    pub(crate) embed_w: &'a GpuBuffer,
    pub(crate) final_norm_w: &'a GpuBuffer,
    pub(crate) lm_head_w: &'a GpuBuffer,
    pub(crate) final_hidden: &'a mut GpuBuffer,
    pub(crate) logits: &'a mut GpuBuffer,
    pub(crate) counter: &'a mut GpuBuffer,
    pub(crate) linear_attn_snapshot: Option<&'a mut LinearAttnSnapshot>,
    pub(crate) loop_state: &'a Qwen36DecodeLoopState,
    /// `(rope, cache)` of the just-sampled token that the
    /// speculative pass starts from. The speculative driver uses
    /// `rope + k` for RoPE rotation of draft step k, and
    /// `cache + k` for the KV slot. In dense MTP they agree; in
    /// SpecPrefill+MTP rope is the absolute prompt timeline while
    /// cache is the compact slot.
    pub(crate) base_position: PositionPair,
    pub(crate) h_base_in: &'a [u8],
    pub(crate) first_token: u32,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) emit_stage_timings: bool,
    pub(crate) max_drafts: usize,
}

pub(crate) fn run_speculative_extension(
    mut args: Qwen36SpeculativeExtension<'_>,
) -> Result<SpeculativeStepResult> {
    let dynamic_k = args.loop_state.speculative_draft_count(args.max_drafts);

    if let Some(snapshot) = args.linear_attn_snapshot.as_deref_mut() {
        refresh_linear_attn_state(args.ordinal, args.layers, snapshot)
            .context("refresh linear-attn snapshot before batched verify")?;

        let mut result = run_speculative_decode_step_batched(
            args.ordinal,
            args.geom,
            args.mtp,
            args.forward_scratch,
            args.chain_scratch,
            args.embed_w,
            args.lm_head_w,
            args.h_base_in,
            args.first_token,
            args.base_position.rope,
            args.base_position.cache,
            dynamic_k,
            |inputs| -> Result<Vec<(u32, Vec<u8>)>> {
                run_batched_spec_verify_inputs(Qwen36BatchedSpecVerifyInputs {
                    ordinal: args.ordinal,
                    geom: args.geom,
                    store: args.store,
                    weight_prefix: args.weight_prefix,
                    layers: args.layers,
                    execution: args.execution,
                    persistent_scratch: args.persistent_scratch.as_deref_mut(),
                    final_norm_w: args.final_norm_w,
                    lm_head_w: args.lm_head_w,
                    stage_timings: args.stage_timings,
                    inputs,
                    emit_stage_timings: args.emit_stage_timings,
                })
            },
        )
        .context("batched speculative decode step")?;

        if let Some(replay) = args.loop_state.partial_accept_replay_inputs(
            args.first_token,
            args.base_position,
            &result,
            dynamic_k,
        ) {
            result.replay_steps = replay.len();
            restore_and_replay_accepted_prefix(Qwen36SpecReplayAccepted {
                ordinal: args.ordinal,
                geom: args.geom,
                store: args.store,
                weight_prefix: args.weight_prefix,
                layers: args.layers,
                execution: args.execution,
                snapshot,
                persistent_scratch: args.persistent_scratch.as_deref_mut(),
                stage_timings: args.stage_timings,
                replay_inputs: &replay,
                emit_stage_timings: args.emit_stage_timings,
            })?;
        }
        Ok(result)
    } else {
        run_speculative_decode_step(
            args.ordinal,
            args.geom,
            args.mtp,
            args.forward_scratch,
            args.chain_scratch,
            args.embed_w,
            args.lm_head_w,
            args.h_base_in,
            args.first_token,
            args.base_position.rope,
            args.base_position.cache,
            dynamic_k,
            |position, input| -> Result<(u32, Vec<u8>)> {
                run_sequential_spec_verify_input(Qwen36SequentialSpecVerifyInput {
                    ordinal: args.ordinal,
                    geom: args.geom,
                    store: args.store,
                    weight_prefix: args.weight_prefix,
                    layers: args.layers,
                    execution: args.execution,
                    persistent_scratch: args.persistent_scratch.as_deref_mut(),
                    final_norm_w: args.final_norm_w,
                    lm_head_w: args.lm_head_w,
                    final_hidden: args.final_hidden,
                    logits: args.logits,
                    counter: args.counter,
                    stage_timings: args.stage_timings,
                    position,
                    input,
                    emit_stage_timings: args.emit_stage_timings,
                })
            },
        )
        .context("speculative decode step")
    }
}

pub(crate) struct Qwen36BatchedLmHeadTop1<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) final_norm_w: &'a GpuBuffer,
    pub(crate) lm_head_w: &'a GpuBuffer,
    pub(crate) execution: &'a Qwen36ExecutionOptions,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) final_hiddens: Vec<Vec<u8>>,
}

pub(crate) fn run_batched_lm_head_top1(
    args: Qwen36BatchedLmHeadTop1<'_>,
) -> Result<Vec<(u32, Vec<u8>)>> {
    let n = args.final_hiddens.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let hidden = args.geom.hidden as usize;
    let t_lm_head_start = std::time::Instant::now();
    let mut concat = Vec::with_capacity(n * hidden * 2);
    for fh in &args.final_hiddens {
        concat.extend_from_slice(fh);
    }
    let fh_buf = GpuBuffer::from_host_bytes(args.ordinal, ScalarType::BF16, &[n, hidden], &concat)?;
    let mut logits_buf = GpuBuffer::zeros(
        args.ordinal,
        ScalarType::BF16,
        &[n, args.geom.vocab as usize],
    )?;
    kernel_ffi::qwen36_moe::lm_head_batched_launch_with_options(
        args.ordinal,
        n as i32,
        args.geom.hidden,
        args.geom.vocab,
        args.geom.rms_norm_eps,
        &fh_buf,
        args.final_norm_w,
        args.lm_head_w,
        &mut logits_buf,
        None,
        &args.execution.prefill_kernel,
    )?;
    let logits_bytes = logits_buf.to_host_bytes().context("d2h batched logits")?;
    args.stage_timings.record_lm_head(t_lm_head_start.elapsed());

    let row_bytes = args.geom.vocab as usize * 2;
    let mut results = Vec::with_capacity(n);
    for (i, fh) in args.final_hiddens.into_iter().enumerate() {
        let row = &logits_bytes[i * row_bytes..(i + 1) * row_bytes];
        results.push((argmax_bf16_logits(row), fh));
    }
    Ok(results)
}
