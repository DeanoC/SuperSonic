//! Runtime-owned production dispatch for one Qwen3.6 token.

use anyhow::{Context, Result};

use crate::qwen36_moe::decode::{
    run_chained_decode_fast, run_chained_decode_fast_with_cache_pos,
    run_chained_decode_fast_with_expert_prefetch,
    run_chained_decode_fast_with_expert_prefetch_and_cache_pos,
};
use crate::qwen36_moe::persistent_decode::{LmHeadFold, PersistentScratch};
use crate::qwen36_moe::types::{
    DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers, MultiLayerGeom, PositionPair,
};

pub type ChainExpertPrefetchCallback<'a> =
    dyn FnMut(ExpertPrefetchPhase, usize, &[ExpertRoute]) -> Result<()> + 'a;

pub struct Qwen36ChainStep<'a> {
    pub ordinal: usize,
    pub geom: &'a MultiLayerGeom,
    pub layers: &'a mut [LayerBuffers],
    pub persistent_scratch: Option<&'a mut PersistentScratch>,
    pub initial_hidden: &'a [u8],
    pub position: PositionPair,
    pub step: usize,
    pub accurate_stage_timings: bool,
    pub fold: Option<LmHeadFold<'a>>,
    pub download_final_hidden: bool,
    pub expert_prefetch: Option<&'a mut ChainExpertPrefetchCallback<'a>>,
}

pub struct Qwen36ChainStepOutput {
    pub outputs: DecodeOutputs,
    pub lm_head_folded: bool,
    pub lm_head_folded_top1: bool,
}

pub fn run_chain_step(mut args: Qwen36ChainStep<'_>) -> Result<Qwen36ChainStepOutput> {
    let rope = args.position.rope;
    let cache = args.position.cache;
    let mut lm_head_folded = false;
    let mut lm_head_folded_top1 = false;

    let outputs = if let Some(scratch) = args.persistent_scratch.as_deref_mut() {
        if let Some(prefetch) = args.expert_prefetch.as_deref_mut() {
            drop(args.fold);
            scratch
                .run_sparse_with_expert_prefetch(
                    args.ordinal,
                    args.initial_hidden,
                    rope,
                    cache,
                    prefetch,
                )
                .with_context(|| {
                    format!(
                        "segmented persistent sparse decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        } else {
            lm_head_folded = args.fold.is_some();
            lm_head_folded_top1 = args
                .fold
                .as_ref()
                .is_some_and(|fold| fold.top1_out.is_some());
            scratch
                .run(
                    args.ordinal,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.fold,
                    args.download_final_hidden,
                )
                .with_context(|| {
                    format!(
                        "persistent decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        }
    } else {
        drop(args.fold);
        match (
            args.position.is_dense(),
            args.expert_prefetch.as_deref_mut(),
        ) {
            (false, Some(prefetch)) => run_chained_decode_fast_with_expert_prefetch_and_cache_pos(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                cache,
                args.accurate_stage_timings,
                prefetch,
            )
            .with_context(|| {
                format!(
                    "chained sparse-prefill decode (step {}, rope {}, cache {})",
                    args.step, rope, cache
                )
            })?,
            (false, None) => run_chained_decode_fast_with_cache_pos(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                cache,
                args.accurate_stage_timings,
            )
            .with_context(|| {
                format!(
                    "chained sparse-prefill decode (step {}, rope {}, cache {})",
                    args.step, rope, cache
                )
            })?,
            (true, Some(prefetch)) => run_chained_decode_fast_with_expert_prefetch(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                args.accurate_stage_timings,
                prefetch,
            )
            .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?,
            (true, None) => run_chained_decode_fast(
                args.ordinal,
                args.geom,
                args.layers,
                args.initial_hidden,
                rope,
                args.accurate_stage_timings,
            )
            .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?,
        }
    };

    Ok(Qwen36ChainStepOutput {
        outputs,
        lm_head_folded,
        lm_head_folded_top1,
    })
}
