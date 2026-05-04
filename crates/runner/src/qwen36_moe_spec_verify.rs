use anyhow::{Context, Result};
use model_store::BakedStore;

use crate::qwen36_moe_decode::{
    run_chained_decode_fast, DecodeOutputs, LayerBuffers, MultiLayerGeom,
};
use crate::qwen36_moe_host::lookup_embed_row;
use crate::qwen36_moe_persistent_decode::PersistentScratch;
use crate::qwen36_moe_timing::Qwen36StageTimingTotals;

pub(crate) struct Qwen36SpecChainStep<'a> {
    pub(crate) ordinal: usize,
    pub(crate) geom: &'a MultiLayerGeom,
    pub(crate) store: &'a BakedStore,
    pub(crate) weight_prefix: &'a str,
    pub(crate) layers: &'a mut [LayerBuffers],
    pub(crate) persistent_scratch: Option<&'a mut PersistentScratch>,
    pub(crate) stage_timings: &'a mut Qwen36StageTimingTotals,
    pub(crate) position: i32,
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
            "spec verify embed lookup token {} at position {}",
            args.input, args.position
        )
    })?;
    args.stage_timings.record_embed(t_embed_start.elapsed());

    let t_chain_start = std::time::Instant::now();
    let outputs = if let Some(scratch) = args.persistent_scratch {
        scratch.run(args.ordinal, &initial_hidden, args.position, None)?
    } else {
        run_chained_decode_fast(
            args.ordinal,
            args.geom,
            args.layers,
            &initial_hidden,
            args.position,
            args.emit_stage_timings,
        )?
    };
    args.stage_timings
        .record_chain(t_chain_start.elapsed(), &outputs);
    args.stage_timings.count_generation_step();
    Ok(outputs)
}
