//! Runtime-owned production dispatch for one Qwen3.6 token.

use anyhow::{Context, Result};

use crate::qwen36_moe::decode::{
    run_chained_decode_fast, run_chained_decode_fast_with_cache_pos,
    run_chained_decode_fast_with_expert_prefetch,
    run_chained_decode_fast_with_expert_prefetch_and_cache_pos, Qwen36ExecutionOptions,
};
use crate::qwen36_moe::layers::validate_sparse_prefetch_policy;
use crate::qwen36_moe::layers::LoadedQwen36Layers;
use crate::qwen36_moe::persistent_decode::LmHeadFold;
use crate::qwen36_moe::residency::MoeExpertResidencyManager;
use crate::qwen36_moe::types::{
    AttnLayerBuffers, DecodeOutputs, ExpertPrefetchPhase, ExpertRoute, LayerBuffers,
    MultiLayerGeom, PositionPair,
};

pub type ChainExpertPrefetchCallback<'a> = dyn FnMut(&mut MoeExpertResidencyManager, ExpertPrefetchPhase, usize, &[ExpertRoute]) -> Result<()>
    + 'a;

pub struct Qwen36ChainStep<'a> {
    pub ordinal: usize,
    pub geom: &'a MultiLayerGeom,
    pub loaded_layers: &'a mut LoadedQwen36Layers,
    pub initial_hidden: &'a [u8],
    pub position: PositionPair,
    pub step: usize,
    pub accurate_stage_timings: bool,
    pub execution: &'a Qwen36ExecutionOptions,
    pub fold: Option<LmHeadFold<'a>>,
    pub download_final_hidden: bool,
    pub expert_prefetch: Option<&'a mut ChainExpertPrefetchCallback<'a>>,
}

pub struct Qwen36ChainStepOutput {
    pub outputs: DecodeOutputs,
    pub lm_head_folded: bool,
    pub lm_head_folded_top1: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainPath {
    Persistent,
    PersistentSparse,
    ChainedDense,
    ChainedDensePrefetch,
    ChainedSplit,
    ChainedSplitPrefetch,
}

fn select_chain_path(persistent: bool, position: PositionPair, sparse_owner: bool) -> ChainPath {
    match (persistent, position.is_dense(), sparse_owner) {
        (true, _, false) => ChainPath::Persistent,
        (true, _, true) => ChainPath::PersistentSparse,
        (false, true, false) => ChainPath::ChainedDense,
        (false, true, true) => ChainPath::ChainedDensePrefetch,
        (false, false, false) => ChainPath::ChainedSplit,
        (false, false, true) => ChainPath::ChainedSplitPrefetch,
    }
}

fn fold_flags(folded: bool, top1: bool) -> (bool, bool) {
    (folded, folded && top1)
}

fn validate_chain_position(position: PositionPair, layers: &[LayerBuffers]) -> Result<()> {
    if position.rope < 0 || position.cache < 0 {
        anyhow::bail!(
            "Qwen3.6 chain positions must be non-negative: rope={} cache={}",
            position.rope,
            position.cache
        );
    }
    for (layer_idx, layer) in layers.iter().enumerate() {
        if let AttnLayerBuffers::Full {
            kv_cache: Some(cache),
            ..
        } = &layer.attn
        {
            if position.cache >= cache.kv_max_t {
                anyhow::bail!(
                    "Qwen3.6 chain KV capacity exceeded at layer {layer_idx}: \
                     cache slot {}, capacity is {}",
                    position.cache,
                    cache.kv_max_t
                );
            }
        }
    }
    Ok(())
}

pub fn run_chain_step(mut args: Qwen36ChainStep<'_>) -> Result<Qwen36ChainStepOutput> {
    if args.geom.num_layers < 0 || args.loaded_layers.len() != args.geom.num_layers as usize {
        anyhow::bail!(
            "Qwen3.6 chain layer count mismatch: owner has {}, geometry declares {}",
            args.loaded_layers.len(),
            args.geom.num_layers
        );
    }
    validate_chain_position(args.position, args.loaded_layers.layers())?;
    let sparse_owner = args.loaded_layers.has_sparse_expert_residency();
    validate_sparse_prefetch_policy(sparse_owner, args.expert_prefetch.is_some())?;
    let rope = args.position.rope;
    let cache = args.position.cache;
    let path = select_chain_path(
        args.loaded_layers.persistent_enabled(),
        args.position,
        sparse_owner,
    );
    let (lm_head_folded, lm_head_folded_top1) = fold_flags(
        matches!(path, ChainPath::Persistent) && args.fold.is_some(),
        args.fold
            .as_ref()
            .is_some_and(|fold| fold.top1_out.is_some()),
    );
    let (layers, persistent_scratch, moe_expert_residency) = args.loaded_layers.execution_parts();
    let outputs = match path {
        ChainPath::PersistentSparse => {
            drop(args.fold);
            let manager = moe_expert_residency.ok_or_else(|| {
                anyhow::anyhow!("Qwen3.6 expert prefetch callback requires sparse expert residency")
            })?;
            let scratch = persistent_scratch
                .ok_or_else(|| anyhow::anyhow!("Qwen3.6 persistent decode is not enabled"))?;
            let prefetch = args.expert_prefetch.as_deref_mut().ok_or_else(|| {
                anyhow::anyhow!("Qwen3.6 sparse persistent path requires a prefetch callback")
            })?;
            let mut runtime_prefetch = |phase, layer_idx, routes: &[ExpertRoute]| {
                prefetch(manager, phase, layer_idx, routes)
            };
            scratch
                .run_sparse_with_expert_prefetch(
                    args.ordinal,
                    args.initial_hidden,
                    rope,
                    cache,
                    &mut runtime_prefetch,
                )
                .with_context(|| {
                    format!(
                        "segmented persistent sparse decode (step {}, rope {}, cache {})",
                        args.step, rope, cache
                    )
                })?
        }
        ChainPath::Persistent => {
            let scratch = persistent_scratch
                .ok_or_else(|| anyhow::anyhow!("Qwen3.6 persistent decode is not enabled"))?;
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
        chained_path => {
            debug_assert!(matches!(
                chained_path,
                ChainPath::ChainedDense
                    | ChainPath::ChainedDensePrefetch
                    | ChainPath::ChainedSplit
                    | ChainPath::ChainedSplitPrefetch
            ));
            debug_assert!(persistent_scratch.is_none());
            drop(args.fold);
            let mut runtime_prefetch = if let Some(prefetch) = args.expert_prefetch.as_deref_mut() {
                let manager = moe_expert_residency.ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen3.6 expert prefetch callback requires sparse expert residency"
                    )
                })?;
                Some(move |phase, layer_idx, routes: &[ExpertRoute]| {
                    prefetch(manager, phase, layer_idx, routes)
                })
            } else {
                None
            };
            match (args.position.is_dense(), runtime_prefetch.as_mut()) {
                (false, Some(prefetch)) => {
                    run_chained_decode_fast_with_expert_prefetch_and_cache_pos(
                        args.ordinal,
                        args.geom,
                        layers,
                        args.initial_hidden,
                        rope,
                        cache,
                        args.accurate_stage_timings,
                        args.execution,
                        prefetch,
                    )
                    .with_context(|| {
                        format!(
                            "chained sparse-prefill decode (step {}, rope {}, cache {})",
                            args.step, rope, cache
                        )
                    })?
                }
                (false, None) => run_chained_decode_fast_with_cache_pos(
                    args.ordinal,
                    args.geom,
                    layers,
                    args.initial_hidden,
                    rope,
                    cache,
                    args.accurate_stage_timings,
                    args.execution,
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
                    layers,
                    args.initial_hidden,
                    rope,
                    args.accurate_stage_timings,
                    args.execution,
                    prefetch,
                )
                .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?,
                (true, None) => run_chained_decode_fast(
                    args.ordinal,
                    args.geom,
                    layers,
                    args.initial_hidden,
                    rope,
                    args.accurate_stage_timings,
                    args.execution,
                )
                .with_context(|| format!("chained decode (step {}, rope {})", args.step, rope))?,
            }
        }
    };

    Ok(Qwen36ChainStepOutput {
        outputs,
        lm_head_folded,
        lm_head_folded_top1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen36_moe::layer_loader::Qwen36WeightMode;
    use crate::qwen36_moe::residency::{MoeExpertResidencyConfig, MoeExpertResidencyManager};

    #[test]
    fn chain_plan_preserves_dense_split_prefetch_and_fold_semantics() {
        assert_eq!(
            select_chain_path(true, PositionPair::dense(3), false),
            ChainPath::Persistent
        );
        assert_eq!(
            select_chain_path(true, PositionPair::split(7, 3), true),
            ChainPath::PersistentSparse
        );
        assert_eq!(
            select_chain_path(false, PositionPair::dense(3), false),
            ChainPath::ChainedDense
        );
        assert_eq!(
            select_chain_path(false, PositionPair::split(7, 3), true),
            ChainPath::ChainedSplitPrefetch
        );
        assert_eq!(fold_flags(false, false), (false, false));
        assert_eq!(fold_flags(true, false), (true, false));
        assert_eq!(fold_flags(true, true), (true, true));
    }

    #[test]
    fn chain_positions_reject_negative_values_before_dispatch() {
        for position in [PositionPair::split(-1, 0), PositionPair::split(0, -1)] {
            let err = validate_chain_position(position, &[])
                .expect_err("negative chain position must fail");
            assert!(err.to_string().contains("non-negative"));
        }
    }

    #[test]
    fn chain_path_uses_sparse_owner_state_after_policy_validation() {
        assert_eq!(
            select_chain_path(true, PositionPair::dense(3), true),
            ChainPath::PersistentSparse
        );
        assert_eq!(
            select_chain_path(false, PositionPair::dense(3), true),
            ChainPath::ChainedDensePrefetch
        );
        assert_eq!(
            select_chain_path(false, PositionPair::split(7, 3), true),
            ChainPath::ChainedSplitPrefetch
        );
    }

    #[test]
    fn nonpersistent_sparse_owner_rejects_missing_prefetch_before_dispatch() {
        let manager = MoeExpertResidencyManager::new(
            0,
            MoeExpertResidencyConfig::new(1).expect("test residency config"),
        );
        let mut loaded = LoadedQwen36Layers::with_backing(
            Vec::new(),
            Qwen36WeightMode::Int4,
            None,
            Some(manager),
        );
        let geom = MultiLayerGeom {
            hidden: 1,
            vocab: 1,
            num_layers: 0,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            rotary_dim: 0,
            rope_theta: 10_000.0,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 1,
            head_v_dim: 1,
            conv_kernel_dim: 2,
            num_experts: 1,
            moe_intermediate: 1,
            shared_intermediate: 1,
            top_k: 1,
        };

        let err = match run_chain_step(Qwen36ChainStep {
            ordinal: 0,
            geom: &geom,
            loaded_layers: &mut loaded,
            initial_hidden: &[0, 0],
            position: PositionPair::dense(0),
            step: 0,
            accurate_stage_timings: false,
            fold: None,
            download_final_hidden: true,
            expert_prefetch: None,
            execution: &Qwen36ExecutionOptions::default(),
        }) {
            Ok(_) => panic!("sparse owner without prefetch must fail"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("requires an expert prefetch policy"));
    }
}
