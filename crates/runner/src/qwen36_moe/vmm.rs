use anyhow::{Context, Result};
use gpu_hal::{Backend, VirtualArena};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_cli::layers::{load_all_layer_buffers, Qwen36WeightMode};
use crate::qwen36_moe_cli::vmm_config::{should_try_moe_expert_vmm, MoeExpertVmmMode};
use crate::qwen36_moe_residency::MoeExpertResidencyManager;
use crate::qwen36_moe_residency_types::MoeExpertResidencyConfig;
use crate::qwen36_moe_telemetry::{MoeIslandPrefetchMode, VirtualKvStats};
use crate::qwen36_moe_types::{AttnLayerBuffers, LayerBuffers, MultiLayerGeom};

const MIB: f64 = (1024 * 1024) as f64;

pub(crate) struct Qwen36DecodeLayers {
    pub(crate) layers: Vec<LayerBuffers>,
    pub(crate) moe_expert_arena: Option<VirtualArena>,
    pub(crate) moe_expert_residency: Option<MoeExpertResidencyManager>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_decode_layers_with_vmm_strategy(
    store: &BakedStore,
    ordinal: usize,
    backend: Backend,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    kv_max_t: usize,
    kv_fp8: bool,
    kv_vmm: bool,
    moe_vmm_mode: MoeExpertVmmMode,
    moe_island_cap_experts: Option<usize>,
    moe_island_protected_experts: Option<usize>,
    moe_prefetch_mode: MoeIslandPrefetchMode,
    moe_prefetch_ranks: usize,
    moe_transition_min_observations: u32,
    moe_async_prefetch: bool,
    moe_async_staging_pages: usize,
    persistent_decode: bool,
) -> Result<Qwen36DecodeLayers> {
    if let Some(cap_experts) = moe_island_cap_experts {
        if cap_experts < geom.top_k as usize {
            anyhow::bail!(
                "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS={cap_experts} is smaller than model top_k={}; \
                 sparse decode needs at least one layer's routed experts resident",
                geom.top_k
            );
        }
        if !should_try_moe_expert_vmm(
            MoeExpertVmmMode::Force,
            backend,
            weight_mode.is_int4(),
            &format!("{weight_mode:?}"),
            ordinal,
        )? {
            unreachable!("forced VMM expert check should either return true or error");
        }
        let config = MoeExpertResidencyConfig::new(1)?;
        let mut manager = MoeExpertResidencyManager::new(ordinal, config);
        let layers = load_all_layer_buffers(
            store,
            ordinal,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            None,
            Some(&mut manager),
        )
        .context("reserve Qwen3.6-MoE routed experts for sparse VMM residency")?;
        let max_resident_pages = manager
            .page_budget_for_routed_experts(cap_experts)
            .context("derive sparse MoE page budget from routed expert tensor layout")?;
        manager
            .set_max_resident_pages(max_resident_pages)
            .context("apply sparse MoE page budget")?;
        if moe_async_prefetch {
            manager
                .enable_async_prefetch(moe_async_staging_pages)
                .context("enable sparse MoE async prefetch")?;
        }
        let max_protected_pages = if let Some(protected_experts) = moe_island_protected_experts {
            let pages = manager
                .page_budget_for_routed_experts(protected_experts)
                .context(
                    "derive sparse MoE protected page budget from routed expert tensor layout",
                )?
                .min(max_resident_pages);
            manager.set_max_protected_pages(pages);
            pages
        } else {
            0
        };
        let arena_stats = manager.arena().stats();
        let residency_stats = manager.stats();
        println!(
            "  [vmm] Qwen3.6-MoE sparse routed expert residency active on backend={} device {ordinal}: \
             tensors={} max_pages={} protected_pages={} logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB",
            backend,
            residency_stats.registered_tensors,
            max_resident_pages,
            max_protected_pages,
            arena_stats.logical_bytes as f64 / MIB,
            arena_stats.resident_bytes as f64 / MIB,
            arena_stats.reserved_bytes as f64 / MIB,
        );
        if persistent_decode {
            println!(
                "  [vmm] sparse MoE residency will use segmented persistent decode \
                 (router prefetch + FFN resume per layer)"
            );
        }
        if moe_prefetch_mode.uses_previous_token_routes() {
            println!(
                "  [vmm] sparse MoE {} lookahead prefetch active \
                 (ranks={moe_prefetch_ranks}/{})",
                moe_prefetch_mode.as_str(),
                geom.top_k
            );
            if moe_prefetch_mode.transition_weighted() {
                println!(
                    "  [vmm] sparse MoE transition predictor warmup \
                     min_observations={moe_transition_min_observations}"
                );
            }
        }
        if moe_async_prefetch {
            println!(
                "  [vmm] sparse MoE async page-in active (staging_pages={moe_async_staging_pages})"
            );
        }
        return Ok(Qwen36DecodeLayers {
            layers,
            moe_expert_arena: None,
            moe_expert_residency: Some(manager),
        });
    }

    if should_try_moe_expert_vmm(
        moe_vmm_mode,
        backend,
        weight_mode.is_int4(),
        &format!("{weight_mode:?}"),
        ordinal,
    )? {
        let mut arena = BakedStore::virtual_weight_arena(ordinal);
        match load_all_layer_buffers(
            store,
            ordinal,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            Some(&mut arena),
            None,
        ) {
            Ok(layers) => {
                let stats = arena.stats();
                println!(
                    "  [vmm] Qwen3.6-MoE routed expert slabs active on backend={} device {ordinal}: \
                     allocations={} mappings={} logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB",
                    backend,
                    stats.allocations,
                    stats.mapping_count,
                    stats.logical_bytes as f64 / MIB,
                    stats.resident_bytes as f64 / MIB,
                    stats.reserved_bytes as f64 / MIB,
                );
                return Ok(Qwen36DecodeLayers {
                    layers,
                    moe_expert_arena: Some(arena),
                    moe_expert_residency: None,
                });
            }
            Err(err) if moe_vmm_mode == MoeExpertVmmMode::Auto => {
                eprintln!(
                    "[vmm] Qwen3.6-MoE routed expert VMM load failed ({err:#}); falling back to dense expert buffers"
                );
                drop(arena);
            }
            Err(err) => return Err(err.context("load Qwen3.6-MoE routed experts into VMM")),
        }
    }

    let layers = load_all_layer_buffers(
        store,
        ordinal,
        geom,
        text_config,
        weight_prefix,
        weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        None,
        None,
    )?;
    Ok(Qwen36DecodeLayers {
        layers,
        moe_expert_arena: None,
        moe_expert_residency: None,
    })
}

pub(crate) fn virtual_kv_stats_for_layers(layers: &[LayerBuffers]) -> VirtualKvStats {
    let mut out = VirtualKvStats::default();
    for layer in layers {
        let AttnLayerBuffers::Full {
            kv_cache: Some(cache),
            ..
        } = &layer.attn
        else {
            continue;
        };
        let mut layer_has_virtual_kv = false;
        for buffer in [
            cache.virtual_kv_cache_k.as_ref(),
            cache.virtual_kv_cache_v.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let stats = buffer.stats();
            out.logical_bytes += stats.logical_bytes;
            out.reserved_bytes += stats.reserved_bytes;
            out.resident_bytes += stats.resident_bytes;
            out.logical_resident_bytes += stats.logical_resident_bytes;
            out.mappings += stats.mapping_count;
            layer_has_virtual_kv = true;
        }
        if layer_has_virtual_kv {
            out.layers += 1;
        }
    }
    out
}

pub(crate) fn print_virtual_kv_stats_if_active(
    stats: VirtualKvStats,
    kv_fp8: bool,
    backend: Backend,
    ordinal: usize,
) {
    if stats.layers == 0 {
        return;
    }

    println!(
        "  [vmm] Qwen3.6-MoE {} KV active on backend={} device {ordinal}: \
         layers={} mappings={} logical={:.2}MiB logical_resident={:.2}MiB \
         resident={:.2}MiB reserved={:.2}MiB",
        if kv_fp8 { "FP8" } else { "BF16" },
        backend,
        stats.layers,
        stats.mappings,
        stats.logical_bytes as f64 / MIB,
        stats.logical_resident_bytes as f64 / MIB,
        stats.resident_bytes as f64 / MIB,
        stats.reserved_bytes as f64 / MIB,
    );
}
