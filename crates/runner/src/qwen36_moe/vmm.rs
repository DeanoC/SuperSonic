use anyhow::{Context, Result};
use gpu_hal::Backend;
use model_store::{BakedStore, VirtualArenaTransferBackend};
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_cli::layers::Qwen36WeightMode;
use crate::qwen36_moe_cli::vmm_config::{should_try_moe_expert_vmm, MoeExpertVmmMode};
use crate::qwen36_moe_telemetry::{MoeIslandPrefetchMode, VirtualKvStats};
use crate::qwen36_moe_types::{AttnLayerBuffers, LayerBuffers, MultiLayerGeom};
use supersonic_runtime::qwen36_moe::layer_loader::{
    load_qwen36_layers, Qwen36LayerLoadStrategy, SparseExpertLoadOptions,
};
use supersonic_runtime::qwen36_moe::layers::LoadedQwen36Layers;

const MIB: f64 = (1024 * 1024) as f64;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Qwen36LayerLoadTimings {
    pub(crate) detail_available: bool,
    pub(crate) hal_available: bool,
    pub(crate) buffers: std::time::Duration,
    pub(crate) vmm_setup: std::time::Duration,
    pub(crate) prewarm: std::time::Duration,
    pub(crate) hal_total: std::time::Duration,
    pub(crate) hal_alloc: std::time::Duration,
    pub(crate) hal_copy_h_to_d: std::time::Duration,
    pub(crate) hal_memset: std::time::Duration,
    pub(crate) hal_vmm: std::time::Duration,
    pub(crate) hal_alloc_bytes: u64,
    pub(crate) hal_copy_h_to_d_bytes: u64,
    pub(crate) hal_memset_bytes: u64,
    pub(crate) hal_vmm_bytes: u64,
}

pub(crate) struct Qwen36LayerLoadResult {
    pub(crate) loaded: LoadedQwen36Layers,
    pub(crate) timings: Qwen36LayerLoadTimings,
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
    moe_fixed_hot_experts: Option<usize>,
    moe_prefetch_mode: MoeIslandPrefetchMode,
    moe_prefetch_ranks: usize,
    moe_transition_min_observations: u32,
    moe_async_prefetch: bool,
    moe_async_staging_pages: usize,
    moe_prefetch_evict: bool,
    moe_prefetch_evict_min_probability: f64,
    moe_virtual_transfer_backend: VirtualArenaTransferBackend,
    persistent_decode: bool,
) -> Result<Qwen36LayerLoadResult> {
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
        let buffers_start = std::time::Instant::now();
        let loaded = load_qwen36_layers(
            store,
            ordinal,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            Qwen36LayerLoadStrategy::SparseExperts(SparseExpertLoadOptions {
                cap_experts,
                protected_experts: moe_island_protected_experts,
                fixed_hot_experts: moe_fixed_hot_experts,
                async_prefetch: moe_async_prefetch,
                async_staging_pages: moe_async_staging_pages,
                prefetch_evict: moe_prefetch_evict,
                transfer_backend: moe_virtual_transfer_backend,
            }),
            &crate::qwen36_moe_cli::options::load_options_from_environment(),
        )
        .context("reserve Qwen3.6-MoE routed experts for sparse VMM residency")?;
        let buffers_elapsed = buffers_start.elapsed();
        let vmm_setup_start = std::time::Instant::now();
        let storage_direct =
            moe_virtual_transfer_backend == VirtualArenaTransferBackend::GpuDirectStorage;
        let manager = loaded
            .sparse_expert_residency()
            .expect("sparse load strategy must retain its residency manager");
        let arena_stats = manager.arena().stats();
        let residency_stats = manager.stats();
        println!(
            "  [vmm] Qwen3.6-MoE sparse routed expert residency active on backend={} device {ordinal}: \
             tensors={} max_pages={} protected_pages={} fixed_hot_pages={} logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB",
            backend,
            residency_stats.registered_tensors,
            manager.max_resident_pages(),
            residency_stats.protected_page_budget,
            residency_stats.fixed_hot_page_budget,
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
        if moe_async_prefetch && !storage_direct {
            println!(
                "  [vmm] sparse MoE async page-in active (staging_pages={moe_async_staging_pages})"
            );
        }
        if storage_direct {
            println!("  [vmm] sparse MoE virtual page-in transfer backend: gpu-direct-storage");
            if moe_async_prefetch {
                println!(
                    "  [vmm] sparse MoE async page-in disabled for gpu-direct-storage backend"
                );
            }
        }
        if moe_prefetch_evict {
            println!(
                "  [vmm] sparse MoE prefetch eviction active \
                 (transition_min_probability={moe_prefetch_evict_min_probability:.2})"
            );
        }
        let vmm_setup_elapsed = vmm_setup_start.elapsed();
        return Ok(Qwen36LayerLoadResult {
            loaded,
            timings: Qwen36LayerLoadTimings {
                detail_available: true,
                buffers: buffers_elapsed,
                vmm_setup: vmm_setup_elapsed,
                ..Default::default()
            },
        });
    }

    if should_try_moe_expert_vmm(
        moe_vmm_mode,
        backend,
        weight_mode.is_int4(),
        &format!("{weight_mode:?}"),
        ordinal,
    )? {
        let buffers_start = std::time::Instant::now();
        match load_qwen36_layers(
            store,
            ordinal,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            Qwen36LayerLoadStrategy::VirtualExperts {
                transfer_backend: moe_virtual_transfer_backend,
            },
            &crate::qwen36_moe_cli::options::load_options_from_environment(),
        ) {
            Ok(loaded) => {
                let buffers_elapsed = buffers_start.elapsed();
                let vmm_setup_start = std::time::Instant::now();
                let stats = loaded
                    .virtual_expert_arena()
                    .expect("virtual load strategy must retain its arena")
                    .stats();
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
                let vmm_setup_elapsed = vmm_setup_start.elapsed();
                return Ok(Qwen36LayerLoadResult {
                    loaded,
                    timings: Qwen36LayerLoadTimings {
                        detail_available: true,
                        buffers: buffers_elapsed,
                        vmm_setup: vmm_setup_elapsed,
                        ..Default::default()
                    },
                });
            }
            Err(err) if moe_vmm_mode == MoeExpertVmmMode::Auto => {
                eprintln!(
                    "[vmm] Qwen3.6-MoE routed expert VMM load failed ({err:#}); falling back to dense expert buffers"
                );
            }
            Err(err) => return Err(err.context("load Qwen3.6-MoE routed experts into VMM")),
        }
    }

    let buffers_start = std::time::Instant::now();
    let loaded = load_qwen36_layers(
        store,
        ordinal,
        geom,
        text_config,
        weight_prefix,
        weight_mode,
        kv_max_t,
        kv_fp8,
        kv_vmm,
        Qwen36LayerLoadStrategy::Dense,
        &crate::qwen36_moe_cli::options::load_options_from_environment(),
    )?;
    let buffers_elapsed = buffers_start.elapsed();
    Ok(Qwen36LayerLoadResult {
        loaded,
        timings: Qwen36LayerLoadTimings {
            detail_available: true,
            buffers: buffers_elapsed,
            ..Default::default()
        },
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
