use anyhow::Result;

use crate::decode_engine::DecodeEngine;

pub(crate) fn report_qwen35_virtual_kv_after_prefill(engine: &mut DecodeEngine) -> Result<()> {
    let virtual_kv_stats = engine.virtual_kv_memory_stats();
    if virtual_kv_stats.layers > 0 {
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        let pct = if virtual_kv_stats.reserved_bytes > 0 {
            100.0 * virtual_kv_stats.resident_bytes as f64 / virtual_kv_stats.reserved_bytes as f64
        } else {
            0.0
        };
        eprintln!(
            "[vmm] virtual KV logical={:.2}MiB resident={:.2}MiB reserved={:.2}MiB ({pct:.1}%) mappings={} layers={}",
            mib(virtual_kv_stats.logical_bytes),
            mib(virtual_kv_stats.resident_bytes),
            mib(virtual_kv_stats.reserved_bytes),
            virtual_kv_stats.mappings,
            virtual_kv_stats.layers
        );
        if std::env::var_os("SUPERSONIC_VMM_KV_STATS").is_some() {
            for (layer_idx, stats) in engine.virtual_kv_memory_stats_by_layer() {
                let layer_pct = if stats.reserved_bytes > 0 {
                    100.0 * stats.resident_bytes as f64 / stats.reserved_bytes as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[vmm] layer={layer_idx} logical={:.2}MiB logical_resident={:.2}MiB backup={:.2}MiB resident={:.2}MiB reserved={:.2}MiB ({layer_pct:.1}%) mappings={}",
                    mib(stats.logical_bytes),
                    mib(stats.logical_resident_bytes),
                    mib(stats.logical_backup_bytes),
                    mib(stats.resident_bytes),
                    mib(stats.reserved_bytes),
                    stats.mappings
                );
            }
        }
    }

    if std::env::var_os("SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL").is_some() {
        let before = engine.virtual_kv_memory_stats();
        if before.layers == 0 {
            eprintln!("[vmm] SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL set but virtual KV is inactive");
        } else {
            verify_qwen35_virtual_kv_eviction(engine)?;
        }
    }
    Ok(())
}

fn verify_qwen35_virtual_kv_eviction(engine: &mut DecodeEngine) -> Result<()> {
    let verify_bytes = std::env::var_os("SUPERSONIC_VMM_KV_VERIFY_EVICT_BYTES").is_some();
    let kv_before = if verify_bytes {
        Some(engine.full_attention_prefix_cache_snapshots_bf16_host()?)
    } else {
        None
    };
    engine.evict_virtual_kv_to_host()?;
    let evicted = engine.virtual_kv_memory_stats();
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[vmm] evicted virtual KV to host logical_backup={:.2}MiB resident={:.2}MiB reserved={:.2}MiB mappings={}",
        mib(evicted.logical_backup_bytes),
        mib(evicted.resident_bytes),
        mib(evicted.reserved_bytes),
        evicted.mappings
    );
    if std::env::var_os("SUPERSONIC_VMM_KV_RESTORE_TO_VMM").is_some() {
        engine.restore_virtual_kv_from_host_to_vmm()?;
    } else {
        engine.restore_virtual_kv_from_host()?;
    }
    if let Some(kv_before) = kv_before {
        let kv_after = engine.full_attention_prefix_cache_snapshots_bf16_host()?;
        if let Some(mismatch) = first_qwen35_virtual_kv_mismatch(&kv_before, &kv_after) {
            eprintln!("[vmm] warning: virtual KV eviction byte-restore mismatch: {mismatch}");
        }
    }
    let restored = engine.virtual_kv_memory_stats();
    eprintln!(
        "[vmm] restored virtual KV from host logical_resident={:.2}MiB resident={:.2}MiB reserved={:.2}MiB mappings={}",
        mib(restored.logical_resident_bytes),
        mib(restored.resident_bytes),
        mib(restored.reserved_bytes),
        restored.mappings
    );
    Ok(())
}

type Qwen35VirtualKvSnapshot = (usize, Vec<u8>, Vec<u8>, usize);

fn first_qwen35_virtual_kv_mismatch(
    before: &[Qwen35VirtualKvSnapshot],
    after: &[Qwen35VirtualKvSnapshot],
) -> Option<String> {
    for (
        (before_layer, before_k, before_v, before_len),
        (after_layer, after_k, after_v, after_len),
    ) in before.iter().zip(after.iter())
    {
        if before_layer != after_layer || before_len != after_len {
            return Some(format!(
                "layer/id mismatch before={before_layer}:{before_len} after={after_layer}:{after_len}"
            ));
        }
        let k_diff = before_k
            .iter()
            .zip(after_k.iter())
            .position(|(a, b)| a != b);
        let v_diff = before_v
            .iter()
            .zip(after_v.iter())
            .position(|(a, b)| a != b);
        if before_k.len() != after_k.len() || before_v.len() != after_v.len() {
            return Some(format!(
                "layer={before_layer} len mismatch k {}->{} v {}->{}",
                before_k.len(),
                after_k.len(),
                before_v.len(),
                after_v.len()
            ));
        }
        if k_diff.is_some() || v_diff.is_some() {
            let sample = |before: &[u8], after: &[u8], diff: Option<usize>| {
                let start = diff.unwrap_or(0);
                let end = (start + 16).min(before.len()).min(after.len());
                format!(
                    "before={:?} after={:?}",
                    &before[start..end],
                    &after[start..end]
                )
            };
            return Some(format!(
                "layer={before_layer} first_k_diff={:?} first_v_diff={:?} k_sample={} v_sample={}",
                k_diff,
                v_diff,
                sample(before_k, after_k, k_diff),
                sample(before_v, after_v, v_diff)
            ));
        }
    }
    None
}
