use std::path::PathBuf;

use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
use supersonic_core::registry::ModelVariant;
use supersonic_runtime::qwen36_moe::engine::Qwen36MoeEngine;
use supersonic_runtime::session::InferenceSession;
use supersonic_runtime::state::{
    build_resolved,
    model_source::{ModelSource, ResolvedModelSource},
    LoaderConfig,
};

fn greedy_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .expect("serving logits must be non-empty")
}

#[test]
fn dense_load_reset_and_reuse_preserve_resident_model_without_serving_allocations(
) -> anyhow::Result<()> {
    let flm_path = PathBuf::from(
        std::env::var_os("SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM")
            .expect("SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM must name the production FLM"),
    );
    let state = build_resolved(
        LoaderConfig {
            model: ModelVariant::Qwen3_6_35B_A3B.to_string(),
            model_dir: flm_path.clone(),
            backend: "hip".to_string(),
            device: 0,
            max_context: 16_384,
            int4: false,
            q4km: false,
            q4km_gptq: false,
            fp8_runtime: false,
            kv_fp8: false,
            dflash: false,
            dflash_draft_dir: None,
            dflash_block: None,
            dflash_tap_layers: None,
            api_key: None,
            cors_allow_origin: None,
            response_store_max_entries: 16,
            max_queued_requests: 4,
            queue_timeout_ms: 1_000,
            no_download: true,
            prefix_cache_enabled: true,
            prefix_cache_dir: None,
            prefix_cache_min_tokens: 128,
            prefix_cache_max_entries: 1,
            prefix_cache_max_bytes: None,
            prefix_cache_memory_ttl_secs: 600,
            prefix_cache_disk_ttl_secs: 86_400,
        },
        ResolvedModelSource {
            source: ModelSource::Flm(flm_path.clone()),
            model: ModelVariant::Qwen3_6_35B_A3B,
        },
    )?;
    assert!(
        state.qwen36_moe_engine.is_none(),
        "FLM startup must not retain a compatibility engine owner"
    );
    let session = state
        .session
        .as_ref()
        .expect("FLM startup must own the engine through InferenceSession");
    assert_eq!(std::sync::Arc::strong_count(session), 1);
    let mut session = session.blocking_lock();
    assert_eq!(session.prefix_snapshot_bytes(1), usize::MAX);
    let engine = match &mut *session {
        InferenceSession::Qwen36Moe(engine) => engine,
        _ => panic!("FLM startup must select the production Qwen3.6 session variant"),
    };

    assert!(engine.tokenizer().get_vocab_size(false) > 0);
    assert!(!engine.chat_template_source().is_empty());
    assert!(!engine.eos_ids().is_empty());
    let evidence = engine.load_evidence().clone();
    assert_eq!(evidence.flm_path, flm_path);
    assert_eq!(evidence.architecture_id, ARCH_QWEN3_6_MOE);
    assert_eq!(evidence.model_id, MODEL_QWEN3_6_MOE_V1);
    assert!(!evidence.storage_abi_ids.is_empty());
    assert!(evidence.direct_profile.native_int4 > 0);
    assert_eq!(evidence.direct_profile.bf16_fallback, 0);
    assert!(evidence.source_bytes > 0);
    assert!(evidence.device_upload_bytes > 0);
    assert_eq!(evidence.source_open_count, 1);
    assert!(evidence.source_open_duration > std::time::Duration::ZERO);
    assert!(evidence.store_open_duration > std::time::Duration::ZERO);
    assert!(evidence.config_duration > std::time::Duration::ZERO);
    assert!(evidence.descriptor_duration > std::time::Duration::ZERO);
    assert!(evidence.allocation_duration > std::time::Duration::ZERO);
    assert!(evidence.upload_duration > std::time::Duration::ZERO);
    assert!(evidence.total_duration >= evidence.source_open_duration);
    assert!(evidence.total_duration >= evidence.descriptor_duration);
    assert!(evidence.total_duration >= evidence.allocation_duration);
    assert!(evidence.total_duration >= evidence.upload_duration);
    assert!(evidence.resident_allocation_count > 0);
    assert!(!evidence.resident_allocation_pointers.is_empty());
    assert!(!evidence.mapped_virtual_ranges.is_empty());
    assert!(evidence.config.is_some());
    assert!(evidence.tokenizer_timings.asset_lookup > std::time::Duration::ZERO);
    assert!(evidence.tokenizer_timings.parse > std::time::Duration::ZERO);
    assert!(evidence.tokenizer_timings.build > std::time::Duration::ZERO);
    assert!(evidence.hal_profile.total_calls > 0);
    assert!(evidence.hal_profile.alloc_bytes > 0);

    let loaded = engine.test_only_reset_snapshot()?;
    assert!(loaded.mutable_nonzero_labels.is_empty());
    assert_eq!(loaded.source_open_count, evidence.source_open_count);
    assert_eq!(
        loaded.resident_allocation_pointers,
        evidence.resident_allocation_pointers
    );
    assert_eq!(
        loaded.resident_allocation_pointers.len() as u64,
        evidence.resident_allocation_count
    );
    assert_eq!(loaded.mapped_virtual_ranges, evidence.mapped_virtual_ranges);

    engine.test_only_dirty_reset_state()?;
    let dirty = engine.test_only_reset_snapshot()?;
    for category in [
        "linear-conv-state",
        "linear-recurrent-state",
        "kv-vmm",
        "kv-shadow",
        "persistent-scratch",
        "logits",
        "counter",
        "final-hidden",
    ] {
        assert!(
            dirty
                .mutable_nonzero_labels
                .iter()
                .any(|label| label.contains(category)),
            "missing dirty category {category}: {:?}",
            dirty.mutable_nonzero_labels
        );
    }
    assert!(dirty.route_history_entries > 0);
    assert_eq!(dirty.route_observations, 0);
    assert_eq!(dirty.transition_candidates, 0);
    assert!(dirty.next_position.is_some());
    assert_eq!(dirty.source_open_count, loaded.source_open_count);
    assert!(
        dirty
            .mapped_virtual_ranges
            .iter()
            .any(|range| range.stats.mapping_count > 1),
        "dirty hook did not create a discontiguous VMM mapping"
    );

    gpu_hal::hal_profile_set_enabled(true);
    gpu_hal::hal_profile_reset();
    engine.reset()?;
    let after = engine.test_only_reset_snapshot()?;
    engine.reset()?;
    let after_repeated = engine.test_only_reset_snapshot()?;
    let reset_profile = gpu_hal::hal_profile_snapshot();
    gpu_hal::hal_profile_set_enabled(false);

    assert!(after.mutable_nonzero_labels.is_empty());
    assert_eq!(after.route_history_entries, 0);
    assert_eq!(after.route_observations, 0);
    assert_eq!(after.transition_candidates, 0);
    assert_eq!(after.next_position, None);
    assert_eq!(
        after.resident_allocation_pointers,
        dirty.resident_allocation_pointers
    );
    assert_eq!(after.mapped_virtual_ranges, dirty.mapped_virtual_ranges);
    assert_eq!(
        after.persistent_descriptor_bytes,
        dirty.persistent_descriptor_bytes
    );
    assert_eq!(after.source_open_count, dirty.source_open_count);
    assert_eq!(after_repeated, after);
    assert_eq!(reset_profile.alloc_calls, 0);
    assert!(
        reset_profile.entries.iter().all(|entry| {
            !entry.op.starts_with("vmm_reserve")
                && !entry.op.starts_with("vmm_map")
                && !entry.op.starts_with("vmm_unmap")
        }),
        "{:?}",
        reset_profile.entries
    );

    let prompt = engine
        .tokenizer()
        .encode("Hello from Sofia", false)
        .map_err(|error| anyhow::anyhow!("encode lifecycle prompt: {error}"))?
        .get_ids()
        .to_vec();
    assert!(!prompt.is_empty());
    let serving_load_sequence = Qwen36MoeEngine::test_only_observed_load_sequence();
    let serving_resident_pointers = after.resident_allocation_pointers.clone();

    gpu_hal::hal_profile_set_enabled(true);
    gpu_hal::hal_profile_reset();
    let first_prefill_logits = engine.prefill(&prompt)?;
    assert_eq!(first_prefill_logits.len(), 248_320);
    let mut generated_ids = vec![greedy_token(&first_prefill_logits)];
    let decode_logits = engine.decode_step(generated_ids[0], prompt.len())?;
    assert_eq!(decode_logits.len(), 248_320);
    generated_ids.push(greedy_token(&decode_logits));
    assert!(!generated_ids.is_empty());
    assert_eq!(
        engine.test_only_reset_snapshot()?.next_position,
        Some(prompt.len() + 1)
    );

    engine.reset()?;
    assert_eq!(
        engine.test_only_reset_snapshot()?.next_position,
        None,
        "reset must return the engine to prefill-ready state"
    );
    let repeated_prefill_logits = engine.prefill(&prompt)?;
    assert_eq!(repeated_prefill_logits.len(), first_prefill_logits.len());
    for (index, (first, repeated)) in first_prefill_logits
        .iter()
        .zip(&repeated_prefill_logits)
        .enumerate()
    {
        assert_eq!(
            repeated.to_bits(),
            first.to_bits(),
            "repeat-prefill logit {index} changed across reset/reuse"
        );
    }

    let reused = engine.test_only_reset_snapshot()?;
    let serving_profile = gpu_hal::hal_profile_snapshot();
    gpu_hal::hal_profile_set_enabled(false);

    assert_eq!(
        Qwen36MoeEngine::test_only_observed_load_sequence(),
        serving_load_sequence
    );
    assert_eq!(reused.source_open_count, evidence.source_open_count);
    assert_eq!(
        reused.resident_allocation_pointers,
        serving_resident_pointers
    );
    assert_eq!(serving_profile.alloc_calls, 0);
    assert!(
        serving_profile.entries.iter().all(|entry| {
            !entry.op.starts_with("vmm_reserve")
                && !entry.op.starts_with("vmm_map")
                && !entry.op.starts_with("vmm_unmap")
        }),
        "{:?}",
        serving_profile.entries
    );
    Ok(())
}
