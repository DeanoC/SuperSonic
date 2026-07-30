use std::path::PathBuf;

use gpu_hal::Backend;
use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
use model_store::VirtualArenaTransferBackend;
use supersonic_runtime::qwen36_moe::engine::{Qwen36MoeEngine, Qwen36MoeLoadConfig};
use supersonic_runtime::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
use supersonic_runtime::qwen36_moe_config::{
    Qwen36KvVmmMode, Qwen36MoeRuntimeConfig, Qwen36MoeRuntimeConfigInputs,
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
    let moe = Qwen36MoeRuntimeConfig::from_inputs(
        &Qwen36MoeRuntimeConfigInputs {
            vmm_mode: Some("0"),
            ..Default::default()
        },
        false,
        Backend::Hip,
        8,
    )?;
    let mut engine = Qwen36MoeEngine::load(Qwen36MoeLoadConfig {
        flm_path: flm_path.clone(),
        backend: Backend::Hip,
        device_ordinal: 0,
        max_context_len: 16_384,
        policy: Qwen36MoeLoadPolicy {
            persistent_decode: true,
            kv_fp8: false,
            kv_vmm: Qwen36KvVmmMode::Force,
            moe,
            virtual_transfer_backend: VirtualArenaTransferBackend::PageableH2d,
        },
        verify_block_hashes: false,
        execution_options: supersonic_runtime::qwen36_moe::decode::Qwen36ExecutionOptions::default(
        ),
        accurate_stage_timings: false,
    })?;

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
