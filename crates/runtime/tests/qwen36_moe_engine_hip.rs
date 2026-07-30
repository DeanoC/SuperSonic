use std::path::PathBuf;

use gpu_hal::Backend;
use model_store::flm::{ARCH_QWEN3_6_MOE, MODEL_QWEN3_6_MOE_V1};
use model_store::VirtualArenaTransferBackend;
use supersonic_runtime::qwen36_moe::engine::{Qwen36MoeEngine, Qwen36MoeLoadConfig};
use supersonic_runtime::qwen36_moe::load_policy::Qwen36MoeLoadPolicy;
use supersonic_runtime::qwen36_moe_config::{
    Qwen36KvVmmMode, Qwen36MoeRuntimeConfig, Qwen36MoeRuntimeConfigInputs,
};

#[test]
fn load_and_reset_preserve_resident_model() -> anyhow::Result<()> {
    let flm_path = PathBuf::from(
        std::env::var_os("SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM")
            .expect("SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM must name the production FLM"),
    );
    let moe = Qwen36MoeRuntimeConfig::from_inputs(
        &Qwen36MoeRuntimeConfigInputs {
            vmm_mode: Some("1"),
            island_cap_experts: Some("8"),
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
        max_context_len: 16,
        policy: Qwen36MoeLoadPolicy {
            persistent_decode: true,
            kv_fp8: false,
            kv_vmm: Qwen36KvVmmMode::Force,
            moe,
            virtual_transfer_backend: VirtualArenaTransferBackend::PageableH2d,
        },
        verify_block_hashes: false,
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
    assert!(evidence.resident_allocation_count > 0);
    assert!(!evidence.resident_weight_pointers.is_empty());
    assert!(!evidence.mapped_virtual_addresses.is_empty());

    gpu_hal::hal_profile_set_enabled(true);
    gpu_hal::hal_profile_reset();
    engine.reset()?;
    engine.reset()?;
    let reset_profile = gpu_hal::hal_profile_snapshot();
    gpu_hal::hal_profile_set_enabled(false);

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
    let after = engine.load_evidence();
    assert_eq!(after.load_sequence, evidence.load_sequence);
    assert_eq!(after.source_open_count, evidence.source_open_count);
    assert_eq!(
        after.resident_allocation_count,
        evidence.resident_allocation_count
    );
    assert_eq!(
        after.resident_weight_pointers,
        evidence.resident_weight_pointers
    );
    assert_eq!(
        after.mapped_virtual_addresses,
        evidence.mapped_virtual_addresses
    );
    Ok(())
}
