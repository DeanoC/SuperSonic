use model_store::VirtualArenaTransferBackend;

use crate::qwen36_moe_config::{Qwen36KvVmmMode, Qwen36MoeRuntimeConfig};

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen36MoeLoadPolicy {
    pub persistent_decode: bool,
    pub kv_fp8: bool,
    pub kv_vmm: Qwen36KvVmmMode,
    pub moe: Qwen36MoeRuntimeConfig,
    pub virtual_transfer_backend: VirtualArenaTransferBackend,
}

#[cfg(test)]
mod tests {
    use model_store::VirtualArenaTransferBackend;

    use super::Qwen36MoeLoadPolicy;
    use crate::qwen36_moe_config::{Qwen36KvVmmMode, Qwen36MoeRuntimeConfig};

    #[test]
    fn resolved_load_policy_carries_runtime_decisions_without_translation() {
        let moe = Qwen36MoeRuntimeConfig::default();
        let policy = Qwen36MoeLoadPolicy {
            persistent_decode: true,
            kv_fp8: true,
            kv_vmm: Qwen36KvVmmMode::Force,
            moe: moe.clone(),
            virtual_transfer_backend: VirtualArenaTransferBackend::GpuDirectStorage,
        };

        assert!(policy.persistent_decode);
        assert!(policy.kv_fp8);
        assert_eq!(policy.kv_vmm, Qwen36KvVmmMode::Force);
        assert_eq!(policy.moe, moe);
        assert_eq!(
            policy.virtual_transfer_backend,
            VirtualArenaTransferBackend::GpuDirectStorage
        );
    }
}
