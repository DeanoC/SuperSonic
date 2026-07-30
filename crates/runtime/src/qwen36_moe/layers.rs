use gpu_hal::VirtualArena;

use crate::qwen36_moe::layer_loader::Qwen36WeightMode;
use crate::qwen36_moe::persistent_decode::PersistentScratch;
use crate::qwen36_moe::residency::MoeExpertResidencyManager;
use crate::qwen36_moe::types::{LayerBuffers, MultiLayerGeom};
use gpu_hal::GpuBuffer;

/// Runtime-owned production layer set and the backing allocations that keep
/// every descriptor pointer valid for the lifetime of the model.
pub struct LoadedQwen36Layers {
    layers: Vec<LayerBuffers>,
    weight_mode: Qwen36WeightMode,
    moe_expert_arena: Option<VirtualArena>,
    moe_expert_residency: Option<MoeExpertResidencyManager>,
    persistent_scratch: Option<PersistentScratch>,
}

#[derive(Debug, Clone, Copy)]
pub struct PersistentScratchStats {
    pub descriptor_bytes: usize,
    pub workspace_bytes: usize,
    pub hidden_bytes: usize,
}

impl LoadedQwen36Layers {
    pub fn dense(layers: Vec<LayerBuffers>, weight_mode: Qwen36WeightMode) -> Self {
        Self {
            layers,
            weight_mode,
            moe_expert_arena: None,
            moe_expert_residency: None,
            persistent_scratch: None,
        }
    }

    pub(crate) fn with_backing(
        layers: Vec<LayerBuffers>,
        weight_mode: Qwen36WeightMode,
        moe_expert_arena: Option<VirtualArena>,
        moe_expert_residency: Option<MoeExpertResidencyManager>,
    ) -> Self {
        Self {
            layers,
            weight_mode,
            moe_expert_arena,
            moe_expert_residency,
            persistent_scratch: None,
        }
    }

    pub fn weight_mode(&self) -> Qwen36WeightMode {
        self.weight_mode
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layers(&self) -> &[LayerBuffers] {
        &self.layers
    }

    pub fn layers_mut_before_persistent(&mut self) -> anyhow::Result<&mut [LayerBuffers]> {
        if self.persistent_scratch.is_some() {
            anyhow::bail!(
                "Qwen3.6 layers cannot be mutably exposed while persistent descriptors are active"
            );
        }
        Ok(&mut self.layers)
    }

    pub fn has_virtual_expert_arena(&self) -> bool {
        self.moe_expert_arena.is_some()
    }

    pub fn has_sparse_expert_residency(&self) -> bool {
        self.moe_expert_residency.is_some()
    }

    pub fn virtual_expert_arena(&self) -> Option<&VirtualArena> {
        self.moe_expert_arena.as_ref()
    }

    pub fn sparse_expert_residency(&self) -> Option<&MoeExpertResidencyManager> {
        self.moe_expert_residency.as_ref()
    }

    pub fn persistent_enabled(&self) -> bool {
        self.persistent_scratch.is_some()
    }

    pub fn persistent_scratch_stats(&self) -> Option<PersistentScratchStats> {
        self.persistent_scratch
            .as_ref()
            .map(|scratch| PersistentScratchStats {
                descriptor_bytes: scratch.layer_descs_dev.len_bytes(),
                workspace_bytes: scratch.workspace.len_bytes(),
                hidden_bytes: scratch.hidden_ping.len_bytes(),
            })
    }

    pub fn enable_persistent(
        &mut self,
        ordinal: usize,
        geom: &MultiLayerGeom,
    ) -> anyhow::Result<()> {
        if self.persistent_scratch.is_some() {
            return Ok(());
        }
        self.persistent_scratch = Some(PersistentScratch::new(ordinal, geom, &mut self.layers)?);
        Ok(())
    }

    pub(crate) fn execution_parts(
        &mut self,
    ) -> (
        &mut [LayerBuffers],
        Option<&mut PersistentScratch>,
        Option<&mut MoeExpertResidencyManager>,
    ) {
        (
            &mut self.layers,
            self.persistent_scratch.as_mut(),
            self.moe_expert_residency.as_mut(),
        )
    }

    pub fn run_dense_prefill_tokens_from_device_embedding(
        &mut self,
        ordinal: usize,
        embed_w: &GpuBuffer,
        tokens: &[u32],
        start_position: i32,
        start_cache_pos: i32,
    ) -> anyhow::Result<std::time::Duration> {
        self.persistent_scratch
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.6 persistent decode is not enabled"))?
            .run_dense_prefill_tokens_from_device_embedding(
                ordinal,
                embed_w,
                tokens,
                start_position,
                start_cache_pos,
            )
    }

    pub fn run_from_device_embedding_no_download(
        &mut self,
        ordinal: usize,
        embed_w: &GpuBuffer,
        token: u32,
        position: i32,
        cache_pos: i32,
    ) -> anyhow::Result<std::time::Duration> {
        self.persistent_scratch
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.6 persistent decode is not enabled"))?
            .run_from_device_embedding_no_download(ordinal, embed_w, token, position, cache_pos)
    }

    pub fn run_segmented_profile(
        &mut self,
        ordinal: usize,
        initial_hidden: &[u8],
        position: i32,
        cache_pos: i32,
    ) -> anyhow::Result<crate::qwen36_moe::types::DecodeOutputs> {
        self.persistent_scratch
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.6 persistent decode is not enabled"))?
            .run_segmented_profile(ordinal, initial_hidden, position, cache_pos)
    }

    /// Exposes base-model state only to runner-owned experimental MTP code.
    ///
    /// # Safety
    ///
    /// The closure must not move, replace, or retain any layer or scratch
    /// allocation. Production serving must use runtime chain/prefill methods.
    pub unsafe fn with_experimental_parts<R>(
        &mut self,
        f: impl FnOnce(&mut [LayerBuffers], Option<&mut PersistentScratch>) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        f(&mut self.layers, self.persistent_scratch.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen36_moe::layer_loader::Qwen36WeightMode;

    #[test]
    fn owner_retains_weight_mode_and_hides_replaceable_backing_parts() {
        let loaded = LoadedQwen36Layers::dense(Vec::new(), Qwen36WeightMode::Int4);

        assert_eq!(loaded.weight_mode(), Qwen36WeightMode::Int4);
        assert_eq!(loaded.len(), 0);
        assert!(!loaded.has_virtual_expert_arena());
        assert!(!loaded.has_sparse_expert_residency());
        assert!(!loaded.persistent_enabled());
    }
}
