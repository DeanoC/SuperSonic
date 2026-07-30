use gpu_hal::VirtualArena;

use crate::qwen36_moe::residency::MoeExpertResidencyManager;
use crate::qwen36_moe::types::LayerBuffers;

/// Runtime-owned production layer set and the backing allocations that keep
/// every descriptor pointer valid for the lifetime of the model.
pub struct LoadedQwen36Layers {
    pub layers: Vec<LayerBuffers>,
    pub moe_expert_arena: Option<VirtualArena>,
    pub moe_expert_residency: Option<MoeExpertResidencyManager>,
}

impl LoadedQwen36Layers {
    pub fn dense(layers: Vec<LayerBuffers>) -> Self {
        Self {
            layers,
            moe_expert_arena: None,
            moe_expert_residency: None,
        }
    }

    pub fn as_slice(&self) -> &[LayerBuffers] {
        &self.layers
    }

    pub fn as_mut_slice(&mut self) -> &mut [LayerBuffers] {
        &mut self.layers
    }
}
