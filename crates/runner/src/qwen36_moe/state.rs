//! Qwen3.6-MoE linear-attention state snapshot/restore adapters.
//!
//! The runtime crate owns the generic GPU-buffer snapshot contract. The
//! runner owns concrete `LayerBuffers`, so this module preserves the existing
//! runner-facing API by projecting those layer buffers into runtime state
//! views.
#![allow(dead_code)]

use anyhow::Result;
use supersonic_runtime::qwen36_moe::state as runtime_state;

use crate::qwen36_moe_types::{AttnLayerBuffers, LayerBuffers};

pub type LinearAttnLayerSnapshot = runtime_state::LinearAttnLayerSnapshot;
pub type LinearAttnSnapshot = runtime_state::LinearAttnSnapshot;

/// Allocate fresh shadow buffers for every Linear layer and copy the current
/// state into them. Full-attention layers are represented as `None` slots.
pub fn save_linear_attn_state(
    ordinal: usize,
    layers: &[LayerBuffers],
) -> Result<LinearAttnSnapshot> {
    let views = linear_attn_state_refs(layers);
    runtime_state::save_linear_attn_state_from_views(ordinal, &views)
}

/// Refresh an existing snapshot in place without reallocating its shadow
/// buffers.
pub fn refresh_linear_attn_state(
    ordinal: usize,
    layers: &[LayerBuffers],
    snapshot: &mut LinearAttnSnapshot,
) -> Result<()> {
    let views = linear_attn_state_refs(layers);
    runtime_state::refresh_linear_attn_state_from_views(ordinal, &views, snapshot)
}

/// Restore the layers' mutable linear-attention state from `snapshot`.
pub fn restore_linear_attn_state(
    ordinal: usize,
    layers: &mut [LayerBuffers],
    snapshot: &LinearAttnSnapshot,
) -> Result<()> {
    let mut views = linear_attn_state_muts(layers);
    runtime_state::restore_linear_attn_state_from_views(ordinal, &mut views, snapshot)
}

fn linear_attn_state_refs(
    layers: &[LayerBuffers],
) -> Vec<Option<runtime_state::LinearAttnLayerStateRef<'_>>> {
    layers
        .iter()
        .map(|layer| match &layer.attn {
            AttnLayerBuffers::Full { .. } => None,
            AttnLayerBuffers::Linear {
                conv_state,
                recurrent_state,
                ..
            } => Some(runtime_state::LinearAttnLayerStateRef {
                conv_state,
                recurrent_state,
            }),
        })
        .collect()
}

fn linear_attn_state_muts(
    layers: &mut [LayerBuffers],
) -> Vec<Option<runtime_state::LinearAttnLayerStateMut<'_>>> {
    layers
        .iter_mut()
        .map(|layer| match &mut layer.attn {
            AttnLayerBuffers::Full { .. } => None,
            AttnLayerBuffers::Linear {
                conv_state,
                recurrent_state,
                ..
            } => Some(runtime_state::LinearAttnLayerStateMut {
                conv_state,
                recurrent_state,
            }),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_count_with_no_layers() {
        let snap = LinearAttnSnapshot { layers: Vec::new() };
        assert_eq!(snap.linear_layer_count(), 0);
    }
}
