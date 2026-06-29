//! Runtime-owned Qwen3.6-MoE linear-attention state snapshot/restore
//! primitives.
//!
//! Qwen3.6 linear-attention layers mutate `conv_state` and
//! `recurrent_state` every token. Speculative verify chains need a cheap
//! rollback point, so the runtime owns the generic GPU-buffer snapshot
//! contract while model-specific crates provide adapters from their concrete
//! layer-buffer layout.

use anyhow::{Context, Result};
use gpu_hal::{copy_d2d, GpuBuffer, ScalarType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearAttnStateLayout {
    pub conv_dtype: ScalarType,
    pub conv_shape: Vec<usize>,
    pub conv_len_bytes: usize,
    pub recurrent_dtype: ScalarType,
    pub recurrent_shape: Vec<usize>,
    pub recurrent_len_bytes: usize,
}

impl LinearAttnStateLayout {
    pub fn from_buffers(conv_state: &GpuBuffer, recurrent_state: &GpuBuffer) -> Self {
        Self {
            conv_dtype: conv_state.dtype(),
            conv_shape: conv_state.shape().to_vec(),
            conv_len_bytes: conv_state.len_bytes(),
            recurrent_dtype: recurrent_state.dtype(),
            recurrent_shape: recurrent_state.shape().to_vec(),
            recurrent_len_bytes: recurrent_state.len_bytes(),
        }
    }
}

/// Borrowed live state for a single linear-attention layer.
#[derive(Clone, Copy)]
pub struct LinearAttnLayerStateRef<'a> {
    pub conv_state: &'a GpuBuffer,
    pub recurrent_state: &'a GpuBuffer,
}

impl<'a> LinearAttnLayerStateRef<'a> {
    pub fn layout(self) -> LinearAttnStateLayout {
        LinearAttnStateLayout::from_buffers(self.conv_state, self.recurrent_state)
    }
}

/// Mutably borrowed live state for restoring a linear-attention layer.
pub struct LinearAttnLayerStateMut<'a> {
    pub conv_state: &'a mut GpuBuffer,
    pub recurrent_state: &'a mut GpuBuffer,
}

impl<'a> LinearAttnLayerStateMut<'a> {
    pub fn layout(&self) -> LinearAttnStateLayout {
        LinearAttnStateLayout::from_buffers(self.conv_state, self.recurrent_state)
    }
}

/// Per-linear-attention-layer shadow buffers. Full-attention layers have no
/// entry in the parent snapshot.
pub struct LinearAttnLayerSnapshot {
    pub conv_state: GpuBuffer,
    pub recurrent_state: GpuBuffer,
}

impl LinearAttnLayerSnapshot {
    pub fn layout(&self) -> LinearAttnStateLayout {
        LinearAttnStateLayout::from_buffers(&self.conv_state, &self.recurrent_state)
    }
}

/// Snapshot of every linear-attention layer's state across a layer slice.
/// Indexed by layer position; `None` entries correspond to full-attention
/// layers that do not carry linear state.
pub struct LinearAttnSnapshot {
    pub layers: Vec<Option<LinearAttnLayerSnapshot>>,
}

impl LinearAttnSnapshot {
    pub fn linear_layer_count(&self) -> usize {
        self.layers.iter().filter(|layer| layer.is_some()).count()
    }

    pub fn layer_layouts(&self) -> Vec<Option<LinearAttnStateLayout>> {
        self.layers
            .iter()
            .map(|layer| layer.as_ref().map(LinearAttnLayerSnapshot::layout))
            .collect()
    }
}

pub fn count_linear_attn_layers(layers: &[Option<LinearAttnStateLayout>]) -> usize {
    layers.iter().filter(|layer| layer.is_some()).count()
}

pub fn validate_linear_attn_snapshot_layouts(
    operation: &str,
    layers: &[Option<LinearAttnStateLayout>],
    snapshots: &[Option<LinearAttnStateLayout>],
) -> Result<()> {
    if snapshots.len() != layers.len() {
        anyhow::bail!(
            "{operation}: snapshot has {} layer slots but layers slice has {} entries - snapshot/layers shape mismatch",
            snapshots.len(),
            layers.len()
        );
    }
    for (idx, (layer, snapshot)) in layers.iter().zip(snapshots.iter()).enumerate() {
        validate_linear_attn_snapshot_slot(operation, idx, layer.as_ref(), snapshot.as_ref())?;
    }
    Ok(())
}

pub fn save_linear_attn_state_from_views(
    ordinal: usize,
    layers: &[Option<LinearAttnLayerStateRef<'_>>],
) -> Result<LinearAttnSnapshot> {
    let mut snap_layers: Vec<Option<LinearAttnLayerSnapshot>> = Vec::with_capacity(layers.len());
    for (idx, layer) in layers.iter().enumerate() {
        match layer {
            None => snap_layers.push(None),
            Some(layer) => {
                let conv_shadow = GpuBuffer::zeros(
                    ordinal,
                    layer.conv_state.dtype(),
                    layer.conv_state.shape(),
                )
                .with_context(|| format!("alloc conv_state shadow for linear layer {idx}"))?;
                let rec_shadow = GpuBuffer::zeros(
                    ordinal,
                    layer.recurrent_state.dtype(),
                    layer.recurrent_state.shape(),
                )
                .with_context(|| format!("alloc recurrent_state shadow for linear layer {idx}"))?;
                let mut layer_snap = LinearAttnLayerSnapshot {
                    conv_state: conv_shadow,
                    recurrent_state: rec_shadow,
                };
                copy_into_layer_snapshot(ordinal, idx, *layer, &mut layer_snap)?;
                snap_layers.push(Some(layer_snap));
            }
        }
    }
    Ok(LinearAttnSnapshot {
        layers: snap_layers,
    })
}

pub fn refresh_linear_attn_state_from_views(
    ordinal: usize,
    layers: &[Option<LinearAttnLayerStateRef<'_>>],
    snapshot: &mut LinearAttnSnapshot,
) -> Result<()> {
    validate_layer_slot_count(
        "refresh_linear_attn_state",
        layers.len(),
        snapshot.layers.len(),
    )?;
    for (idx, (layer, slot)) in layers.iter().zip(snapshot.layers.iter_mut()).enumerate() {
        match (layer, slot.as_mut()) {
            (None, None) => {}
            (None, Some(_)) => bail_full_live_linear_snapshot("refresh_linear_attn_state", idx)?,
            (Some(_), None) => bail_linear_live_full_snapshot("refresh_linear_attn_state", idx)?,
            (Some(layer), Some(layer_snap)) => {
                copy_into_layer_snapshot(ordinal, idx, *layer, layer_snap)?;
            }
        }
    }
    Ok(())
}

pub fn restore_linear_attn_state_from_views(
    ordinal: usize,
    layers: &mut [Option<LinearAttnLayerStateMut<'_>>],
    snapshot: &LinearAttnSnapshot,
) -> Result<()> {
    validate_layer_slot_count(
        "restore_linear_attn_state",
        layers.len(),
        snapshot.layers.len(),
    )?;
    for (idx, (layer, slot)) in layers.iter_mut().zip(snapshot.layers.iter()).enumerate() {
        match (layer.as_mut(), slot) {
            (None, None) => {}
            (None, Some(_)) => bail_full_live_linear_snapshot("restore_linear_attn_state", idx)?,
            (Some(_), None) => bail_linear_live_full_snapshot("restore_linear_attn_state", idx)?,
            (Some(layer), Some(layer_snap)) => {
                copy_from_layer_snapshot(ordinal, idx, layer, layer_snap)?;
            }
        }
    }
    Ok(())
}

fn validate_layer_slot_count(
    operation: &str,
    layer_count: usize,
    snapshot_count: usize,
) -> Result<()> {
    if snapshot_count != layer_count {
        anyhow::bail!(
            "{operation}: snapshot has {snapshot_count} layer slots but layers slice has {layer_count} entries - snapshot/layers shape mismatch"
        );
    }
    Ok(())
}

fn validate_linear_attn_snapshot_slot(
    operation: &str,
    idx: usize,
    layer: Option<&LinearAttnStateLayout>,
    snapshot: Option<&LinearAttnStateLayout>,
) -> Result<()> {
    match (layer, snapshot) {
        (None, None) => Ok(()),
        (None, Some(_)) => bail_full_live_linear_snapshot(operation, idx),
        (Some(_), None) => bail_linear_live_full_snapshot(operation, idx),
        (Some(layer), Some(snapshot)) => {
            validate_linear_attn_layout(operation, idx, layer, snapshot)
        }
    }
}

fn validate_linear_attn_layout(
    operation: &str,
    idx: usize,
    layer: &LinearAttnStateLayout,
    snapshot: &LinearAttnStateLayout,
) -> Result<()> {
    if layer.conv_dtype != snapshot.conv_dtype
        || layer.conv_shape != snapshot.conv_shape
        || layer.conv_len_bytes != snapshot.conv_len_bytes
    {
        anyhow::bail!(
            "{operation}: layer {idx} conv_state layout mismatch (live dtype={:?} shape={:?} bytes={}, snapshot dtype={:?} shape={:?} bytes={})",
            layer.conv_dtype,
            layer.conv_shape,
            layer.conv_len_bytes,
            snapshot.conv_dtype,
            snapshot.conv_shape,
            snapshot.conv_len_bytes
        );
    }
    if layer.recurrent_dtype != snapshot.recurrent_dtype
        || layer.recurrent_shape != snapshot.recurrent_shape
        || layer.recurrent_len_bytes != snapshot.recurrent_len_bytes
    {
        anyhow::bail!(
            "{operation}: layer {idx} recurrent_state layout mismatch (live dtype={:?} shape={:?} bytes={}, snapshot dtype={:?} shape={:?} bytes={})",
            layer.recurrent_dtype,
            layer.recurrent_shape,
            layer.recurrent_len_bytes,
            snapshot.recurrent_dtype,
            snapshot.recurrent_shape,
            snapshot.recurrent_len_bytes
        );
    }
    Ok(())
}

fn validate_buffer_layout(
    operation: &str,
    idx: usize,
    name: &str,
    live: &GpuBuffer,
    snapshot: &GpuBuffer,
) -> Result<()> {
    if live.dtype() != snapshot.dtype()
        || live.shape() != snapshot.shape()
        || live.len_bytes() != snapshot.len_bytes()
    {
        anyhow::bail!(
            "{operation}: layer {idx} {name} layout mismatch (live dtype={:?} shape={:?} bytes={}, snapshot dtype={:?} shape={:?} bytes={})",
            live.dtype(),
            live.shape(),
            live.len_bytes(),
            snapshot.dtype(),
            snapshot.shape(),
            snapshot.len_bytes()
        );
    }
    Ok(())
}

fn copy_into_layer_snapshot(
    ordinal: usize,
    idx: usize,
    layer: LinearAttnLayerStateRef<'_>,
    layer_snap: &mut LinearAttnLayerSnapshot,
) -> Result<()> {
    validate_buffer_layout(
        "linear_attn_snapshot",
        idx,
        "conv_state",
        layer.conv_state,
        &layer_snap.conv_state,
    )?;
    validate_buffer_layout(
        "linear_attn_snapshot",
        idx,
        "recurrent_state",
        layer.recurrent_state,
        &layer_snap.recurrent_state,
    )?;

    let n_conv = layer.conv_state.len_bytes();
    copy_d2d(
        ordinal,
        layer_snap.conv_state.as_mut_ptr(),
        layer.conv_state.as_ptr(),
        n_conv,
    )
    .with_context(|| format!("snapshot conv_state for layer {idx}"))?;

    let n_rec = layer.recurrent_state.len_bytes();
    copy_d2d(
        ordinal,
        layer_snap.recurrent_state.as_mut_ptr(),
        layer.recurrent_state.as_ptr(),
        n_rec,
    )
    .with_context(|| format!("snapshot recurrent_state for layer {idx}"))?;
    Ok(())
}

fn copy_from_layer_snapshot(
    ordinal: usize,
    idx: usize,
    layer: &mut LinearAttnLayerStateMut<'_>,
    layer_snap: &LinearAttnLayerSnapshot,
) -> Result<()> {
    validate_buffer_layout(
        "restore_linear_attn_state",
        idx,
        "conv_state",
        layer.conv_state,
        &layer_snap.conv_state,
    )?;
    validate_buffer_layout(
        "restore_linear_attn_state",
        idx,
        "recurrent_state",
        layer.recurrent_state,
        &layer_snap.recurrent_state,
    )?;

    let n_conv = layer.conv_state.len_bytes();
    copy_d2d(
        ordinal,
        layer.conv_state.as_mut_ptr(),
        layer_snap.conv_state.as_ptr(),
        n_conv,
    )
    .with_context(|| format!("restore conv_state for layer {idx}"))?;

    let n_rec = layer.recurrent_state.len_bytes();
    copy_d2d(
        ordinal,
        layer.recurrent_state.as_mut_ptr(),
        layer_snap.recurrent_state.as_ptr(),
        n_rec,
    )
    .with_context(|| format!("restore recurrent_state for layer {idx}"))?;
    Ok(())
}

fn bail_full_live_linear_snapshot(operation: &str, idx: usize) -> Result<()> {
    anyhow::bail!(
        "{operation}: layer {idx} is Full but snapshot has a Linear slot - snapshot/layers pattern mismatch (Full/Linear swap)"
    );
}

fn bail_linear_live_full_snapshot(operation: &str, idx: usize) -> Result<()> {
    anyhow::bail!(
        "{operation}: layer {idx} is Linear but snapshot has no slot - snapshot/layers pattern mismatch (Linear/Full swap)"
    );
}

#[cfg(test)]
mod tests {
    use super::{count_linear_attn_layers, validate_linear_attn_snapshot_layouts};
    use crate::qwen36_moe::state::LinearAttnStateLayout;
    use gpu_hal::ScalarType;

    fn layout(conv_len_bytes: usize, recurrent_len_bytes: usize) -> LinearAttnStateLayout {
        LinearAttnStateLayout {
            conv_dtype: ScalarType::BF16,
            conv_shape: vec![conv_len_bytes / 2],
            conv_len_bytes,
            recurrent_dtype: ScalarType::F32,
            recurrent_shape: vec![recurrent_len_bytes / 4],
            recurrent_len_bytes,
        }
    }

    #[test]
    fn linear_attn_layout_count_ignores_full_layers() {
        let layers = vec![Some(layout(16, 32)), None, Some(layout(8, 16))];

        assert_eq!(count_linear_attn_layers(&layers), 2);
    }

    #[test]
    fn snapshot_layout_validation_reports_pattern_mismatch_layer() {
        let layers = vec![Some(layout(16, 32)), None];
        let snapshots = vec![None, None];

        let err =
            validate_linear_attn_snapshot_layouts("refresh_linear_attn_state", &layers, &snapshots)
                .expect_err("linear/full mismatch must be rejected");

        assert!(
            err.to_string().contains("layer 0 is Linear"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn snapshot_layout_validation_reports_buffer_mismatch_layer() {
        let layers = vec![Some(layout(16, 32))];
        let snapshots = vec![Some(layout(18, 32))];

        let err =
            validate_linear_attn_snapshot_layouts("restore_linear_attn_state", &layers, &snapshots)
                .expect_err("conv size mismatch must be rejected");

        assert!(
            err.to_string()
                .contains("layer 0 conv_state layout mismatch"),
            "unexpected error: {err}"
        );
    }
}
