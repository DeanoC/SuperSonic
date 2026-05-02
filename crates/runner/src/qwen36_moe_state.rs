//! Qwen3.6-MoE linear-attention state snapshot/restore primitives.
//!
//! Phase 6.4c.1: foundation for any operation that needs to checkpoint
//! the linear-attention layers' mutable state. The intended consumer is
//! the speculative-decode driver (Phase 6.4c+) — it needs to save state
//! before running speculative verify chains and restore on rejection
//! (since linear-attention's `conv_state` + `recurrent_state` mutate per
//! token and don't naturally roll back).
//!
//! ## What's stateful
//!
//! Each linear-attention layer in `LayerBuffers::Linear` holds two
//! mutable GpuBuffers:
//!   - `conv_state`: `[qkv_dim, kernel-1]` BF16 — the depthwise conv1d
//!     window state.
//!   - `recurrent_state`: `[V * K * Vd]` F32 — the linear-attn delta-rule
//!     recurrent state.
//!
//! Full-attention layers don't carry state (their KV cache is positional
//! — overwriting a slot is a free rollback). So a snapshot only needs
//! per-linear-layer copies.
//!
//! ## Cost
//!
//! At 35B-A3B (30 linear layers, qkv_dim≈3168, recurrent_state ≈ 60 MiB
//! F32 across all layers — `[V=2 × K=16 × Vd=128]` per layer per
//! the docstring): a full snapshot copies ~62 MiB GPU→GPU. At 7900 XTX
//! peak ~960 GB/s, that's ~65 µs of D2D bandwidth — negligible.
//!
//! ## Allocation model
//!
//! [`LinearAttnSnapshot`] owns its shadow buffers. The intended pattern
//! for the speculative-decode driver is to allocate ONE snapshot at
//! engine startup and reuse it across speculative iterations:
//!
//! ```ignore
//! let mut snap = save_linear_attn_state(ordinal, &layers)?;
//! // ... per-spec-step:
//! refresh_linear_attn_state(ordinal, &layers, &mut snap)?;
//! // ... run K verify chains (state mutates) ...
//! if rejected {
//!     restore_linear_attn_state(ordinal, &mut layers, &snap)?;
//! }
//! ```
//!
//! [`refresh_linear_attn_state`] reuses the snapshot's buffers without
//! re-allocating; [`save_linear_attn_state`] does both alloc and copy
//! in one call (convenience for the first save).

use anyhow::{Context, Result};
use gpu_hal::{copy_d2d, GpuBuffer};

use crate::qwen36_moe_decode::{AttnLayerBuffers, LayerBuffers};

/// Per-linear-attn-layer state shadow buffers. Only `Linear` layers
/// have state; `Full` layers are represented as `None` in the parent
/// snapshot's vec.
pub struct LinearAttnLayerSnapshot {
    /// Shadow of `LayerBuffers::Linear::conv_state`. Same shape +
    /// dtype (BF16 `[qkv_dim, kernel-1]`).
    pub conv_state: GpuBuffer,
    /// Shadow of `LayerBuffers::Linear::recurrent_state`. Same shape
    /// + dtype (F32 `[V * K * Vd]`).
    pub recurrent_state: GpuBuffer,
}

/// Snapshot of every linear-attn layer's state across a `LayerBuffers`
/// slice. Indexed by layer position (0..num_layers); `None` entries
/// correspond to `Full` layers that don't have state.
pub struct LinearAttnSnapshot {
    pub layers: Vec<Option<LinearAttnLayerSnapshot>>,
}

impl LinearAttnSnapshot {
    /// Returns the count of Linear layers actually captured (= count
    /// of `Some` entries).
    pub fn linear_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.is_some()).count()
    }
}

/// Allocate fresh shadow buffers for every Linear layer + copy the
/// current state into them. The convenience entry point — for steady-
/// state speculative use prefer [`refresh_linear_attn_state`] which
/// reuses an existing snapshot's buffers.
pub fn save_linear_attn_state(
    ordinal: usize,
    layers: &[LayerBuffers],
) -> Result<LinearAttnSnapshot> {
    let mut snap_layers: Vec<Option<LinearAttnLayerSnapshot>> =
        Vec::with_capacity(layers.len());
    for (idx, layer) in layers.iter().enumerate() {
        match &layer.attn {
            AttnLayerBuffers::Full { .. } => snap_layers.push(None),
            AttnLayerBuffers::Linear {
                conv_state,
                recurrent_state,
                ..
            } => {
                let conv_shadow = GpuBuffer::zeros(
                    ordinal,
                    conv_state.dtype(),
                    conv_state.shape(),
                )
                .with_context(|| {
                    format!("alloc conv_state shadow for linear layer {idx}")
                })?;
                let rec_shadow = GpuBuffer::zeros(
                    ordinal,
                    recurrent_state.dtype(),
                    recurrent_state.shape(),
                )
                .with_context(|| {
                    format!("alloc recurrent_state shadow for linear layer {idx}")
                })?;
                let mut layer_snap = LinearAttnLayerSnapshot {
                    conv_state: conv_shadow,
                    recurrent_state: rec_shadow,
                };
                copy_into_layer(ordinal, idx, conv_state, recurrent_state, &mut layer_snap)?;
                snap_layers.push(Some(layer_snap));
            }
        }
    }
    Ok(LinearAttnSnapshot {
        layers: snap_layers,
    })
}

/// Refresh an existing snapshot in place. Caller must ensure the
/// snapshot was originally created against the SAME `layers` slice
/// (same Linear/Full layout, same per-layer shapes); otherwise the
/// shape sanity checks will fail. Cheaper than rebuilding the
/// snapshot from scratch — D2D copies only, no allocations.
pub fn refresh_linear_attn_state(
    ordinal: usize,
    layers: &[LayerBuffers],
    snapshot: &mut LinearAttnSnapshot,
) -> Result<()> {
    if snapshot.layers.len() != layers.len() {
        anyhow::bail!(
            "refresh_linear_attn_state: snapshot has {} layer slots but \
             layers slice has {} entries — snapshot/layers shape mismatch",
            snapshot.layers.len(),
            layers.len()
        );
    }
    for (idx, (layer, slot)) in layers.iter().zip(snapshot.layers.iter_mut()).enumerate() {
        match (&layer.attn, slot.as_mut()) {
            (AttnLayerBuffers::Full { .. }, None) => {
                // Both agree: no state to copy.
            }
            (AttnLayerBuffers::Full { .. }, Some(_)) => {
                anyhow::bail!(
                    "refresh_linear_attn_state: layer {idx} is Full but \
                     snapshot has a Linear slot — snapshot/layers \
                     pattern mismatch (Full↔Linear swap)"
                );
            }
            (AttnLayerBuffers::Linear { .. }, None) => {
                anyhow::bail!(
                    "refresh_linear_attn_state: layer {idx} is Linear but \
                     snapshot has no slot — snapshot/layers pattern \
                     mismatch (Linear↔Full swap)"
                );
            }
            (
                AttnLayerBuffers::Linear {
                    conv_state,
                    recurrent_state,
                    ..
                },
                Some(layer_snap),
            ) => {
                copy_into_layer(ordinal, idx, conv_state, recurrent_state, layer_snap)?;
            }
        }
    }
    Ok(())
}

/// Restore the layers' state from `snapshot`. Inverse of
/// [`save_linear_attn_state`] / [`refresh_linear_attn_state`]: D2D
/// copies snapshot → layers for every Linear layer.
pub fn restore_linear_attn_state(
    ordinal: usize,
    layers: &mut [LayerBuffers],
    snapshot: &LinearAttnSnapshot,
) -> Result<()> {
    if snapshot.layers.len() != layers.len() {
        anyhow::bail!(
            "restore_linear_attn_state: snapshot has {} layer slots but \
             layers slice has {} entries — snapshot/layers shape mismatch",
            snapshot.layers.len(),
            layers.len()
        );
    }
    for (idx, (layer, slot)) in layers
        .iter_mut()
        .zip(snapshot.layers.iter())
        .enumerate()
    {
        match (&mut layer.attn, slot) {
            (AttnLayerBuffers::Full { .. }, None) => {}
            (AttnLayerBuffers::Full { .. }, Some(_)) => {
                anyhow::bail!(
                    "restore_linear_attn_state: layer {idx} is Full but \
                     snapshot has a Linear slot"
                );
            }
            (AttnLayerBuffers::Linear { .. }, None) => {
                anyhow::bail!(
                    "restore_linear_attn_state: layer {idx} is Linear but \
                     snapshot has no slot"
                );
            }
            (
                AttnLayerBuffers::Linear {
                    conv_state,
                    recurrent_state,
                    ..
                },
                Some(layer_snap),
            ) => {
                let n_conv = conv_state.len_bytes();
                if n_conv != layer_snap.conv_state.len_bytes() {
                    anyhow::bail!(
                        "restore_linear_attn_state: layer {idx} conv_state \
                         size mismatch (live={n_conv}, snapshot={})",
                        layer_snap.conv_state.len_bytes()
                    );
                }
                copy_d2d(
                    ordinal,
                    conv_state.as_mut_ptr(),
                    layer_snap.conv_state.as_ptr(),
                    n_conv,
                )
                .with_context(|| format!("restore conv_state for layer {idx}"))?;

                let n_rec = recurrent_state.len_bytes();
                if n_rec != layer_snap.recurrent_state.len_bytes() {
                    anyhow::bail!(
                        "restore_linear_attn_state: layer {idx} recurrent_state \
                         size mismatch (live={n_rec}, snapshot={})",
                        layer_snap.recurrent_state.len_bytes()
                    );
                }
                copy_d2d(
                    ordinal,
                    recurrent_state.as_mut_ptr(),
                    layer_snap.recurrent_state.as_ptr(),
                    n_rec,
                )
                .with_context(|| {
                    format!("restore recurrent_state for layer {idx}")
                })?;
            }
        }
    }
    Ok(())
}

/// Copy live state → snapshot for one Linear layer. Shared between
/// `save_linear_attn_state` (initial copy) and `refresh_linear_attn_state`
/// (per-spec-step refresh).
fn copy_into_layer(
    ordinal: usize,
    idx: usize,
    conv_state: &GpuBuffer,
    recurrent_state: &GpuBuffer,
    layer_snap: &mut LinearAttnLayerSnapshot,
) -> Result<()> {
    let n_conv = conv_state.len_bytes();
    if n_conv != layer_snap.conv_state.len_bytes() {
        anyhow::bail!(
            "linear_attn_snapshot: layer {idx} conv_state size mismatch \
             (live={n_conv}, snapshot={})",
            layer_snap.conv_state.len_bytes()
        );
    }
    copy_d2d(
        ordinal,
        layer_snap.conv_state.as_mut_ptr(),
        conv_state.as_ptr(),
        n_conv,
    )
    .with_context(|| format!("snapshot conv_state for layer {idx}"))?;

    let n_rec = recurrent_state.len_bytes();
    if n_rec != layer_snap.recurrent_state.len_bytes() {
        anyhow::bail!(
            "linear_attn_snapshot: layer {idx} recurrent_state size mismatch \
             (live={n_rec}, snapshot={})",
            layer_snap.recurrent_state.len_bytes()
        );
    }
    copy_d2d(
        ordinal,
        layer_snap.recurrent_state.as_mut_ptr(),
        recurrent_state.as_ptr(),
        n_rec,
    )
    .with_context(|| format!("snapshot recurrent_state for layer {idx}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Library-level tests that don't require GPU. Tests that exercise
    // the actual GPU copy paths live in
    // `crates/runner/tests/qwen36_moe_linear_state.rs` (skipped when
    // HIP isn't compiled).
    use super::*;

    #[test]
    fn snapshot_count_with_no_layers() {
        let snap = LinearAttnSnapshot { layers: Vec::new() };
        assert_eq!(snap.linear_layer_count(), 0);
    }
}
