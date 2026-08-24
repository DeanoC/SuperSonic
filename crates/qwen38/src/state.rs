use std::ffi::c_void;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use serde::{Deserialize, Serialize};

use crate::config::TextConfig;
use crate::weights::LayerKind;

/// Mutable per-layer state (BF16 KV cache, convolution state, recurrent state).
pub struct LayerState {
    pub kind: LayerKind,
    pub kv_cache_k: Option<GpuBuffer>,
    pub kv_cache_v: Option<GpuBuffer>,
    pub kv_filled: usize,
    pub conv_state: Option<GpuBuffer>,
    pub recurrent_state: Option<GpuBuffer>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelStateDiskSnapshot {
    pub layers: Vec<LayerStateDiskSnapshot>,
}

#[derive(Serialize, Deserialize)]
pub struct LayerStateDiskSnapshot {
    kind: String,
    kv_cache_k: Option<BufferDiskSnapshot>,
    kv_cache_v: Option<BufferDiskSnapshot>,
    kv_filled: usize,
    conv_state: Option<BufferDiskSnapshot>,
    recurrent_state: Option<BufferDiskSnapshot>,
}

#[derive(Serialize, Deserialize)]
struct BufferDiskSnapshot {
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl LayerState {
    pub fn new_linear(ordinal: usize, config: &TextConfig) -> Result<Self, GpuError> {
        let qkv_out_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[qkv_out_dim, config.linear_conv_kernel_dim - 1],
        )?;
        let recurrent_state = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ],
        )?;
        Ok(Self {
            kind: LayerKind::Linear,
            kv_cache_k: None,
            kv_cache_v: None,
            kv_filled: 0,
            conv_state: Some(conv_state),
            recurrent_state: Some(recurrent_state),
        })
    }

    pub fn new_full(_ordinal: usize) -> Self {
        Self {
            kind: LayerKind::Full,
            kv_cache_k: None,
            kv_cache_v: None,
            kv_filled: 0,
            conv_state: None,
            recurrent_state: None,
        }
    }

    /// Ensure BF16 KV cache capacity for `needed` positions.
    pub fn ensure_kv_capacity(
        &mut self,
        needed: usize,
        ordinal: usize,
        config: &TextConfig,
        kv_chunk_size: usize,
    ) -> Result<(), GpuError> {
        let needed = needed + 1;
        if let (Some(ref k), Some(ref v)) = (&self.kv_cache_k, &self.kv_cache_v) {
            let current_cap = k.shape()[2];
            if current_cap >= needed {
                return Ok(());
            }
            let new_cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            self.kv_cache_k = Some(k.grow_seq_dim(2, new_cap)?);
            self.kv_cache_v = Some(v.grow_seq_dim(2, new_cap)?);
        } else {
            let cap = ((needed + kv_chunk_size - 1) / kv_chunk_size) * kv_chunk_size;
            let nkv = config.num_key_value_heads;
            let hd = config.head_dim;
            self.kv_cache_k = Some(GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, nkv, cap, hd],
            )?);
            self.kv_cache_v = Some(GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[1, nkv, cap, hd],
            )?);
        }
        Ok(())
    }

    pub fn set_kv_filled(&mut self, filled: usize) {
        self.kv_filled = filled;
    }

    pub fn kv_capacity(&self) -> usize {
        self.kv_cache_k.as_ref().map(|k| k.shape()[2]).unwrap_or(0)
    }

    pub fn kv_cache_k_ptr(&self) -> Option<*mut c_void> {
        self.kv_cache_k
            .as_ref()
            .map(|buffer| buffer.as_ptr() as *mut c_void)
    }

    pub fn kv_cache_v_ptr(&self) -> Option<*mut c_void> {
        self.kv_cache_v
            .as_ref()
            .map(|buffer| buffer.as_ptr() as *mut c_void)
    }

    pub fn kv_cache_k_offset_ptr(&self, byte_offset: usize) -> Option<*const c_void> {
        self.kv_cache_k
            .as_ref()
            .map(|buffer| buffer.offset_ptr(byte_offset))
    }

    pub fn kv_cache_v_offset_ptr(&self, byte_offset: usize) -> Option<*const c_void> {
        self.kv_cache_v
            .as_ref()
            .map(|buffer| buffer.offset_ptr(byte_offset))
    }

    pub fn to_disk_snapshot(&self) -> Result<LayerStateDiskSnapshot, GpuError> {
        Ok(LayerStateDiskSnapshot {
            kind: match self.kind {
                LayerKind::Linear => "linear".to_string(),
                LayerKind::Full => "full".to_string(),
            },
            kv_cache_k: buffer_to_disk(&self.kv_cache_k)?,
            kv_cache_v: buffer_to_disk(&self.kv_cache_v)?,
            kv_filled: self.kv_filled,
            conv_state: buffer_to_disk(&self.conv_state)?,
            recurrent_state: buffer_to_disk(&self.recurrent_state)?,
        })
    }

    fn from_disk_snapshot(
        snapshot: LayerStateDiskSnapshot,
        config: &TextConfig,
        ordinal: usize,
    ) -> Result<Self, GpuError> {
        let expected_kind = match snapshot.kind.as_str() {
            "linear" => LayerKind::Linear,
            "full" => LayerKind::Full,
            other => {
                return Err(GpuError::InvalidArg(format!(
                    "unknown Qwen disk layer kind {other}"
                )))
            }
        };
        let mut layer = match expected_kind {
            LayerKind::Linear => Self::new_linear(ordinal, config)?,
            LayerKind::Full => Self::new_full(ordinal),
        };
        layer.kv_cache_k = buffer_from_disk(snapshot.kv_cache_k, ordinal)?;
        layer.kv_cache_v = buffer_from_disk(snapshot.kv_cache_v, ordinal)?;
        layer.kv_filled = snapshot.kv_filled;
        layer.conv_state = buffer_from_disk(snapshot.conv_state, ordinal)?;
        layer.recurrent_state = buffer_from_disk(snapshot.recurrent_state, ordinal)?;
        Ok(layer)
    }

    pub fn resident_gpu_bytes(&self) -> usize {
        [
            &self.kv_cache_k,
            &self.kv_cache_v,
            &self.conv_state,
            &self.recurrent_state,
        ]
        .into_iter()
        .flatten()
        .map(GpuBuffer::len_bytes)
        .fold(0usize, usize::saturating_add)
    }

    pub fn clone_gpu(&self) -> Result<Self, GpuError> {
        let clone_opt = |buffer: &Option<GpuBuffer>| -> Result<Option<GpuBuffer>, GpuError> {
            buffer.as_ref().map(GpuBuffer::clone_device).transpose()
        };
        Ok(Self {
            kind: self.kind,
            kv_cache_k: clone_opt(&self.kv_cache_k)?,
            kv_cache_v: clone_opt(&self.kv_cache_v)?,
            kv_filled: self.kv_filled,
            conv_state: clone_opt(&self.conv_state)?,
            recurrent_state: clone_opt(&self.recurrent_state)?,
        })
    }
}

fn buffer_to_disk(buffer: &Option<GpuBuffer>) -> Result<Option<BufferDiskSnapshot>, GpuError> {
    let Some(buffer) = buffer else {
        return Ok(None);
    };
    Ok(Some(BufferDiskSnapshot {
        dtype: dtype_name(buffer.dtype()).to_string(),
        shape: buffer.shape().to_vec(),
        bytes: buffer.to_host_bytes()?,
    }))
}

fn buffer_from_disk(
    snapshot: Option<BufferDiskSnapshot>,
    ordinal: usize,
) -> Result<Option<GpuBuffer>, GpuError> {
    let Some(snapshot) = snapshot else {
        return Ok(None);
    };
    let dtype = ScalarType::from_name(&snapshot.dtype).ok_or_else(|| {
        GpuError::InvalidArg(format!("unknown Qwen disk buffer dtype {}", snapshot.dtype))
    })?;
    Ok(Some(GpuBuffer::from_host_bytes(
        ordinal,
        dtype,
        &snapshot.shape,
        &snapshot.bytes,
    )?))
}

fn dtype_name(dtype: ScalarType) -> &'static str {
    match dtype {
        ScalarType::F16 => "f16",
        ScalarType::BF16 => "bf16",
        ScalarType::F32 => "f32",
        ScalarType::U8 => "u8",
        ScalarType::U32 => "u32",
        ScalarType::I64 => "i64",
        ScalarType::F8E4M3 => "f8_e4m3",
    }
}

/// All mutable state for the model.
pub struct ModelState {
    pub layers: Vec<LayerState>,
    /// KV cache for the optional NextN/MTP full-attention block.
    pub mtp: Option<LayerState>,
}

impl ModelState {
    pub fn new(config: &TextConfig, ordinal: usize) -> Result<Self, GpuError> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            if config.is_full_attention(idx) {
                layers.push(LayerState::new_full(ordinal));
            } else {
                layers.push(LayerState::new_linear(ordinal, config)?);
            }
        }
        Ok(Self {
            layers,
            mtp: (config.mtp_num_hidden_layers > 0).then(|| LayerState::new_full(ordinal)),
        })
    }

    pub fn reset_for_prefill_reuse(&mut self) {
        if let Some(ls) = self.mtp.as_mut() {
            ls.kv_filled = 0;
        }
        for ls in &mut self.layers {
            ls.kv_filled = 0;
            if ls.kv_cache_k.is_none() || ls.kv_cache_v.is_none() {
                ls.kv_cache_k = None;
                ls.kv_cache_v = None;
            }
        }
    }

    pub fn clone_gpu(&self) -> Result<Self, GpuError> {
        let layers = self
            .layers
            .iter()
            .map(LayerState::clone_gpu)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            mtp: self.mtp.as_ref().map(LayerState::clone_gpu).transpose()?,
        })
    }

    pub fn resident_gpu_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(LayerState::resident_gpu_bytes)
            .chain(self.mtp.iter().map(LayerState::resident_gpu_bytes))
            .fold(0usize, usize::saturating_add)
    }

    pub fn to_disk_snapshot(&self) -> Result<ModelStateDiskSnapshot, GpuError> {
        Ok(ModelStateDiskSnapshot {
            layers: self
                .layers
                .iter()
                .map(LayerState::to_disk_snapshot)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    pub fn from_disk_snapshot(
        snapshot: ModelStateDiskSnapshot,
        config: &TextConfig,
        ordinal: usize,
    ) -> Result<Self, GpuError> {
        if snapshot.layers.len() != config.num_hidden_layers {
            return Err(GpuError::InvalidArg(format!(
                "Qwen disk snapshot layer count {} != config {}",
                snapshot.layers.len(),
                config.num_hidden_layers
            )));
        }
        let mut layers = Vec::with_capacity(snapshot.layers.len());
        for (idx, layer_snapshot) in snapshot.layers.into_iter().enumerate() {
            let expected = if config.is_full_attention(idx) {
                "full"
            } else {
                "linear"
            };
            if layer_snapshot.kind != expected {
                return Err(GpuError::InvalidArg(format!(
                    "Qwen disk snapshot layer {idx} kind {} != expected {expected}",
                    layer_snapshot.kind
                )));
            }
            layers.push(LayerState::from_disk_snapshot(
                layer_snapshot,
                config,
                ordinal,
            )?);
        }
        Ok(Self {
            layers,
            mtp: (config.mtp_num_hidden_layers > 0).then(|| LayerState::new_full(ordinal)),
        })
    }

    pub fn snapshot_linear(&self) -> Result<LinearStateSnapshot, GpuError> {
        let mut per_layer = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            match (ls.kind, &ls.conv_state, &ls.recurrent_state) {
                (LayerKind::Linear, Some(conv), Some(rec)) => {
                    per_layer.push(Some((conv.clone_device()?, rec.clone_device()?)));
                }
                _ => per_layer.push(None),
            }
        }
        Ok(LinearStateSnapshot { per_layer })
    }

    pub fn snapshot_linear_into(
        &self,
        snap: &mut LinearStateSnapshot,
        ordinal: usize,
    ) -> Result<(), GpuError> {
        if snap.per_layer.len() != self.layers.len() {
            *snap = self.snapshot_linear()?;
            return Ok(());
        }
        for (i, ls) in self.layers.iter().enumerate() {
            match (
                ls.kind,
                ls.conv_state.as_ref(),
                ls.recurrent_state.as_ref(),
                snap.per_layer[i].as_mut(),
            ) {
                (LayerKind::Linear, Some(conv), Some(rec), Some((conv_dst, rec_dst))) => {
                    if conv_dst.len_bytes() != conv.len_bytes()
                        || rec_dst.len_bytes() != rec.len_bytes()
                    {
                        *snap = self.snapshot_linear()?;
                        return Ok(());
                    }
                    gpu_hal::copy_d2d(
                        ordinal,
                        conv_dst.as_mut_ptr(),
                        conv.as_ptr(),
                        conv.len_bytes(),
                    )?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        rec_dst.as_mut_ptr(),
                        rec.as_ptr(),
                        rec.len_bytes(),
                    )?;
                }
                (LayerKind::Full, _, _, None) => {}
                _ => {
                    *snap = self.snapshot_linear()?;
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub fn restore_linear(
        &mut self,
        snap: &LinearStateSnapshot,
        ordinal: usize,
    ) -> Result<(), GpuError> {
        if snap.per_layer.len() != self.layers.len() {
            return Err(GpuError::InvalidArg(format!(
                "restore_linear: snapshot has {} layers, state has {}",
                snap.per_layer.len(),
                self.layers.len(),
            )));
        }
        for (i, ls) in self.layers.iter_mut().enumerate() {
            match (ls.kind, &snap.per_layer[i]) {
                (LayerKind::Linear, Some((conv_src, rec_src))) => {
                    let conv_dst = ls.conv_state.as_mut().ok_or_else(|| {
                        GpuError::InvalidArg(format!(
                            "restore_linear: layer {i} missing conv_state"
                        ))
                    })?;
                    let rec_dst = ls.recurrent_state.as_mut().ok_or_else(|| {
                        GpuError::InvalidArg(format!(
                            "restore_linear: layer {i} missing recurrent_state"
                        ))
                    })?;
                    if conv_dst.len_bytes() != conv_src.len_bytes()
                        || rec_dst.len_bytes() != rec_src.len_bytes()
                    {
                        return Err(GpuError::InvalidArg(format!(
                            "restore_linear: layer {i} size mismatch (conv dst={} src={}, rec dst={} src={})",
                            conv_dst.len_bytes(),
                            conv_src.len_bytes(),
                            rec_dst.len_bytes(),
                            rec_src.len_bytes(),
                        )));
                    }
                    gpu_hal::copy_d2d(
                        ordinal,
                        conv_dst.as_mut_ptr(),
                        conv_src.as_ptr(),
                        conv_src.len_bytes(),
                    )?;
                    gpu_hal::copy_d2d(
                        ordinal,
                        rec_dst.as_mut_ptr(),
                        rec_src.as_ptr(),
                        rec_src.len_bytes(),
                    )?;
                }
                (LayerKind::Full, None) => {}
                (LayerKind::Linear, None) => {
                    return Err(GpuError::InvalidArg(format!(
                        "restore_linear: layer {i} is Linear but snapshot slot is None"
                    )));
                }
                (LayerKind::Full, Some(_)) => {
                    return Err(GpuError::InvalidArg(format!(
                        "restore_linear: layer {i} is Full but snapshot slot is Some"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Sidecar holding `(conv_state, recurrent_state)` for every linear-attention
/// layer at some earlier logical position.
pub struct LinearStateSnapshot {
    pub per_layer: Vec<Option<(GpuBuffer, GpuBuffer)>>,
}
