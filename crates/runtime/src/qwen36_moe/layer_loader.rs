use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{
    current_backend, Backend, GpuBuffer, ScalarType, VirtualAllocationRole, VirtualArena,
};
use model_store::manifest::LayoutTag;
use model_store::store::{Int4StorageKind, Int4StorageView};
use model_store::{BakedStore, TensorStorageExtent, VirtualArenaTransferBackend};
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe::decode::Qwen36DiagnosticObserver;
use crate::qwen36_moe::layers::LoadedQwen36Layers;
use crate::qwen36_moe::residency::{
    MoeExpertProjection, MoeExpertResidencyConfig, MoeExpertResidencyManager,
};
use crate::qwen36_moe::types::{
    AttnLayerBuffers, FfnInt4Sidecars, FfnLayerBuffers, FullAttnInt4Sidecars, FullAttnKvCache,
    LayerBuffers, LinearAttnInt4Sidecars, LoadedInt4Sidecar, MultiLayerGeom, ResidentWeight,
};

#[cfg(test)]
pub(crate) static GPU_BACKEND_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Open a BakedStore from the bake dir, loading one tensor by name to a
/// fresh GpuBuffer. The wrapper exists to attach a useful context message
/// when a tensor is missing (the bake-validation in `inspect_bake` already
/// runs as part of the dry-run, so a missing tensor here is a real bug).
pub fn load_to_gpu(store: &BakedStore, ordinal: usize, name: &str) -> Result<GpuBuffer> {
    load_to_gpu_with_options(store, ordinal, name, &Qwen36LoadOptions::default())
}

#[derive(Clone)]
pub struct Qwen36LoadOptions {
    pub registered_mmap_upload: bool,
    pub kv_fp8_sidecar: qwen35::state::KvFp8SidecarOptions,
    diagnostic_observer: Option<Arc<Qwen36DiagnosticObserver>>,
}

impl Default for Qwen36LoadOptions {
    fn default() -> Self {
        Self {
            registered_mmap_upload: false,
            kv_fp8_sidecar: qwen35::state::KvFp8SidecarOptions::default(),
            diagnostic_observer: None,
        }
    }
}

impl Qwen36LoadOptions {
    pub fn with_registered_mmap_upload(mut self, enabled: bool) -> Self {
        self.registered_mmap_upload = enabled;
        self
    }

    pub fn with_kv_fp8_sidecar(
        mut self,
        kv_fp8_sidecar: qwen35::state::KvFp8SidecarOptions,
    ) -> Self {
        self.kv_fp8_sidecar = kv_fp8_sidecar;
        self
    }

    pub fn with_diagnostic_observer(mut self, observer: Arc<Qwen36DiagnosticObserver>) -> Self {
        self.diagnostic_observer = Some(observer);
        self
    }

    fn emit_diagnostic(&self, message: &str) {
        if let Some(observer) = self.diagnostic_observer.as_ref() {
            observer(message);
        }
    }
}

fn load_to_gpu_with_options(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
    options: &Qwen36LoadOptions,
) -> Result<GpuBuffer> {
    let resolved = resolve_qwen36_store_name(store, name);
    if options.registered_mmap_upload && current_backend() == Backend::Hip {
        if let Ok(Some(extent)) = store.tensor_storage_extent(resolved.as_ref()) {
            if should_use_registered_mmap_upload_for_extent(&extent) {
                match store.load_to_gpu_registered_mmap(resolved.as_ref(), ordinal) {
                    Ok(buffer) => return Ok(buffer),
                    Err(err) => {
                        options.emit_diagnostic(&format!(
                            "[flm] registered mmap upload failed for {}; falling back to pageable upload: {err}",
                            resolved.as_ref()
                        ));
                    }
                }
            }
        }
    }
    store
        .load_to_gpu(resolved.as_ref(), ordinal)
        .with_context(|| format!("BakedStore::load_to_gpu({name})"))
}

fn load_to_resident_weight(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
) -> Result<ResidentWeight> {
    load_to_gpu(store, ordinal, name).map(ResidentWeight::Dense)
}

fn load_to_virtual_resident_weight(
    store: &BakedStore,
    arena: &mut VirtualArena,
    name: &str,
    transfer_backend: VirtualArenaTransferBackend,
) -> Result<ResidentWeight> {
    let resolved = resolve_qwen36_store_name(store, name);
    let id = store
        .load_to_virtual_arena_with_backend(
            arena,
            resolved.as_ref(),
            VirtualAllocationRole::MoeExpert,
            transfer_backend,
        )
        .with_context(|| format!("BakedStore::load_to_virtual_arena({name})"))?;
    let allocation = arena
        .allocation(id)
        .ok_or_else(|| anyhow!("virtual allocation id {id} disappeared after loading {name}"))?;
    let buffer = allocation.buffer();
    Ok(ResidentWeight::Virtual {
        allocation_id: id,
        ptr: buffer.as_ptr(),
        dtype: buffer.dtype(),
        shape: buffer.shape().to_vec(),
        len_bytes: buffer.len_bytes(),
    })
}

fn load_to_sparse_resident_weight(
    store: &BakedStore,
    manager: &mut MoeExpertResidencyManager,
    layer_idx: usize,
    projection: MoeExpertProjection,
    name: &str,
    expert_count: usize,
) -> Result<ResidentWeight> {
    let resolved = resolve_qwen36_store_name(store, name);
    manager
        .register_tensor(
            store,
            layer_idx,
            projection,
            resolved.as_ref(),
            expert_count,
        )
        .with_context(|| format!("reserve sparse MoE expert tensor {name}"))?;
    manager
        .resident_weight(layer_idx, projection)
        .with_context(|| format!("build sparse resident weight for {name}"))
}

pub fn store_contains_qwen36(store: &BakedStore, name: &str) -> bool {
    store.contains(resolve_qwen36_store_name(store, name).as_ref())
}

pub fn store_layout_qwen36<'a>(store: &'a BakedStore, name: &str) -> Option<&'a LayoutTag> {
    store.layout(resolve_qwen36_store_name(store, name).as_ref())
}

const REGISTERED_MMAP_UPLOAD_MIN_BYTES: u64 = 256 * 1024 * 1024;

fn should_use_registered_mmap_upload_for_extent(extent: &TensorStorageExtent) -> bool {
    extent.storage_dtype == "u8"
        && extent.upload_dtype == "u8"
        && extent.byte_len >= REGISTERED_MMAP_UPLOAD_MIN_BYTES
        && matches!(extent.layout, LayoutTag::Int4Quantized)
        && extent.name.ends_with(".mlp.experts.gate_up_proj")
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_store::{TensorStorageExtent, TensorStorageSourceKind};
    use std::path::PathBuf;

    fn test_extent(
        name: &str,
        byte_len: u64,
        storage_dtype: &str,
        layout: LayoutTag,
    ) -> TensorStorageExtent {
        TensorStorageExtent {
            source_kind: TensorStorageSourceKind::FlmContainer,
            source_path: PathBuf::from("/tmp/model.flm"),
            name: name.to_string(),
            file_offset: 4096,
            byte_len,
            storage_dtype: storage_dtype.to_string(),
            storage_shape: vec![128, 8192, 256],
            layout,
            upload_dtype: storage_dtype.to_string(),
            upload_shape: vec![128, 8192, 256],
        }
    }

    #[test]
    fn registered_mmap_upload_policy_targets_native_int4_gate_up_slabs() {
        assert!(should_use_registered_mmap_upload_for_extent(&test_extent(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            268_435_456,
            "u8",
            LayoutTag::Int4Quantized,
        )));
        assert!(!should_use_registered_mmap_upload_for_extent(&test_extent(
            "model.language_model.layers.0.mlp.experts.down_proj",
            268_435_456,
            "u8",
            LayoutTag::Int4Quantized,
        )));
        assert!(!should_use_registered_mmap_upload_for_extent(&test_extent(
            "lm_head.weight",
            1_017_118_720,
            "bf16",
            LayoutTag::Raw,
        )));
        assert!(!should_use_registered_mmap_upload_for_extent(&test_extent(
            "mtp.layers.0.mlp.experts.gate_up_proj",
            524_288,
            "u8",
            LayoutTag::Int4Quantized,
        )));
    }

    #[test]
    fn registered_mmap_upload_policy_requires_native_int4_extent() {
        assert!(!should_use_registered_mmap_upload_for_extent(&test_extent(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            268_435_456,
            "u8",
            LayoutTag::Raw,
        )));
        assert!(should_use_registered_mmap_upload_for_extent(&test_extent(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            268_435_456,
            "u8",
            LayoutTag::Int4Quantized,
        )));
    }

    #[test]
    fn persistent_decode_support_is_limited_to_native_kernel_encodings() {
        assert!(!Qwen36WeightMode::Bf16.supports_persistent_decode());
        assert!(Qwen36WeightMode::Int4.supports_persistent_decode());
        assert!(!Qwen36WeightMode::Q4Km.supports_persistent_decode());
        assert!(Qwen36WeightMode::Fp8.supports_persistent_decode());
    }
}

pub fn resolve_qwen36_store_name<'a>(store: &BakedStore, name: &'a str) -> Cow<'a, str> {
    if store.contains(name) {
        return Cow::Borrowed(name);
    }
    if name.contains(".mlp.experts.") {
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let alt = format!("model.{rest}");
            if store.contains(&alt) {
                return Cow::Owned(alt);
            }
        }
    }
    Cow::Borrowed(name)
}

/// Pinned by `oracle/bake_int4.py` and the kernel — every INT4 tensor in
/// the bake quantizes at this group size. Detected per-tensor via a
/// `*_int4_scale` sidecar; if any quantizable tensor is present and
/// uses a different group_size we'd surface that as an error.
pub const QWEN36_MOE_INT4_GROUP_SIZE: i32 = 128;
const QWEN36_MOE_FP8_BLOCK_SIZE: i32 = 128;
pub const QWEN36_MOE_LOWBIT_NATIVE_INT4: i32 = 4;
pub const QWEN36_MOE_LOWBIT_GGML_Q4_K: i32 = 12;
pub const QWEN36_MOE_LOWBIT_GGML_Q5_K: i32 = 13;
pub const QWEN36_MOE_LOWBIT_GGML_Q6_K: i32 = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36WeightMode {
    Bf16,
    Int4,
    Q4Km,
    Fp8,
}

#[derive(Debug, Clone, Copy)]
pub struct SparseExpertLoadOptions {
    pub cap_experts: usize,
    pub protected_experts: Option<usize>,
    pub fixed_hot_experts: Option<usize>,
    pub async_prefetch: bool,
    pub async_staging_pages: usize,
    pub prefetch_evict: bool,
    pub transfer_backend: VirtualArenaTransferBackend,
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen36LayerLoadStrategy {
    Dense,
    VirtualExperts {
        transfer_backend: VirtualArenaTransferBackend,
    },
    SparseExperts(SparseExpertLoadOptions),
}

impl Qwen36WeightMode {
    pub fn is_int4(self) -> bool {
        matches!(self, Self::Int4 | Self::Q4Km)
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Int4 => "INT4 GPTQ",
            Self::Q4Km => "Q4_K_M GGML",
            Self::Fp8 => "FP8 native",
        }
    }

    pub fn supports_persistent_decode(self) -> bool {
        matches!(self, Self::Int4 | Self::Fp8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36LayerWeightEncoding {
    Bf16,
    NativeInt4,
    GgmlKBlock,
    Fp8,
}

fn classify_projection_set(
    group_size: i32,
    projection_types: &[i32],
) -> Result<Qwen36LayerWeightEncoding> {
    if group_size < 0 {
        return Ok(Qwen36LayerWeightEncoding::Fp8);
    }
    if projection_types
        .iter()
        .all(|&projection_type| projection_type == QWEN36_MOE_LOWBIT_NATIVE_INT4)
    {
        if matches!(group_size, 32 | QWEN36_MOE_INT4_GROUP_SIZE) {
            Ok(Qwen36LayerWeightEncoding::NativeInt4)
        } else {
            Err(anyhow!(
                "unsupported Qwen3.6 native INT4 group size {group_size}; expected 32 or {}",
                QWEN36_MOE_INT4_GROUP_SIZE
            ))
        }
    } else if projection_types.iter().all(|&projection_type| {
        matches!(
            projection_type,
            QWEN36_MOE_LOWBIT_GGML_Q4_K | QWEN36_MOE_LOWBIT_GGML_Q5_K | QWEN36_MOE_LOWBIT_GGML_Q6_K
        )
    }) {
        if group_size == QWEN36_MOE_INT4_GROUP_SIZE {
            Ok(Qwen36LayerWeightEncoding::GgmlKBlock)
        } else {
            Err(anyhow!(
                "unsupported Qwen3.6 GGML K-block group size {group_size}; expected {}",
                QWEN36_MOE_INT4_GROUP_SIZE
            ))
        }
    } else {
        Err(anyhow!(
            "mixed or unsupported Qwen3.6 projection types {projection_types:?}"
        ))
    }
}

pub fn classify_layer_weight_encoding(
    layers: &[LayerBuffers],
) -> Result<Qwen36LayerWeightEncoding> {
    let mut stack_encoding = None;
    for (layer_idx, layer) in layers.iter().enumerate() {
        let attn_encoding = match &layer.attn {
            AttnLayerBuffers::Full { int4: None, .. }
            | AttnLayerBuffers::Linear { int4: None, .. } => Qwen36LayerWeightEncoding::Bf16,
            AttnLayerBuffers::Full { int4: Some(s), .. } => classify_projection_set(
                s.group_size,
                &[s.q_proj_type, s.k_proj_type, s.v_proj_type, s.o_proj_type],
            )?,
            AttnLayerBuffers::Linear { int4: Some(s), .. } => classify_projection_set(
                s.group_size,
                &[s.in_proj_qkv_type, s.in_proj_z_type, s.out_proj_type],
            )?,
        };
        let ffn_encoding = match &layer.ffn.int4 {
            None => Qwen36LayerWeightEncoding::Bf16,
            Some(s) => classify_projection_set(
                s.group_size,
                &[
                    s.gate_up_proj_type,
                    s.down_proj_type,
                    s.shared_gate_proj_type,
                    s.shared_up_proj_type,
                    s.shared_down_proj_type,
                ],
            )?,
        };
        if attn_encoding != ffn_encoding {
            return Err(anyhow!(
                "Qwen3.6 layer {layer_idx} mixes {attn_encoding:?} attention with \
                 {ffn_encoding:?} FFN weights"
            ));
        }
        if let Some(expected) = stack_encoding {
            if expected != attn_encoding {
                return Err(anyhow!(
                    "Qwen3.6 layer {layer_idx} uses {attn_encoding:?}, but earlier layers use \
                     {expected:?}"
                ));
            }
        } else {
            stack_encoding = Some(attn_encoding);
        }
    }
    Ok(stack_encoding.unwrap_or(Qwen36LayerWeightEncoding::Bf16))
}

pub(crate) fn ensure_legacy_int4_execution_supported(
    layers: &[LayerBuffers],
    execution_path: &str,
) -> Result<()> {
    let validate_row_group =
        |layer_idx: usize, projection: &str, sidecar: &LoadedInt4Sidecar| -> Result<()> {
            if sidecar.view.kind == Int4StorageKind::RowGroupSymmetric {
                if !matches!(execution_path, "chained decode" | "persistent decode") {
                    return Err(anyhow!(
                        "Qwen3.6 {execution_path} does not advertise descriptor-driven row-group \
                         INT4 execution (layer {layer_idx} {projection})"
                    ));
                }
                let desc = crate::qwen36_moe::persistent_decode::build_int4_weight_desc(sidecar)
                    .with_context(|| {
                        format!("Qwen3.6 {execution_path} layer {layer_idx} {projection}")
                    })?;
                if desc.encoding != 2 || desc.zero != std::ptr::null() {
                    return Err(anyhow!(
                        "Qwen3.6 {execution_path} layer {layer_idx} {projection} did not build \
                         the encoding-2 null-zero descriptor"
                    ));
                }
            }
            Ok(())
        };

    for (layer_idx, layer) in layers.iter().enumerate() {
        match &layer.attn {
            AttnLayerBuffers::Full { int4: Some(s), .. } => {
                for (projection, sidecar) in [
                    ("q_proj", &s.q_proj),
                    ("k_proj", &s.k_proj),
                    ("v_proj", &s.v_proj),
                    ("o_proj", &s.o_proj),
                ] {
                    validate_row_group(layer_idx, projection, sidecar)?;
                }
            }
            AttnLayerBuffers::Linear { int4: Some(s), .. } => {
                for (projection, sidecar) in [
                    ("linear_in_proj_qkv", &s.in_proj_qkv),
                    ("linear_in_proj_z", &s.in_proj_z),
                    ("linear_out_proj", &s.out_proj),
                ] {
                    validate_row_group(layer_idx, projection, sidecar)?;
                }
            }
            AttnLayerBuffers::Full { int4: None, .. }
            | AttnLayerBuffers::Linear { int4: None, .. } => {}
        }

        if let Some(s) = &layer.ffn.int4 {
            for (projection, sidecar) in [
                ("experts_gate_up", &s.gate_up_proj),
                ("experts_down", &s.down_proj),
                ("shared_expert_gate_proj", &s.shared_gate_proj),
                ("shared_expert_up_proj", &s.shared_up_proj),
                ("shared_expert_down_proj", &s.shared_down_proj),
            ] {
                validate_row_group(layer_idx, projection, sidecar)?;
            }
        }
    }
    Ok(())
}

fn ggml_lowbit_type_for_layout(store: &BakedStore, name: &str) -> Result<i32> {
    let layout = store_layout_qwen36(store, name)
        .ok_or_else(|| anyhow!("missing layout metadata for {name}"))?;
    match layout {
        LayoutTag::GgmlQ4K => Ok(QWEN36_MOE_LOWBIT_GGML_Q4_K),
        LayoutTag::GgmlQ5K => Ok(QWEN36_MOE_LOWBIT_GGML_Q5_K),
        LayoutTag::GgmlQ6K => Ok(QWEN36_MOE_LOWBIT_GGML_Q6_K),
        other => Err(anyhow!(
            "{name}: expected GGML K-block layout for raw q4km projection, got {other:?}"
        )),
    }
}

fn ignored_lowbit_sidecar(ordinal: usize) -> Result<GpuBuffer> {
    GpuBuffer::zeros(ordinal, ScalarType::U8, &[1]).context("alloc ignored lowbit sidecar")
}

fn load_int4_sidecar_from_view(
    store: &BakedStore,
    ordinal: usize,
    view: &Int4StorageView,
) -> Result<LoadedInt4Sidecar> {
    let scale = load_to_gpu(store, ordinal, &view.scale_tensor)?;
    let zero = view
        .zero_tensor
        .as_deref()
        .map(|name| load_to_gpu(store, ordinal, name))
        .transpose()?;
    Ok(LoadedInt4Sidecar {
        scale,
        zero,
        view: view.clone(),
    })
}

fn projection_shape(store: &BakedStore, name: &str) -> Result<Vec<usize>> {
    let resolved = resolve_qwen36_store_name(store, name);
    store
        .shape(resolved.as_ref())
        .map(<[usize]>::to_vec)
        .ok_or_else(|| anyhow!("missing projection shape for {name}"))
}

fn checked_projection_geometry(shape: &[usize], name: &str) -> Result<(usize, usize)> {
    if !matches!(shape, [_, _] | [_, _, _]) {
        return Err(anyhow!(
            "{name}: quantized projection must have rank 2 or 3, got {shape:?}"
        ));
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows == 0 || cols == 0 {
        return Err(anyhow!(
            "{name}: quantized projection dimensions must be nonzero, got {shape:?}"
        ));
    }
    Ok((rows, cols))
}

fn explicit_tile_v1_view(
    store: &BakedStore,
    name: &str,
    scale_name: &str,
    zero_name: &str,
) -> Result<Int4StorageView> {
    let packed_shape = projection_shape(store, name)?;
    let (rows, packed_cols) = checked_projection_geometry(&packed_shape, name)?;
    let cols = packed_cols
        .checked_mul(2)
        .ok_or_else(|| anyhow!("{name}: logical INT4 column count overflows"))?;
    if rows % QWEN36_MOE_INT4_GROUP_SIZE as usize != 0
        || cols % QWEN36_MOE_INT4_GROUP_SIZE as usize != 0
    {
        return Err(anyhow!(
            "{name}: tile-v1 shape [{rows}, {cols}] is not divisible by {}",
            QWEN36_MOE_INT4_GROUP_SIZE
        ));
    }
    let packed_expert_stride_bytes = rows
        .checked_mul(packed_cols)
        .ok_or_else(|| anyhow!("{name}: packed expert stride overflows"))?;
    let scale_row_stride_elements = cols / QWEN36_MOE_INT4_GROUP_SIZE as usize;
    let scale_expert_stride_elements = (rows / QWEN36_MOE_INT4_GROUP_SIZE as usize)
        .checked_mul(scale_row_stride_elements)
        .ok_or_else(|| anyhow!("{name}: scale expert stride overflows"))?;
    let mut logical_shape = packed_shape;
    *logical_shape.last_mut().expect("checked projection rank") = cols;
    let has_expert_axis = shape_is_rank3(&logical_shape);
    Ok(Int4StorageView {
        kind: Int4StorageKind::TileV1,
        group_size: QWEN36_MOE_INT4_GROUP_SIZE as usize,
        packed_tensor: resolve_qwen36_store_name(store, name).into_owned(),
        scale_tensor: resolve_qwen36_store_name(store, scale_name).into_owned(),
        zero_tensor: Some(resolve_qwen36_store_name(store, zero_name).into_owned()),
        logical_shape,
        packed_row_stride_bytes: packed_cols,
        packed_expert_stride_bytes: has_expert_axis
            .then_some(packed_expert_stride_bytes)
            .unwrap_or(0),
        scale_row_stride_elements,
        scale_expert_stride_elements: has_expert_axis
            .then_some(scale_expert_stride_elements)
            .unwrap_or(0),
        output_group_size: QWEN36_MOE_INT4_GROUP_SIZE as usize,
        implicit_zero_code: None,
    })
}

fn shape_is_rank3(shape: &[usize]) -> bool {
    shape.len() == 3
}

fn explicit_fp8_view(store: &BakedStore, name: &str, scale_name: &str) -> Result<Int4StorageView> {
    let logical_shape = projection_shape(store, name)?;
    let (rows, cols) = checked_projection_geometry(&logical_shape, name)?;
    let group_size = QWEN36_MOE_FP8_BLOCK_SIZE as usize;
    if rows % group_size != 0 || cols % group_size != 0 {
        return Err(anyhow!(
            "{name}: FP8 block shape [{rows}, {cols}] is not divisible by {group_size}"
        ));
    }
    let packed_expert_stride_bytes = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("{name}: FP8 expert stride overflows"))?;
    let scale_row_stride_elements = cols / group_size;
    let scale_expert_stride_elements =
        (rows / group_size)
            .checked_mul(scale_row_stride_elements)
            .ok_or_else(|| anyhow!("{name}: FP8 scale expert stride overflows"))?;
    Ok(Int4StorageView {
        kind: Int4StorageKind::TileV1,
        group_size,
        packed_tensor: resolve_qwen36_store_name(store, name).into_owned(),
        scale_tensor: resolve_qwen36_store_name(store, scale_name).into_owned(),
        zero_tensor: None,
        logical_shape: logical_shape.clone(),
        packed_row_stride_bytes: cols,
        packed_expert_stride_bytes: shape_is_rank3(&logical_shape)
            .then_some(packed_expert_stride_bytes)
            .unwrap_or(0),
        scale_row_stride_elements,
        scale_expert_stride_elements: shape_is_rank3(&logical_shape)
            .then_some(scale_expert_stride_elements)
            .unwrap_or(0),
        output_group_size: group_size,
        implicit_zero_code: None,
    })
}

fn load_int4_projection_sidecar(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
    scale_name: &str,
    zero_name: &str,
) -> Result<LoadedInt4Sidecar> {
    let resolved = resolve_qwen36_store_name(store, name);
    let view = match store.int4_storage_view(resolved.as_ref()) {
        Some(view) => view.clone(),
        None => explicit_tile_v1_view(store, name, scale_name, zero_name)?,
    };
    load_int4_sidecar_from_view(store, ordinal, &view)
}

fn load_fp8_projection_sidecar(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
    scale_name: &str,
) -> Result<LoadedInt4Sidecar> {
    let view = explicit_fp8_view(store, name, scale_name)?;
    load_int4_sidecar_from_view(store, ordinal, &view)
}

fn load_q4km_projection_sidecar(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
    scale_name: &str,
    zero_name: &str,
) -> Result<(i32, LoadedInt4Sidecar)> {
    let layout = store_layout_qwen36(store, name)
        .ok_or_else(|| anyhow!("missing layout metadata for {name}"))?;
    match layout {
        LayoutTag::GgmlQ4K | LayoutTag::GgmlQ5K | LayoutTag::GgmlQ6K => {
            let shape = projection_shape(store, name)?;
            Ok((
                ggml_lowbit_type_for_layout(store, name)?,
                LoadedInt4Sidecar {
                    scale: ignored_lowbit_sidecar(ordinal)?,
                    zero: Some(ignored_lowbit_sidecar(ordinal)?),
                    view: Int4StorageView {
                        kind: Int4StorageKind::TileV1,
                        group_size: QWEN36_MOE_INT4_GROUP_SIZE as usize,
                        packed_tensor: name.to_string(),
                        scale_tensor: scale_name.to_string(),
                        zero_tensor: Some(zero_name.to_string()),
                        logical_shape: shape,
                        packed_row_stride_bytes: 0,
                        packed_expert_stride_bytes: 0,
                        scale_row_stride_elements: 0,
                        scale_expert_stride_elements: 0,
                        output_group_size: QWEN36_MOE_INT4_GROUP_SIZE as usize,
                        implicit_zero_code: None,
                    },
                },
            ))
        }
        LayoutTag::Int4Quantized | LayoutTag::Int4RowGroup => Ok((
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
            load_int4_projection_sidecar(store, ordinal, name, scale_name, zero_name)?,
        )),
        other => Err(anyhow!(
            "{name}: expected raw GGML K-block or native INT4 sidecar layout for q4km projection, got {other:?}"
        )),
    }
}

/// Build one layer's worth of GPU-resident weight + state buffers from a
/// BakedStore. Decides full-attn vs linear-attn by consulting the config's
/// `layer_types` (every 4th layer is full per the standard hybrid pattern).
/// `weight_mode` controls whether the weight tensors are loaded as BF16,
/// packed INT4 nibbles + sidecars, or native FP8 bytes + scale sidecars. The
/// INT4 bake naming convention pairs
/// `<name>.weight` (packed) with `<name>.weight_int4_scale`/`_int4_zero`
/// for dense projections, and `<name>` (packed) with `<name>_int4_scale`/
/// `_int4_zero` for fused-expert tensors (no `.weight` suffix in the
/// HuggingFace checkpoint). FP8 uses the same weight names plus
/// `<name>_scale_inv`.
pub fn load_layer_buffers(
    store: &BakedStore,
    ordinal: usize,
    layer_idx: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    // When > 0, allocate a KV cache for full-attention layers sized for
    // `kv_max_t` past tokens. Linear-attn layers use `conv_state` +
    // `recurrent_state` instead (always allocated). 0 = no KV cache,
    // kernel falls back to kv_len=1 (back-compat for the parity test).
    kv_max_t: usize,
    // When true, the layer's KV cache is allocated as FP8 E4M3 bytes
    // with F32 per-(head, position) scales and an optional BF16
    // sidecar (gated by `qwen35::state::kv_fp8_bf16_sidecar_*` env
    // helpers).
    kv_fp8: bool,
    // When true, full-attention K/V caches use VMM reservations for
    // their main K/V storage. Applies to BF16 and FP8 KV; FP8 scale
    // and sidecar buffers stay dense.
    kv_vmm: bool,
    mut expert_arena: Option<&mut VirtualArena>,
    mut expert_residency: Option<&mut MoeExpertResidencyManager>,
    expert_virtual_transfer_backend: VirtualArenaTransferBackend,
    load_options: &Qwen36LoadOptions,
) -> Result<LayerBuffers> {
    let lp = format!("{weight_prefix}.layers.{layer_idx}");

    let attn = if text_config.is_full_attention(layer_idx) {
        let fa = format!("{lp}.self_attn");
        let (q_proj_type, k_proj_type, v_proj_type, o_proj_type) = (
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
        );
        let int4 = match weight_mode {
            Qwen36WeightMode::Int4 => Some(FullAttnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                q_proj_type,
                q_proj: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.q_proj.weight"),
                    &format!("{fa}.q_proj.weight_int4_scale"),
                    &format!("{fa}.q_proj.weight_int4_zero"),
                )?,
                k_proj_type,
                k_proj: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.k_proj.weight"),
                    &format!("{fa}.k_proj.weight_int4_scale"),
                    &format!("{fa}.k_proj.weight_int4_zero"),
                )?,
                v_proj_type,
                v_proj: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.v_proj.weight"),
                    &format!("{fa}.v_proj.weight_int4_scale"),
                    &format!("{fa}.v_proj.weight_int4_zero"),
                )?,
                o_proj_type,
                o_proj: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.o_proj.weight"),
                    &format!("{fa}.o_proj.weight_int4_scale"),
                    &format!("{fa}.o_proj.weight_int4_zero"),
                )?,
            }),
            Qwen36WeightMode::Q4Km => {
                let (q_proj_type, q_proj) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.q_proj.weight"),
                    &format!("{fa}.q_proj.weight_int4_scale"),
                    &format!("{fa}.q_proj.weight_int4_zero"),
                )?;
                let (k_proj_type, k_proj) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.k_proj.weight"),
                    &format!("{fa}.k_proj.weight_int4_scale"),
                    &format!("{fa}.k_proj.weight_int4_zero"),
                )?;
                let (v_proj_type, v_proj) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.v_proj.weight"),
                    &format!("{fa}.v_proj.weight_int4_scale"),
                    &format!("{fa}.v_proj.weight_int4_zero"),
                )?;
                let (o_proj_type, o_proj) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.o_proj.weight"),
                    &format!("{fa}.o_proj.weight_int4_scale"),
                    &format!("{fa}.o_proj.weight_int4_zero"),
                )?;
                Some(FullAttnInt4Sidecars {
                    group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                    q_proj_type,
                    q_proj,
                    k_proj_type,
                    k_proj,
                    v_proj_type,
                    v_proj,
                    o_proj_type,
                    o_proj,
                })
            }
            Qwen36WeightMode::Fp8 => Some(FullAttnInt4Sidecars {
                group_size: -QWEN36_MOE_FP8_BLOCK_SIZE,
                q_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                q_proj: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.q_proj.weight"),
                    &format!("{fa}.q_proj.weight_scale_inv"),
                )?,
                k_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                k_proj: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.k_proj.weight"),
                    &format!("{fa}.k_proj.weight_scale_inv"),
                )?,
                v_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                v_proj: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.v_proj.weight"),
                    &format!("{fa}.v_proj.weight_scale_inv"),
                )?,
                o_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                o_proj: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{fa}.o_proj.weight"),
                    &format!("{fa}.o_proj.weight_scale_inv"),
                )?,
            }),
            Qwen36WeightMode::Bf16 => None,
        };
        // KV cache: allocate per-layer when multi-token decode is requested.
        // BF16 path: [kv_max_t, num_kv_heads * head_dim] BF16 for K and V.
        // FP8 path: same shape but U8 (FP8 E4M3 bytes) plus F32 scales
        // [num_kv_heads, kv_max_t] for K and V, plus optional BF16 sidecar
        // [num_kv_heads, sidecar_window, head_dim] when enabled.
        // VMM path (default on HIP when supported, or
        // SUPERSONIC_VMM_KV=1): K/V are VirtualBuffer reservations with a
        // mapped prefix; dense k/v are None. FP8 scale and sidecar buffers
        // remain regular dense GpuBuffers.
        let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
        let kv_cache = if kv_max_t > 0 {
            let num_kv_heads = geom.num_kv_heads as usize;
            let head_dim = geom.head_dim as usize;

            let (
                k,
                v,
                virtual_kv_cache_k,
                virtual_kv_cache_v,
                virtual_kv_max_t,
                kv_scale_k,
                kv_scale_v,
            ) = if kv_vmm {
                // VMM-backed: reserve full logical capacity AND map all of
                // it up front. We don't evict Qwen3.6 KV in v1 (CpuBackup is
                // off — VirtualBacking::Discard), and the kernel writes K/V
                // at arbitrary `eff_cache_pos` between 0 and kv_max_t-1, so
                // any unmapped page would page-fault mid-launch. The benefit
                // here is stable VA and shared VMM descriptor plumbing, not
                // decode-time KV paging yet.
                let kv_dtype = if kv_fp8 {
                    ScalarType::U8
                } else {
                    ScalarType::BF16
                };
                let kv_dtype_bytes = if kv_fp8 { 1 } else { 2 };
                let map_full_bytes = kv_max_t * kv_dim * kv_dtype_bytes;
                let vk = gpu_hal::VirtualBuffer::reserve_and_map_prefix(
                    ordinal,
                    kv_dtype,
                    &[kv_max_t, kv_dim],
                    map_full_bytes,
                    gpu_hal::VirtualBacking::Discard,
                )
                .with_context(|| format!("vmm reserve kv_cache_k (layer {layer_idx})"))?;
                let vv = gpu_hal::VirtualBuffer::reserve_and_map_prefix(
                    ordinal,
                    kv_dtype,
                    &[kv_max_t, kv_dim],
                    map_full_bytes,
                    gpu_hal::VirtualBacking::Discard,
                )
                .with_context(|| format!("vmm reserve kv_cache_v (layer {layer_idx})"))?;
                let (sk, sv) = if kv_fp8 {
                    let sk = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                        .with_context(|| format!("alloc kv_scale_k (layer {layer_idx})"))?;
                    let sv = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                        .with_context(|| format!("alloc kv_scale_v (layer {layer_idx})"))?;
                    (Some(sk), Some(sv))
                } else {
                    (None, None)
                };
                (None, None, Some(vk), Some(vv), Some(kv_max_t), sk, sv)
            } else if kv_fp8 {
                // dense FP8 path
                let k = GpuBuffer::zeros(ordinal, ScalarType::U8, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_k FP8 (layer {layer_idx})"))?;
                let v = GpuBuffer::zeros(ordinal, ScalarType::U8, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_v FP8 (layer {layer_idx})"))?;
                let sk = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                    .with_context(|| format!("alloc kv_scale_k (layer {layer_idx})"))?;
                let sv = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                    .with_context(|| format!("alloc kv_scale_v (layer {layer_idx})"))?;
                (Some(k), Some(v), None, None, None, Some(sk), Some(sv))
            } else {
                // BF16 path
                let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_k (layer {layer_idx})"))?;
                let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_v (layer {layer_idx})"))?;
                (Some(k), Some(v), None, None, None, None, None)
            };
            let (kv_shadow_k, kv_shadow_v, kv_shadow_window) =
                if kv_fp8 && load_options.kv_fp8_sidecar.enabled {
                    let window = load_options
                        .kv_fp8_sidecar
                        .window_tokens
                        .unwrap_or(kv_max_t)
                        .min(kv_max_t);
                    if window == 0 {
                        (None, None, 0)
                    } else {
                        let sk = GpuBuffer::zeros(
                            ordinal,
                            ScalarType::BF16,
                            &[num_kv_heads, window, head_dim],
                        )
                        .with_context(|| format!("alloc kv_shadow_k (layer {layer_idx})"))?;
                        let sv = GpuBuffer::zeros(
                            ordinal,
                            ScalarType::BF16,
                            &[num_kv_heads, window, head_dim],
                        )
                        .with_context(|| format!("alloc kv_shadow_v (layer {layer_idx})"))?;
                        (Some(sk), Some(sv), window as i32)
                    }
                } else {
                    (None, None, 0)
                };
            Some(FullAttnKvCache {
                k,
                v,
                kv_max_t: kv_max_t as i32,
                kv_scale_k,
                kv_scale_v,
                kv_shadow_k,
                kv_shadow_v,
                kv_shadow_start: -1,
                kv_shadow_window,
                virtual_kv_cache_k,
                virtual_kv_cache_v,
                virtual_kv_max_t,
            })
        } else {
            None
        };
        AttnLayerBuffers::Full {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            q_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight"))?,
            k_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight"))?,
            v_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight"))?,
            q_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.q_norm.weight"))?,
            k_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.k_norm.weight"))?,
            o_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight"))?,
            int4,
            kv_cache,
        }
    } else {
        let la = format!("{lp}.linear_attn");
        let kernel = geom.conv_kernel_dim as usize;
        let key_dim = (geom.num_k_heads as usize) * (geom.head_k_dim as usize);
        let val_dim = (geom.num_v_heads as usize) * (geom.head_v_dim as usize);
        let qkv_dim = 2 * key_dim + val_dim;
        let state_elems =
            (geom.num_v_heads as usize) * (geom.head_k_dim as usize) * (geom.head_v_dim as usize);

        // First-decode-token state: conv + recurrent both zeros. The kernel
        // mutates them in place; PR 4d will persist them across decode steps.
        let conv_state = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kernel - 1])
            .with_context(|| format!("alloc conv_state (layer {layer_idx})"))?;
        let recurrent_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[state_elems])
            .with_context(|| format!("alloc recurrent_state (layer {layer_idx})"))?;

        let (in_proj_qkv_type, in_proj_z_type, out_proj_type) = (
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
            QWEN36_MOE_LOWBIT_NATIVE_INT4,
        );
        let int4 = match weight_mode {
            Qwen36WeightMode::Int4 => Some(LinearAttnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                in_proj_qkv_type,
                in_proj_qkv: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_qkv.weight"),
                    &format!("{la}.in_proj_qkv.weight_int4_scale"),
                    &format!("{la}.in_proj_qkv.weight_int4_zero"),
                )?,
                in_proj_z_type,
                in_proj_z: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_z.weight"),
                    &format!("{la}.in_proj_z.weight_int4_scale"),
                    &format!("{la}.in_proj_z.weight_int4_zero"),
                )?,
                out_proj_type,
                out_proj: load_int4_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.out_proj.weight"),
                    &format!("{la}.out_proj.weight_int4_scale"),
                    &format!("{la}.out_proj.weight_int4_zero"),
                )?,
            }),
            Qwen36WeightMode::Q4Km => {
                let (in_proj_qkv_type, in_proj_qkv) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_qkv.weight"),
                    &format!("{la}.in_proj_qkv.weight_int4_scale"),
                    &format!("{la}.in_proj_qkv.weight_int4_zero"),
                )?;
                let (in_proj_z_type, in_proj_z) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_z.weight"),
                    &format!("{la}.in_proj_z.weight_int4_scale"),
                    &format!("{la}.in_proj_z.weight_int4_zero"),
                )?;
                let (out_proj_type, out_proj) = load_q4km_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.out_proj.weight"),
                    &format!("{la}.out_proj.weight_int4_scale"),
                    &format!("{la}.out_proj.weight_int4_zero"),
                )?;
                Some(LinearAttnInt4Sidecars {
                    group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                    in_proj_qkv_type,
                    in_proj_qkv,
                    in_proj_z_type,
                    in_proj_z,
                    out_proj_type,
                    out_proj,
                })
            }
            Qwen36WeightMode::Fp8 => Some(LinearAttnInt4Sidecars {
                group_size: -QWEN36_MOE_FP8_BLOCK_SIZE,
                in_proj_qkv_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                in_proj_qkv: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_qkv.weight"),
                    &format!("{la}.in_proj_qkv.weight_scale_inv"),
                )?,
                in_proj_z_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                in_proj_z: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_z.weight"),
                    &format!("{la}.in_proj_z.weight_scale_inv"),
                )?,
                out_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
                out_proj: load_fp8_projection_sidecar(
                    store,
                    ordinal,
                    &format!("{la}.out_proj.weight"),
                    &format!("{la}.out_proj.weight_scale_inv"),
                )?,
            }),
            Qwen36WeightMode::Bf16 => None,
        };

        AttnLayerBuffers::Linear {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            in_proj_qkv_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_qkv.weight"))?,
            in_proj_z_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_z.weight"))?,
            in_proj_a_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_a.weight"))?,
            in_proj_b_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_b.weight"))?,
            conv1d_w: load_to_gpu(store, ordinal, &format!("{la}.conv1d.weight"))?,
            // conv1d.bias may be absent — match the loader's behaviour.
            conv1d_bias: store
                .contains(&format!("{la}.conv1d.bias"))
                .then(|| load_to_gpu(store, ordinal, &format!("{la}.conv1d.bias")))
                .transpose()?,
            dt_bias: load_to_gpu(store, ordinal, &format!("{la}.dt_bias"))?,
            a_log: load_to_gpu(store, ordinal, &format!("{la}.A_log"))?,
            norm_w: load_to_gpu(store, ordinal, &format!("{la}.norm.weight"))?,
            out_proj_w: load_to_gpu(store, ordinal, &format!("{la}.out_proj.weight"))?,
            conv_state,
            recurrent_state,
            int4,
        }
    };

    let mp = format!("{lp}.mlp");
    // Fused-expert sidecars use `_int4_scale`/`_int4_zero` (no `.weight`).
    // Shared-expert MLPs use the dense `<name>.weight_int4_scale` form.
    let ffn_int4 = match weight_mode {
        Qwen36WeightMode::Int4 => Some(FfnInt4Sidecars {
            group_size: QWEN36_MOE_INT4_GROUP_SIZE,
            gate_up_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            gate_up_proj: load_int4_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.experts.gate_up_proj"),
                &format!("{mp}.experts.gate_up_proj_int4_scale"),
                &format!("{mp}.experts.gate_up_proj_int4_zero"),
            )?,
            down_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            down_proj: load_int4_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.experts.down_proj"),
                &format!("{mp}.experts.down_proj_int4_scale"),
                &format!("{mp}.experts.down_proj_int4_zero"),
            )?,
            shared_gate_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            shared_gate_proj: load_int4_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.gate_proj.weight"),
                &format!("{mp}.shared_expert.gate_proj.weight_int4_scale"),
                &format!("{mp}.shared_expert.gate_proj.weight_int4_zero"),
            )?,
            shared_up_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            shared_up_proj: load_int4_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.up_proj.weight"),
                &format!("{mp}.shared_expert.up_proj.weight_int4_scale"),
                &format!("{mp}.shared_expert.up_proj.weight_int4_zero"),
            )?,
            shared_down_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            shared_down_proj: load_int4_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.down_proj.weight"),
                &format!("{mp}.shared_expert.down_proj.weight_int4_scale"),
                &format!("{mp}.shared_expert.down_proj.weight_int4_zero"),
            )?,
        }),
        Qwen36WeightMode::Q4Km => {
            let (gate_up_proj_type, gate_up_proj) = load_q4km_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.experts.gate_up_proj"),
                &format!("{mp}.experts.gate_up_proj_int4_scale"),
                &format!("{mp}.experts.gate_up_proj_int4_zero"),
            )?;
            let (down_proj_type, down_proj) = load_q4km_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.experts.down_proj"),
                &format!("{mp}.experts.down_proj_int4_scale"),
                &format!("{mp}.experts.down_proj_int4_zero"),
            )?;
            let (shared_gate_proj_type, shared_gate_proj) = load_q4km_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.gate_proj.weight"),
                &format!("{mp}.shared_expert.gate_proj.weight_int4_scale"),
                &format!("{mp}.shared_expert.gate_proj.weight_int4_zero"),
            )?;
            let (shared_up_proj_type, shared_up_proj) = load_q4km_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.up_proj.weight"),
                &format!("{mp}.shared_expert.up_proj.weight_int4_scale"),
                &format!("{mp}.shared_expert.up_proj.weight_int4_zero"),
            )?;
            let (shared_down_proj_type, shared_down_proj) = load_q4km_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.down_proj.weight"),
                &format!("{mp}.shared_expert.down_proj.weight_int4_scale"),
                &format!("{mp}.shared_expert.down_proj.weight_int4_zero"),
            )?;
            Some(FfnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                gate_up_proj_type,
                gate_up_proj,
                down_proj_type,
                down_proj,
                shared_gate_proj_type,
                shared_gate_proj,
                shared_up_proj_type,
                shared_up_proj,
                shared_down_proj_type,
                shared_down_proj,
            })
        }
        Qwen36WeightMode::Fp8 => Some(FfnInt4Sidecars {
            group_size: -QWEN36_MOE_FP8_BLOCK_SIZE,
            gate_up_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            gate_up_proj: load_fp8_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.experts.gate_up_proj"),
                &format!("{mp}.experts.gate_up_proj_scale_inv"),
            )?,
            down_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            down_proj: load_fp8_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.experts.down_proj"),
                &format!("{mp}.experts.down_proj_scale_inv"),
            )?,
            shared_gate_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            shared_gate_proj: load_fp8_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.gate_proj.weight"),
                &format!("{mp}.shared_expert.gate_proj.weight_scale_inv"),
            )?,
            shared_up_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            shared_up_proj: load_fp8_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.up_proj.weight"),
                &format!("{mp}.shared_expert.up_proj.weight_scale_inv"),
            )?,
            shared_down_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
            shared_down_proj: load_fp8_projection_sidecar(
                store,
                ordinal,
                &format!("{mp}.shared_expert.down_proj.weight"),
                &format!("{mp}.shared_expert.down_proj.weight_scale_inv"),
            )?,
        }),
        Qwen36WeightMode::Bf16 => None,
    };
    let ffn = FfnLayerBuffers {
        post_attn_norm_w: load_to_gpu(
            store,
            ordinal,
            &format!("{lp}.post_attention_layernorm.weight"),
        )?,
        gate_w: load_to_gpu(store, ordinal, &format!("{mp}.gate.weight"))?,
        // Note: experts.gate_up_proj / experts.down_proj have NO `.weight`
        // suffix in the published checkpoint — see expected_tensor_specs.
        gate_up_proj_w: if let Some(manager) = expert_residency.as_mut() {
            load_to_sparse_resident_weight(
                store,
                &mut **manager,
                layer_idx,
                MoeExpertProjection::GateUp,
                &format!("{mp}.experts.gate_up_proj"),
                geom.num_experts as usize,
            )?
        } else {
            match expert_arena.as_mut() {
                Some(arena) => load_to_virtual_resident_weight(
                    store,
                    &mut **arena,
                    &format!("{mp}.experts.gate_up_proj"),
                    expert_virtual_transfer_backend,
                )?,
                None => load_to_gpu_with_options(
                    store,
                    ordinal,
                    &format!("{mp}.experts.gate_up_proj"),
                    load_options,
                )
                .map(ResidentWeight::Dense)?,
            }
        },
        down_proj_w: if let Some(manager) = expert_residency.as_mut() {
            load_to_sparse_resident_weight(
                store,
                &mut **manager,
                layer_idx,
                MoeExpertProjection::Down,
                &format!("{mp}.experts.down_proj"),
                geom.num_experts as usize,
            )?
        } else {
            match expert_arena.as_mut() {
                Some(arena) => load_to_virtual_resident_weight(
                    store,
                    &mut **arena,
                    &format!("{mp}.experts.down_proj"),
                    expert_virtual_transfer_backend,
                )?,
                None => {
                    load_to_resident_weight(store, ordinal, &format!("{mp}.experts.down_proj"))?
                }
            }
        },
        shared_gate_proj_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert.gate_proj.weight"),
        )?,
        shared_up_proj_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert.up_proj.weight"),
        )?,
        shared_down_proj_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert.down_proj.weight"),
        )?,
        shared_expert_gate_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert_gate.weight"),
        )?,
        int4: ffn_int4,
    };

    Ok(LayerBuffers { attn, ffn })
}

fn load_all_layer_buffers(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    kv_max_t: usize,
    kv_fp8: bool,
    kv_vmm: bool,
    mut expert_arena: Option<&mut VirtualArena>,
    mut expert_residency: Option<&mut MoeExpertResidencyManager>,
    expert_virtual_transfer_backend: VirtualArenaTransferBackend,
    load_options: &Qwen36LoadOptions,
) -> Result<Vec<LayerBuffers>> {
    let mut layers = Vec::with_capacity(geom.num_layers as usize);
    for li in 0..geom.num_layers as usize {
        let layer = load_layer_buffers(
            store,
            ordinal,
            li,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            expert_arena.as_deref_mut(),
            expert_residency.as_deref_mut(),
            expert_virtual_transfer_backend,
            load_options,
        )
        .with_context(|| format!("load layer {li} weights"))?;
        layers.push(layer);
    }
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
pub fn load_qwen36_layers(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    kv_max_t: usize,
    kv_fp8: bool,
    kv_vmm: bool,
    strategy: Qwen36LayerLoadStrategy,
    load_options: &Qwen36LoadOptions,
) -> Result<LoadedQwen36Layers> {
    match strategy {
        Qwen36LayerLoadStrategy::Dense => {
            let layers = load_all_layer_buffers(
                store,
                ordinal,
                geom,
                text_config,
                weight_prefix,
                weight_mode,
                kv_max_t,
                kv_fp8,
                kv_vmm,
                None,
                None,
                VirtualArenaTransferBackend::PageableH2d,
                load_options,
            )
            .context("load dense Qwen3.6 layers")?;
            Ok(LoadedQwen36Layers::dense(layers, weight_mode))
        }
        Qwen36LayerLoadStrategy::VirtualExperts { transfer_backend } => {
            let mut arena = BakedStore::virtual_weight_arena(ordinal);
            let layers = load_all_layer_buffers(
                store,
                ordinal,
                geom,
                text_config,
                weight_prefix,
                weight_mode,
                kv_max_t,
                kv_fp8,
                kv_vmm,
                Some(&mut arena),
                None,
                transfer_backend,
                load_options,
            )
            .context("load Qwen3.6 routed experts into a virtual arena")?;
            Ok(LoadedQwen36Layers::with_backing(
                layers,
                weight_mode,
                Some(arena),
                None,
            ))
        }
        Qwen36LayerLoadStrategy::SparseExperts(options) => {
            if options.cap_experts < geom.top_k as usize {
                anyhow::bail!(
                    "sparse expert capacity {} is smaller than model top_k={}",
                    options.cap_experts,
                    geom.top_k
                );
            }
            let config =
                MoeExpertResidencyConfig::new(1)?.with_prefetch_evict(options.prefetch_evict);
            let mut manager = MoeExpertResidencyManager::new(ordinal, config)
                .with_virtual_transfer_backend(options.transfer_backend);
            let layers = load_all_layer_buffers(
                store,
                ordinal,
                geom,
                text_config,
                weight_prefix,
                weight_mode,
                kv_max_t,
                kv_fp8,
                kv_vmm,
                None,
                Some(&mut manager),
                options.transfer_backend,
                load_options,
            )
            .context("reserve Qwen3.6 routed experts for sparse residency")?;
            let max_resident_pages = manager
                .page_budget_for_routed_experts(options.cap_experts)
                .context("derive sparse MoE page budget")?;
            manager
                .set_max_resident_pages(max_resident_pages)
                .context("apply sparse MoE page budget")?;
            if let Some(protected_experts) = options.protected_experts {
                let pages = manager
                    .page_budget_for_routed_experts(protected_experts)
                    .context("derive sparse MoE protected page budget")?
                    .min(max_resident_pages);
                manager.set_max_protected_pages(pages);
            }
            if let Some(fixed_hot_experts) = options.fixed_hot_experts {
                let pages = manager
                    .page_budget_for_routed_experts(fixed_hot_experts)
                    .context("derive sparse MoE fixed-hot page budget")?
                    .min(max_resident_pages);
                manager.set_max_fixed_hot_pages(pages);
            }
            if options.async_prefetch
                && options.transfer_backend != VirtualArenaTransferBackend::GpuDirectStorage
            {
                manager
                    .enable_async_prefetch(options.async_staging_pages)
                    .context("enable sparse MoE async prefetch")?;
            }
            Ok(LoadedQwen36Layers::with_backing(
                layers,
                weight_mode,
                None,
                Some(manager),
            ))
        }
    }
}

#[cfg(test)]
mod direct_load_tests {
    use super::*;
    use crate::qwen36_moe::chain::{run_chain_step, Qwen36ChainStep};
    use crate::qwen36_moe::decode::Qwen36ExecutionOptions;
    use crate::qwen36_moe::persistent_decode::{build_int4_weight_desc, validate_int4_sidecar_set};
    use crate::qwen36_moe::prefill::{run_batched_prefill, supports_batched_path};
    use crate::qwen36_moe::residency::{MoeExpertResidencyConfig, MoeExpertResidencyManager};
    use crate::qwen36_moe::types::PositionPair;
    use gpu_hal::{copy_d2h, copy_h2d};
    use model_store::manifest::{Manifest, TensorMeta, FORMAT_VERSION};
    use model_store::store::{Int4StorageKind, Int4StorageView};
    use qwen36_moe::config::Activation;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(mode: Qwen36WeightMode) -> Self {
            let suffix = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "supersonic-runtime-qwen36-direct-load-{}-{suffix}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create direct-load test directory");
            let (weights, tensors) = synthetic_layer_tensors(mode);
            std::fs::write(model_store::weights_bin_path(&path), weights)
                .expect("write synthetic weights");
            let manifest = Manifest {
                format_version: FORMAT_VERSION,
                converter_version: 1,
                model_family: "test-qwen36-direct-load".to_string(),
                quant_profile: None,
                source_format: None,
                source_quant: None,
                quant_method: None,
                tensors,
            };
            std::fs::write(
                model_store::manifest_path(&path),
                serde_json::to_string(&manifest).expect("serialize synthetic manifest"),
            )
            .expect("write synthetic manifest");
            Self(path)
        }

        fn new_orchestration() -> Self {
            let suffix = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "supersonic-runtime-qwen36-prefill-orchestration-{}-{suffix}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create prefill orchestration test directory");
            let (weights, tensors) = synthetic_orchestration_tensors();
            std::fs::write(model_store::weights_bin_path(&path), weights)
                .expect("write orchestration weights");
            let manifest = Manifest {
                format_version: FORMAT_VERSION,
                converter_version: 1,
                model_family: "test-qwen36-prefill-orchestration".to_string(),
                quant_profile: None,
                source_format: None,
                source_quant: None,
                quant_method: None,
                tensors,
            };
            std::fs::write(
                model_store::manifest_path(&path),
                serde_json::to_string(&manifest).expect("serialize orchestration manifest"),
            )
            .expect("write orchestration manifest");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn text_config() -> TextConfig {
        TextConfig {
            vocab_size: 128,
            hidden_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            eos_token_id: None,
            bos_token_id: None,
            head_dim: 128,
            full_attention_interval: 1,
            attn_output_gate: true,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            layer_types: vec!["full_attention".to_string()],
            rope_parameters: None,
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 128,
            shared_expert_intermediate_size: 128,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: Vec::new(),
            decoder_sparse_step: None,
        }
    }

    fn geom() -> MultiLayerGeom {
        MultiLayerGeom {
            hidden: 128,
            vocab: 128,
            num_layers: 1,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 128,
            rotary_dim: 128,
            rope_theta: 10_000.0,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_kernel_dim: 2,
            num_experts: 1,
            moe_intermediate: 128,
            shared_intermediate: 128,
            top_k: 1,
        }
    }

    fn orchestration_text_config() -> TextConfig {
        TextConfig {
            num_hidden_layers: 2,
            max_position_embeddings: 1024,
            full_attention_interval: 2,
            layer_types: vec!["full_attention".to_string(), "linear_attention".to_string()],
            num_experts: 4,
            num_experts_per_tok: 2,
            ..text_config()
        }
    }

    fn orchestration_geom() -> MultiLayerGeom {
        MultiLayerGeom {
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            ..geom()
        }
    }

    fn test_bf16_bytes(len: usize, mut value: impl FnMut(usize) -> f32) -> Vec<u8> {
        (0..len)
            .flat_map(|idx| half::bf16::from_f32(value(idx)).to_bits().to_le_bytes())
            .collect()
    }

    fn push_test_tensor(
        weights: &mut Vec<u8>,
        tensors: &mut Vec<TensorMeta>,
        name: String,
        shape: &[usize],
        dtype: &str,
        layout: LayoutTag,
        bytes: Vec<u8>,
    ) {
        let offset = weights.len() as u64;
        weights.extend_from_slice(&bytes);
        tensors.push(TensorMeta {
            name,
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
            layout,
            offset,
            byte_len: bytes.len() as u64,
        });
    }

    fn push_test_bf16(
        weights: &mut Vec<u8>,
        tensors: &mut Vec<TensorMeta>,
        name: String,
        shape: &[usize],
        value: impl FnMut(usize) -> f32,
    ) {
        push_test_tensor(
            weights,
            tensors,
            name,
            shape,
            "bf16",
            LayoutTag::Raw,
            test_bf16_bytes(shape.iter().product(), value),
        );
    }

    fn push_test_int4(
        weights: &mut Vec<u8>,
        tensors: &mut Vec<TensorMeta>,
        name: String,
        shape: &[usize],
        packed_seed: u8,
        scale: f32,
        varied: bool,
    ) {
        let mut packed_shape = shape.to_vec();
        *packed_shape.last_mut().expect("INT4 projection rank") /= 2;
        let packed_len = packed_shape.iter().product();
        push_test_tensor(
            weights,
            tensors,
            name.clone(),
            &packed_shape,
            "u8",
            LayoutTag::Int4Quantized,
            (0..packed_len)
                .map(|idx| {
                    let low = if varied {
                        1 + (packed_seed as usize + idx * 3) % 6
                    } else {
                        packed_seed as usize
                    };
                    let high = if varied {
                        1 + (packed_seed as usize + idx * 5 + 2) % 6
                    } else {
                        packed_seed as usize
                    };
                    ((high as u8) << 4) | low as u8
                })
                .collect(),
        );
        let sidecar_shape = match shape {
            [rows, cols] => vec![rows.div_ceil(128), cols.div_ceil(128)],
            [experts, rows, cols] => {
                vec![*experts, rows.div_ceil(128), cols.div_ceil(128)]
            }
            _ => panic!("unsupported INT4 test projection shape {shape:?}"),
        };
        push_test_bf16(
            weights,
            tensors,
            format!("{name}_int4_scale"),
            &sidecar_shape,
            |idx| {
                if varied {
                    scale * (0.8 + 0.15 * (idx % 5) as f32)
                } else {
                    scale
                }
            },
        );
        push_test_bf16(
            weights,
            tensors,
            format!("{name}_int4_zero"),
            &sidecar_shape,
            |_| 0.0,
        );
    }

    fn synthetic_orchestration_tensors() -> (Vec<u8>, Vec<TensorMeta>) {
        let mut weights = Vec::new();
        let mut tensors = Vec::new();
        push_test_bf16(
            &mut weights,
            &mut tensors,
            "model.language_model.embed_tokens.weight".to_string(),
            &[128, 128],
            |idx| {
                let token = idx / 128;
                let column = idx % 128;
                (((token * 17 + column * 3) % 31) as f32 - 15.0) * 0.002
            },
        );

        for layer_idx in 0..2 {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            push_test_bf16(
                &mut weights,
                &mut tensors,
                format!("{prefix}.input_layernorm.weight"),
                &[128],
                |_| 1.0,
            );
            if layer_idx == 0 {
                let attn = format!("{prefix}.self_attn");
                push_test_bf16(
                    &mut weights,
                    &mut tensors,
                    format!("{attn}.q_norm.weight"),
                    &[128],
                    |_| 1.0,
                );
                push_test_bf16(
                    &mut weights,
                    &mut tensors,
                    format!("{attn}.k_norm.weight"),
                    &[128],
                    |_| 1.0,
                );
                for (name, shape, nibble, scale) in [
                    (format!("{attn}.q_proj.weight"), vec![256, 128], 1, 0.001),
                    (format!("{attn}.k_proj.weight"), vec![128, 128], 2, 0.001),
                    (format!("{attn}.v_proj.weight"), vec![128, 128], 3, 0.001),
                    (format!("{attn}.o_proj.weight"), vec![128, 128], 1, 0.001),
                ] {
                    push_test_int4(
                        &mut weights,
                        &mut tensors,
                        name,
                        &shape,
                        nibble,
                        scale,
                        false,
                    );
                }
            } else {
                let attn = format!("{prefix}.linear_attn");
                for (name, shape, nibble, scale) in [
                    (
                        format!("{attn}.in_proj_qkv.weight"),
                        vec![384, 128],
                        1,
                        0.001,
                    ),
                    (format!("{attn}.in_proj_z.weight"), vec![128, 128], 2, 0.001),
                    (format!("{attn}.out_proj.weight"), vec![128, 128], 1, 0.001),
                ] {
                    push_test_int4(
                        &mut weights,
                        &mut tensors,
                        name,
                        &shape,
                        nibble,
                        scale,
                        false,
                    );
                }
                for (suffix, shape, value) in [
                    ("in_proj_a.weight", vec![1, 128], 0.002),
                    ("in_proj_b.weight", vec![1, 128], -0.002),
                    ("conv1d.weight", vec![384, 2], 0.01),
                    ("dt_bias", vec![1], 0.0),
                    ("A_log", vec![1], -1.0),
                    ("norm.weight", vec![128], 1.0),
                ] {
                    push_test_bf16(
                        &mut weights,
                        &mut tensors,
                        format!("{attn}.{suffix}"),
                        &shape,
                        |_| value,
                    );
                }
            }

            let mlp = format!("{prefix}.mlp");
            push_test_bf16(
                &mut weights,
                &mut tensors,
                format!("{prefix}.post_attention_layernorm.weight"),
                &[128],
                |_| 1.0,
            );
            push_test_bf16(
                &mut weights,
                &mut tensors,
                format!("{mlp}.gate.weight"),
                &[4, 128],
                |idx| {
                    let expert = idx / 128;
                    let column = idx % 128;
                    ((((expert + 1) * 11 + column * (expert + 3)) % 29) as f32 - 14.0) * 0.012
                },
            );
            push_test_bf16(
                &mut weights,
                &mut tensors,
                format!("{mlp}.shared_expert_gate.weight"),
                &[1, 128],
                |_| 0.005,
            );
            for (name, shape, nibble, scale) in [
                (
                    format!("{mlp}.experts.gate_up_proj"),
                    vec![4, 256, 128],
                    1,
                    0.0015,
                ),
                (
                    format!("{mlp}.experts.down_proj"),
                    vec![4, 128, 128],
                    3,
                    0.0012,
                ),
                (
                    format!("{mlp}.shared_expert.gate_proj.weight"),
                    vec![128, 128],
                    1,
                    0.0013,
                ),
                (
                    format!("{mlp}.shared_expert.up_proj.weight"),
                    vec![128, 128],
                    2,
                    0.0016,
                ),
                (
                    format!("{mlp}.shared_expert.down_proj.weight"),
                    vec![128, 128],
                    4,
                    0.0012,
                ),
            ] {
                push_test_int4(
                    &mut weights,
                    &mut tensors,
                    name,
                    &shape,
                    nibble,
                    scale,
                    true,
                );
            }
        }
        (weights, tensors)
    }

    fn synthetic_layer_tensors(mode: Qwen36WeightMode) -> (Vec<u8>, Vec<TensorMeta>) {
        let prefix = "model.language_model.layers.0";
        let mut weights = Vec::new();
        let mut tensors = Vec::new();
        let mut push = |name: String, shape: &[usize], dtype: &str, layout: LayoutTag| {
            let elem_bytes = if dtype == "bf16" { 2 } else { 1 };
            let byte_len = shape.iter().product::<usize>() * elem_bytes;
            let offset = weights.len() as u64;
            weights.resize(weights.len() + byte_len, 0);
            tensors.push(TensorMeta {
                name,
                shape: shape.to_vec(),
                dtype: dtype.to_string(),
                layout,
                offset,
                byte_len: byte_len as u64,
            });
        };
        let bf16 = |name: &str,
                    shape: &[usize],
                    push: &mut dyn FnMut(String, &[usize], &str, LayoutTag)| {
            push(name.to_string(), shape, "bf16", LayoutTag::Raw);
        };

        bf16(
            &format!("{prefix}.input_layernorm.weight"),
            &[128],
            &mut push,
        );
        bf16(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[128],
            &mut push,
        );
        bf16(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[128],
            &mut push,
        );
        bf16(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[128],
            &mut push,
        );
        bf16(&format!("{prefix}.mlp.gate.weight"), &[1, 128], &mut push);
        bf16(
            &format!("{prefix}.mlp.shared_expert_gate.weight"),
            &[1, 128],
            &mut push,
        );

        let projections = [
            (format!("{prefix}.self_attn.q_proj.weight"), vec![256, 128]),
            (format!("{prefix}.self_attn.k_proj.weight"), vec![128, 128]),
            (format!("{prefix}.self_attn.v_proj.weight"), vec![128, 128]),
            (format!("{prefix}.self_attn.o_proj.weight"), vec![128, 128]),
            (
                format!("{prefix}.mlp.experts.gate_up_proj"),
                vec![1, 256, 128],
            ),
            (format!("{prefix}.mlp.experts.down_proj"), vec![1, 128, 128]),
            (
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                vec![128, 128],
            ),
            (
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                vec![128, 128],
            ),
            (
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                vec![128, 128],
            ),
        ];
        for (name, shape) in projections {
            if mode == Qwen36WeightMode::Bf16 {
                bf16(&name, &shape, &mut push);
                continue;
            }
            let mut packed_shape = shape.clone();
            *packed_shape.last_mut().expect("projection rank") /= 2;
            push(name.clone(), &packed_shape, "u8", LayoutTag::Int4Quantized);
            bf16(&format!("{name}_int4_scale"), &[1], &mut push);
            bf16(&format!("{name}_int4_zero"), &[1], &mut push);
        }
        (weights, tensors)
    }

    fn load_direct(mode: Qwen36WeightMode) -> LoadedQwen36Layers {
        let tmp = TestDir::new(mode);
        let store = BakedStore::open(tmp.path()).expect("open synthetic direct-load store");
        load_qwen36_layers(
            &store,
            0,
            &geom(),
            &text_config(),
            "model.language_model",
            mode,
            4,
            false,
            false,
            Qwen36LayerLoadStrategy::Dense,
            &Qwen36LoadOptions::default(),
        )
        .expect("runtime direct layer load")
    }

    const ROW_GROUP_FLM_LOGICAL_NAME: &str = "semantic.row_group.weight";

    fn load_row_group_flm_sidecar(store: &BakedStore) -> LoadedInt4Sidecar {
        load_int4_projection_sidecar(
            store,
            0,
            ROW_GROUP_FLM_LOGICAL_NAME,
            "unused-row-group-scale-alias",
            "unused-row-group-zero-alias",
        )
        .expect("load row-group sidecar through BakedStore storage view")
    }

    fn install_row_group_flm_sidecars(store: &BakedStore, loaded: &mut LoadedQwen36Layers) {
        let layer = &mut loaded.layers_mut_before_persistent().unwrap()[0];
        let AttnLayerBuffers::Full {
            int4: Some(attn), ..
        } = &mut layer.attn
        else {
            panic!("expected native-INT4 full-attention sidecars");
        };
        attn.q_proj = load_row_group_flm_sidecar(store);
        attn.k_proj = load_row_group_flm_sidecar(store);
        attn.v_proj = load_row_group_flm_sidecar(store);
        attn.o_proj = load_row_group_flm_sidecar(store);

        let ffn = layer.ffn.int4.as_mut().expect("native-INT4 FFN sidecars");
        ffn.gate_up_proj = load_row_group_flm_sidecar(store);
        ffn.down_proj = load_row_group_flm_sidecar(store);
        ffn.shared_gate_proj = load_row_group_flm_sidecar(store);
        ffn.shared_up_proj = load_row_group_flm_sidecar(store);
        ffn.shared_down_proj = load_row_group_flm_sidecar(store);
    }

    #[test]
    fn runtime_direct_load_owns_bf16_layers_and_kv_state() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let loaded = load_direct(Qwen36WeightMode::Bf16);
        assert_eq!(loaded.weight_mode(), Qwen36WeightMode::Bf16);
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            classify_layer_weight_encoding(loaded.layers()).expect("classify BF16"),
            Qwen36LayerWeightEncoding::Bf16
        );
        let AttnLayerBuffers::Full {
            q_proj_w,
            int4,
            kv_cache,
            ..
        } = &loaded.layers()[0].attn
        else {
            panic!("expected full-attention layer");
        };
        assert_eq!(q_proj_w.dtype(), ScalarType::BF16);
        assert_eq!(q_proj_w.shape(), &[256, 128]);
        assert!(int4.is_none());
        assert_eq!(kv_cache.as_ref().expect("KV cache").kv_max_t, 4);
        assert_eq!(
            loaded.layers()[0].ffn.gate_up_proj_w.dtype(),
            ScalarType::BF16
        );
        assert_eq!(
            loaded.layers()[0].ffn.gate_up_proj_w.shape(),
            &[1, 256, 128]
        );
        assert!(loaded.layers()[0].ffn.int4.is_none());
    }

    #[test]
    fn runtime_direct_load_owns_native_int4_layers_and_sidecars() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let loaded = load_direct(Qwen36WeightMode::Int4);
        assert_eq!(loaded.weight_mode(), Qwen36WeightMode::Int4);
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            classify_layer_weight_encoding(loaded.layers()).expect("classify native INT4"),
            Qwen36LayerWeightEncoding::NativeInt4
        );
        let AttnLayerBuffers::Full { q_proj_w, int4, .. } = &loaded.layers()[0].attn else {
            panic!("expected full-attention layer");
        };
        assert_eq!(q_proj_w.dtype(), ScalarType::U8);
        assert_eq!(q_proj_w.shape(), &[256, 64]);
        let attn_int4 = int4.as_ref().expect("attention INT4 sidecars");
        assert_eq!(attn_int4.q_proj_type, QWEN36_MOE_LOWBIT_NATIVE_INT4);
        assert_eq!(attn_int4.q_proj.scale.dtype(), ScalarType::BF16);
        assert_eq!(
            loaded.layers()[0].ffn.gate_up_proj_w.dtype(),
            ScalarType::U8
        );
        assert_eq!(loaded.layers()[0].ffn.gate_up_proj_w.shape(), &[16_384]);
        let ffn_int4 = loaded.layers()[0]
            .ffn
            .int4
            .as_ref()
            .expect("FFN INT4 sidecars");
        assert_eq!(ffn_int4.gate_up_proj_type, QWEN36_MOE_LOWBIT_NATIVE_INT4);
        assert_eq!(ffn_int4.gate_up_proj.scale.dtype(), ScalarType::BF16);
    }

    #[test]
    fn row_group_loader_carries_rank3_geometry_and_rejects_mixed_layers() {
        const O: usize = 96;
        const K: usize = 128;

        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let tmp = TestDir::new(Qwen36WeightMode::Int4);
        let store = BakedStore::open(tmp.path()).expect("open sidecar fixture");
        let packed_tensor = "model.language_model.layers.0.self_attn.q_proj.weight".to_string();
        let scale_tensor = format!("{packed_tensor}_int4_scale");
        let zero_tensor = format!("{packed_tensor}_int4_zero");
        let row_group_view = Int4StorageView {
            kind: Int4StorageKind::RowGroupSymmetric,
            group_size: 32,
            packed_tensor: packed_tensor.clone(),
            scale_tensor: scale_tensor.clone(),
            zero_tensor: None,
            logical_shape: vec![128, O, K],
            packed_row_stride_bytes: K / 2,
            packed_expert_stride_bytes: O * K / 2,
            scale_row_stride_elements: K / 32,
            scale_expert_stride_elements: O * K / 32,
            output_group_size: 1,
            implicit_zero_code: Some(8),
        };
        let row_group = load_int4_sidecar_from_view(&store, 0, &row_group_view)
            .expect("load row-group sidecar");
        let desc = build_int4_weight_desc(&row_group).expect("build row-group descriptor");

        assert_eq!(desc.scale, row_group.scale.as_ptr());
        assert!(desc.zero.is_null());
        assert_eq!(desc.packed_row_stride_bytes, (K / 2) as u64);
        assert_eq!(desc.packed_expert_stride_bytes, (O * K / 2) as u64);
        assert_eq!(desc.scale_row_stride_elements, (K / 32) as u64);
        assert_eq!(desc.scale_expert_stride_elements, (O * K / 32) as u64);
        assert_eq!(desc.input_group_size, 32);
        assert_eq!(desc.output_group_size, 1);
        assert_eq!(desc.implicit_zero_code, 8);
        assert_eq!(desc.encoding, 2);

        let tile_view = Int4StorageView {
            kind: Int4StorageKind::TileV1,
            group_size: 128,
            zero_tensor: Some(zero_tensor),
            output_group_size: 128,
            implicit_zero_code: None,
            ..row_group_view.clone()
        };
        let tile =
            load_int4_sidecar_from_view(&store, 0, &tile_view).expect("load tile-v1 sidecar");
        assert_eq!(
            build_int4_weight_desc(&tile)
                .expect("build tile-v1 descriptor")
                .encoding,
            1
        );

        let fp8_view = Int4StorageView {
            zero_tensor: None,
            ..tile_view.clone()
        };
        let fp8 = load_int4_sidecar_from_view(&store, 0, &fp8_view).expect("load FP8 sidecar");
        assert_eq!(
            build_int4_weight_desc(&fp8)
                .expect("build FP8 descriptor")
                .encoding,
            3
        );

        let mixed_encoding =
            validate_int4_sidecar_set(0, &[("q_proj", &row_group), ("k_proj", &tile)])
                .expect_err("mixed descriptor encodings must fail closed");
        assert!(
            mixed_encoding.to_string().contains("mixes INT4 encodings"),
            "{mixed_encoding}"
        );

        let group64_view = Int4StorageView {
            group_size: 64,
            scale_row_stride_elements: K / 64,
            scale_expert_stride_elements: O * K / 64,
            ..row_group_view
        };
        let group64 = load_int4_sidecar_from_view(&store, 0, &group64_view)
            .expect("load alternate-group sidecar");
        let mixed_group =
            validate_int4_sidecar_set(0, &[("q_proj", &row_group), ("k_proj", &group64)])
                .expect_err("mixed descriptor group sizes must fail closed");
        assert!(
            mixed_group
                .to_string()
                .contains("mixes INT4 input group sizes"),
            "{mixed_group}"
        );
    }

    #[test]
    fn flm_row_group_geometry_is_admitted_only_on_descriptor_driven_decode_paths() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen36_row_group_rank3.flm");
        let store = BakedStore::open_flm(&fixture).expect("open canonical row-group FLM fixture");
        let storage_view = store
            .int4_storage_view(ROW_GROUP_FLM_LOGICAL_NAME)
            .expect("BakedStore row-group storage lookup");
        assert_eq!(storage_view.kind, Int4StorageKind::RowGroupSymmetric);
        assert_eq!(storage_view.logical_shape, [2, 4, 64]);

        let mut loaded = load_direct(Qwen36WeightMode::Int4);
        install_row_group_flm_sidecars(&store, &mut loaded);
        assert_eq!(
            classify_layer_weight_encoding(loaded.layers()).expect("compatibility classification"),
            Qwen36LayerWeightEncoding::NativeInt4,
            "the legacy compatibility classifier must not hide this regression"
        );

        let AttnLayerBuffers::Full {
            int4: Some(attn), ..
        } = &loaded.layers()[0].attn
        else {
            panic!("expected loaded full-attention INT4 sidecars");
        };
        let desc =
            build_int4_weight_desc(&attn.q_proj).expect("build production-loaded descriptor");
        assert_eq!(desc.packed_row_stride_bytes, 32);
        assert_eq!(desc.packed_expert_stride_bytes, 128);
        assert_eq!(desc.scale_row_stride_elements, 2);
        assert_eq!(desc.scale_expert_stride_elements, 8);
        assert_eq!(desc.input_group_size, 32);
        assert_eq!(desc.output_group_size, 1);
        assert_eq!(desc.implicit_zero_code, 8);
        assert!(desc.zero.is_null());
        assert_eq!(desc.encoding, 2);

        ensure_legacy_int4_execution_supported(loaded.layers(), "persistent decode")
            .expect("persistent decode must admit descriptor-driven row-group INT4");
        ensure_legacy_int4_execution_supported(loaded.layers(), "chained decode")
            .expect("chained decode must admit descriptor-driven row-group INT4");
        let unsupported =
            ensure_legacy_int4_execution_supported(loaded.layers(), "unconverted diagnostic path")
                .expect_err("unconverted paths must reject row-group INT4 before device work");
        assert!(
            unsupported
                .to_string()
                .contains("does not advertise descriptor-driven row-group INT4 execution"),
            "unexpected fail-closed guard error: {unsupported:#}"
        );
    }

    #[test]
    fn production_loaded_row_group_flm_prefill_uses_per_token_fallback() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen36_row_group_rank3.flm");
        let row_group_store =
            BakedStore::open_flm(&fixture).expect("open canonical row-group FLM fixture");
        let mut loaded = load_direct(Qwen36WeightMode::Int4);

        assert!(
            supports_batched_path(loaded.layers(), false, true)
                .expect("classify production tile-v1 load"),
            "production tile-v1 G128 must remain admitted"
        );

        install_row_group_flm_sidecars(&row_group_store, &mut loaded);
        let ffn = loaded.layers()[0]
            .ffn
            .int4
            .as_ref()
            .expect("loaded INT4 FFN sidecars");
        assert_eq!(ffn.group_size, 128, "reproduce stale aggregate metadata");
        assert_eq!(ffn.shared_gate_proj.view.group_size, 32);
        assert!(ffn.shared_gate_proj.zero.is_none());
        assert!(
            !supports_batched_path(loaded.layers(), false, true)
                .expect("classify production row-group load"),
            "encoding-2 G32 must not enter the explicit-zero G128 shared-expert path"
        );

        let direct_store_dir = TestDir::new(Qwen36WeightMode::Int4);
        let direct_store = BakedStore::open(direct_store_dir.path()).expect("open direct store");
        let mut execution = Qwen36ExecutionOptions::default();
        execution.batched_prefill.enable_unqualified_hip_native_int4 = true;
        let mut fallback_calls = 0usize;
        let mut callback = |_loaded: &mut LoadedQwen36Layers, step, token, position| {
            assert_eq!((step, token, position), (0, 7, PositionPair::dense(0)));
            fallback_calls += 1;
            Ok(Default::default())
        };
        run_batched_prefill(
            0,
            &geom(),
            &direct_store,
            "model.language_model",
            &mut loaded,
            &[7],
            &[PositionPair::dense(0)],
            false,
            &execution,
            Some(&mut callback),
            None,
        )
        .expect("row-group production load must use descriptor-driven per-token fallback");
        assert_eq!(fallback_calls, 1, "optimized shared expert was entered");
    }

    fn kv_bytes(loaded: &LoadedQwen36Layers) -> Vec<u8> {
        let AttnLayerBuffers::Full {
            kv_cache: Some(cache),
            ..
        } = &loaded.layers()[0].attn
        else {
            panic!("expected dense full-attention KV cache");
        };
        let k = cache.k.as_ref().expect("dense K cache");
        let mut bytes = vec![0u8; k.len_bytes()];
        copy_d2h(0, bytes.as_mut_ptr().cast(), k.as_ptr(), bytes.len()).expect("download K cache");
        bytes
    }

    #[test]
    fn public_persistent_fast_paths_reject_before_model_state_mutation() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut loaded = load_direct(Qwen36WeightMode::Int4);
        let sentinel = vec![0x5au8; 4 * 128 * 2];
        let AttnLayerBuffers::Full {
            kv_cache: Some(cache),
            ..
        } = &mut loaded.layers_mut_before_persistent().unwrap()[0].attn
        else {
            panic!("expected dense full-attention KV cache");
        };
        let k = cache.k.as_mut().expect("dense K cache");
        copy_h2d(0, k.as_mut_ptr(), sentinel.as_ptr().cast(), sentinel.len())
            .expect("seed K cache");
        loaded
            .enable_persistent(0, &geom())
            .expect("enable persistent");

        let embed =
            GpuBuffer::zeros(0, ScalarType::BF16, &[128, 128]).expect("alloc embedding table");
        let wrong_embed =
            GpuBuffer::zeros(0, ScalarType::F32, &[128, 128]).expect("alloc wrong embedding");
        let initial_state = kv_bytes(&loaded);

        let mut invalid_calls = vec![
            loaded
                .run_from_device_embedding_no_download(0, &embed, 128, 0, 0)
                .expect_err("out-of-vocab token must fail"),
            loaded
                .run_from_device_embedding_no_download(0, &wrong_embed, 0, 0, 0)
                .expect_err("wrong embedding dtype must fail"),
            loaded
                .run_dense_prefill_tokens_from_device_embedding(0, &embed, &[0, 1], 0, 3)
                .expect_err("overflowing token loop KV timeline must fail"),
            match loaded.run_segmented_profile(
                0,
                &[0u8; 128 * 2],
                -1,
                0,
                &Qwen36ExecutionOptions::default(),
            ) {
                Ok(_) => panic!("negative segmented position must fail"),
                Err(err) => err,
            },
        ];
        let original_backend = gpu_hal::current_backend();
        let mismatched_backend = match original_backend {
            Backend::Hip => Backend::Cuda,
            Backend::Cuda | Backend::Metal => Backend::Hip,
        };
        gpu_hal::set_backend(mismatched_backend);
        let backend_err = loaded
            .run_from_device_embedding_no_download(0, &embed, 0, 0, 0)
            .expect_err("active backend mismatch must fail before launch");
        gpu_hal::set_backend(original_backend);
        assert!(backend_err.to_string().contains("backend mismatch"));
        invalid_calls.push(backend_err);

        for err in invalid_calls {
            assert!(
                err.to_string().contains("persistent"),
                "unexpected error: {err:#}"
            );
            assert_eq!(
                kv_bytes(&loaded),
                initial_state,
                "invalid persistent request changed model KV state"
            );
        }
    }

    #[test]
    fn sparse_persistent_chain_and_prefill_require_runner_policy_before_execution() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let tmp = TestDir::new(Qwen36WeightMode::Int4);
        let store = BakedStore::open(tmp.path()).expect("open synthetic direct-load store");
        let mut loaded = load_qwen36_layers(
            &store,
            0,
            &geom(),
            &text_config(),
            "model.language_model",
            Qwen36WeightMode::Int4,
            4,
            false,
            false,
            Qwen36LayerLoadStrategy::Dense,
            &Qwen36LoadOptions::default(),
        )
        .expect("runtime direct layer load");
        loaded
            .enable_persistent(0, &geom())
            .expect("enable persistent");
        loaded.attach_test_sparse_expert_residency(MoeExpertResidencyManager::new(
            0,
            MoeExpertResidencyConfig::new(1).expect("test residency config"),
        ));
        let initial_state = kv_bytes(&loaded);

        let chain_err = match run_chain_step(Qwen36ChainStep {
            ordinal: 0,
            geom: &geom(),
            loaded_layers: &mut loaded,
            initial_hidden: &[0u8; 128 * 2],
            position: PositionPair::dense(0),
            step: 0,
            accurate_stage_timings: false,
            fold: None,
            download_final_hidden: true,
            expert_prefetch: None,
            execution: &Qwen36ExecutionOptions::default(),
        }) {
            Ok(_) => panic!("persistent sparse chain without policy must fail"),
            Err(err) => err,
        };
        assert!(chain_err
            .to_string()
            .contains("requires an expert prefetch policy"));
        assert_eq!(kv_bytes(&loaded), initial_state);

        let prefill_err = run_batched_prefill(
            0,
            &geom(),
            &store,
            "model.language_model",
            &mut loaded,
            &[0],
            &[PositionPair::dense(0)],
            false,
            &Qwen36ExecutionOptions::default(),
            None,
            None,
        )
        .expect_err("sparse prefill without policy must fail");
        assert!(prefill_err
            .to_string()
            .contains("requires a fallback token callback"));
        assert_eq!(kv_bytes(&loaded), initial_state);
    }

    fn bf16_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|pair| half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
            .collect()
    }

    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    fn orchestration_state(loaded: &LoadedQwen36Layers) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut kv = Vec::new();
        let mut conv = Vec::new();
        let mut recurrent = Vec::new();
        for layer in loaded.layers() {
            match &layer.attn {
                AttnLayerBuffers::Full {
                    kv_cache: Some(cache),
                    ..
                } => {
                    kv.extend(bf16_values(
                        &cache
                            .k
                            .as_ref()
                            .expect("dense K cache")
                            .to_host_bytes()
                            .unwrap(),
                    ));
                    kv.extend(bf16_values(
                        &cache
                            .v
                            .as_ref()
                            .expect("dense V cache")
                            .to_host_bytes()
                            .unwrap(),
                    ));
                }
                AttnLayerBuffers::Linear {
                    conv_state,
                    recurrent_state,
                    ..
                } => {
                    conv.extend(bf16_values(&conv_state.to_host_bytes().unwrap()));
                    recurrent.extend(f32_values(&recurrent_state.to_host_bytes().unwrap()));
                }
                _ => {}
            }
        }
        (kv, conv, recurrent)
    }

    fn orchestration_kv_row(
        loaded: &LoadedQwen36Layers,
        cache_slot: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let cache = loaded
            .layers()
            .iter()
            .find_map(|layer| match &layer.attn {
                AttnLayerBuffers::Full {
                    kv_cache: Some(cache),
                    ..
                } => Some(cache),
                _ => None,
            })
            .expect("full-attention cache");
        let row_elems = 128;
        let row_bytes = row_elems * 2;
        let start = cache_slot * row_bytes;
        let end = start + row_bytes;
        let k = cache
            .k
            .as_ref()
            .expect("dense K cache")
            .to_host_bytes()
            .unwrap();
        let v = cache
            .v
            .as_ref()
            .expect("dense V cache")
            .to_host_bytes()
            .unwrap();
        (bf16_values(&k[start..end]), bf16_values(&v[start..end]))
    }

    fn assert_finite_nonzero(label: &str, values: &[f32]) {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{label} contains non-finite values"
        );
        let max_abs = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs > 1e-7, "{label} is unexpectedly all zero");
    }

    fn assert_abs_rel_close(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        max_abs_limit: f32,
        rel_l2_limit: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert_finite_nonzero(&format!("{label} actual"), actual);
        assert_finite_nonzero(&format!("{label} expected"), expected);
        let mut diff_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut expected_max_abs = 0.0f32;
        for (&actual, &expected) in actual.iter().zip(expected) {
            let diff = (actual - expected).abs();
            max_abs = max_abs.max(diff);
            expected_max_abs = expected_max_abs.max(expected.abs());
            diff_sq += (diff as f64).powi(2);
            expected_sq += (expected as f64).powi(2);
        }
        let rel_l2 = diff_sq.sqrt() / expected_sq.sqrt().max(1e-12);
        let allowed_max_abs = max_abs_limit + rel_l2_limit as f32 * expected_max_abs;
        assert!(
            max_abs <= allowed_max_abs && rel_l2 <= rel_l2_limit,
            "{label} mismatch: max_abs={max_abs:.8} (combined limit {allowed_max_abs:.8}), \
             rel_l2={rel_l2:.8} (limit {rel_l2_limit})"
        );
    }

    fn orchestration_embedding_row(token: u32) -> Vec<u8> {
        test_bf16_bytes(128, |column| {
            (((token as usize * 17 + column * 3) % 31) as f32 - 15.0) * 0.002
        })
    }

    #[test]
    fn public_batched_prefill_native_int4_matches_pertoken_across_dense_and_split_chunks() {
        let _backend_lock = GPU_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(gpu_hal::current_backend(), Backend::Hip | Backend::Cuda) {
            eprintln!("skip: orchestration numerical gate requires HIP or CUDA");
            return;
        }

        let tmp = TestDir::new_orchestration();
        let store = BakedStore::open(tmp.path()).expect("open orchestration store");
        let geom = orchestration_geom();
        let config = orchestration_text_config();
        let load = || {
            load_qwen36_layers(
                &store,
                0,
                &geom,
                &config,
                "model.language_model",
                Qwen36WeightMode::Int4,
                514,
                false,
                false,
                Qwen36LayerLoadStrategy::Dense,
                &Qwen36LoadOptions::default(),
            )
            .expect("load complete native-INT4 orchestration owner")
        };
        let mut optimized = load();
        let mut ungrouped = load();
        let mut pertoken = load();
        assert_eq!(
            classify_layer_weight_encoding(optimized.layers()).unwrap(),
            Qwen36LayerWeightEncoding::NativeInt4
        );

        let tokens = (0..513).map(|step| (step % 127) as u32).collect::<Vec<_>>();
        let mut positions = (0..512)
            .map(|position| PositionPair::dense(position as i32))
            .collect::<Vec<_>>();
        positions.push(PositionPair::split(700, 512));

        let mut optimized_options = Qwen36ExecutionOptions::default();
        optimized_options.batched_prefill.attention = true;
        optimized_options.batched_prefill.grouped_ffn = true;
        optimized_options
            .batched_prefill
            .enable_unqualified_hip_native_int4 = true;
        optimized_options
            .diagnostics
            .capture_prefill_boundary_hidden = true;
        optimized_options.diagnostics.route_profile =
            kernel_ffi::qwen36_moe::Qwen36RouteProfileOptions {
                enabled: true,
                max_calls: 1_026,
            };
        kernel_ffi::qwen36_moe::qwen36_route_profile_reset();
        let mut optimized_token_calls = 0usize;
        let mut optimized_token_callback =
            |_loaded: &mut LoadedQwen36Layers, _step, _token, _position| {
                optimized_token_calls += 1;
                Ok(crate::qwen36_moe::prefill::PrefillTokenTimings::default())
            };
        let mut optimized_completions = Vec::new();
        let mut optimized_progress_callback =
            |_timings: &crate::qwen36_moe::prefill::BatchedPrefillTimings,
             total,
             completed,
             _elapsed| {
                assert_eq!(total, 513);
                optimized_completions.push(completed);
            };
        let optimized_timings = run_batched_prefill(
            0,
            &geom,
            &store,
            "model.language_model",
            &mut optimized,
            &tokens,
            &positions,
            false,
            &optimized_options,
            Some(&mut optimized_token_callback),
            Some(&mut optimized_progress_callback),
        )
        .expect("optimized public batched prefill");
        assert_eq!(optimized_timings.chunks, 2);
        assert_eq!(optimized_timings.tokens, 513);
        assert_eq!(optimized_token_calls, 0);
        assert_eq!(optimized_completions, vec![512, 513]);
        assert_eq!(
            optimized_timings
                .boundary_hidden
                .iter()
                .map(|(completed, _)| *completed)
                .collect::<Vec<_>>(),
            vec![512, 513]
        );

        let routes = kernel_ffi::qwen36_moe::qwen36_route_profile_snapshot();
        assert_eq!(routes.calls, 1_026);
        assert_eq!(routes.dropped_calls, 0);
        assert!(routes.route_calls.iter().all(|call| {
            call.experts.len() == 2
                && call.experts[0] != call.experts[1]
                && call.experts.iter().all(|&expert| expert < 4)
        }));
        let mut routed_experts = routes
            .route_calls
            .iter()
            .flat_map(|call| call.experts.iter().copied())
            .collect::<Vec<_>>();
        routed_experts.sort_unstable();
        routed_experts.dedup();
        assert!(
            routed_experts.len() >= 3,
            "fixture must exercise at least three routed experts, got {routed_experts:?}"
        );

        let mut ungrouped_options = optimized_options.clone();
        ungrouped_options.batched_prefill.grouped_ffn = false;
        ungrouped_options.diagnostics.route_profile =
            kernel_ffi::qwen36_moe::Qwen36RouteProfileOptions::default();
        let ungrouped_timings = run_batched_prefill(
            0,
            &geom,
            &store,
            "model.language_model",
            &mut ungrouped,
            &tokens,
            &positions,
            false,
            &ungrouped_options,
            None,
            None,
        )
        .expect("public batched-attention per-token-FFN reference");
        assert_eq!(ungrouped_timings.chunks, 2);
        assert_eq!(ungrouped_timings.tokens, 513);
        for ((completed, optimized_hidden), (_, ungrouped_hidden)) in optimized_timings
            .boundary_hidden
            .iter()
            .zip(&ungrouped_timings.boundary_hidden)
        {
            assert_abs_rel_close(
                &format!("grouped route/post-MoE boundary hidden at {completed}"),
                &bf16_values(optimized_hidden),
                &bf16_values(ungrouped_hidden),
                0.08,
                0.06,
            );
        }

        let mut pertoken_options = Qwen36ExecutionOptions::default();
        pertoken_options.batched_prefill.attention = false;
        pertoken_options.diagnostics.capture_prefill_boundary_hidden = true;
        let pertoken_timings = run_batched_prefill(
            0,
            &geom,
            &store,
            "model.language_model",
            &mut pertoken,
            &tokens,
            &positions,
            false,
            &pertoken_options,
            None,
            None,
        )
        .expect("public per-token runtime reference");
        assert_eq!(pertoken_timings.chunks, 2);
        assert_eq!(pertoken_timings.tokens, 513);
        assert_eq!(
            pertoken_timings
                .boundary_hidden
                .iter()
                .map(|(completed, _)| *completed)
                .collect::<Vec<_>>(),
            vec![512, 513]
        );

        let optimized_state = orchestration_state(&optimized);
        let pertoken_state = orchestration_state(&pertoken);
        for (label, lhs, rhs, max_abs, rel_l2) in [
            (
                "linear conv",
                &optimized_state.1,
                &pertoken_state.1,
                0.08,
                0.08,
            ),
            (
                "linear recurrent",
                &optimized_state.2,
                &pertoken_state.2,
                0.08,
                0.30,
            ),
        ] {
            assert_abs_rel_close(label, lhs, rhs, max_abs, rel_l2);
        }

        let (optimized_k, optimized_v) = orchestration_kv_row(&optimized, 512);
        let (pertoken_k, pertoken_v) = orchestration_kv_row(&pertoken, 512);
        assert_abs_rel_close("split cache K row", &optimized_k, &pertoken_k, 0.08, 0.08);
        assert_abs_rel_close("split cache V row", &optimized_v, &pertoken_v, 0.08, 0.08);

        for ((completed, optimized_hidden), (_, pertoken_hidden)) in optimized_timings
            .boundary_hidden
            .iter()
            .zip(&pertoken_timings.boundary_hidden)
        {
            assert_abs_rel_close(
                &format!("post-MoE boundary hidden at {completed}"),
                &bf16_values(optimized_hidden),
                &bf16_values(pertoken_hidden),
                0.12,
                0.08,
            );
        }

        let next_hidden = orchestration_embedding_row(3);
        let next_position = PositionPair::split(701, 513);
        let optimized_output = run_chain_step(Qwen36ChainStep {
            ordinal: 0,
            geom: &geom,
            loaded_layers: &mut optimized,
            initial_hidden: &next_hidden,
            position: next_position,
            step: 513,
            accurate_stage_timings: false,
            fold: None,
            download_final_hidden: true,
            expert_prefetch: None,
            execution: &optimized_options,
        })
        .expect("decode after optimized prefill");
        let pertoken_output = run_chain_step(Qwen36ChainStep {
            ordinal: 0,
            geom: &geom,
            loaded_layers: &mut pertoken,
            initial_hidden: &next_hidden,
            position: next_position,
            step: 513,
            accurate_stage_timings: false,
            fold: None,
            download_final_hidden: true,
            expert_prefetch: None,
            execution: &pertoken_options,
        })
        .expect("decode after per-token prefill");
        assert_abs_rel_close(
            "post-prefill split decode hidden",
            &bf16_values(&optimized_output.outputs.final_hidden_bytes),
            &bf16_values(&pertoken_output.outputs.final_hidden_bytes),
            0.12,
            0.08,
        );
    }
}
