use anyhow::{anyhow, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType, VirtualArena};

use crate::qwen36_moe::decode::Qwen36ExecutionOptions;
use crate::qwen36_moe::layer_loader::Qwen36WeightMode;
use crate::qwen36_moe::persistent_decode::{PersistentScratch, CACHE_POS_INHERIT};
use crate::qwen36_moe::residency::MoeExpertResidencyManager;
use crate::qwen36_moe::types::{LayerBuffers, MultiLayerGeom};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentKvCapacity {
    pub(crate) layer_idx: usize,
    pub(crate) capacity: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PersistentPositionPlan {
    start_rope: i32,
    start_cache: i32,
    len: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct PersistentEmbeddingMetadata<'a> {
    pub(crate) backend: Backend,
    pub(crate) ordinal: usize,
    pub(crate) dtype: ScalarType,
    pub(crate) shape: &'a [usize],
    pub(crate) len_bytes: usize,
}

pub(crate) fn validate_persistent_position_plan(
    start_rope: i32,
    start_cache: i32,
    len: usize,
    full_attn_kv_capacities: &[PersistentKvCapacity],
) -> Result<PersistentPositionPlan> {
    if start_rope < 0 {
        anyhow::bail!("Qwen3.6 persistent RoPE position must be non-negative, got {start_rope}");
    }
    if start_cache != CACHE_POS_INHERIT && start_cache < 0 {
        anyhow::bail!(
            "Qwen3.6 persistent cache position must be non-negative or \
             CACHE_POS_INHERIT ({CACHE_POS_INHERIT}), got {start_cache}"
        );
    }
    let len_i32 = i32::try_from(len)
        .map_err(|_| anyhow!("Qwen3.6 persistent token count {len} exceeds i32::MAX"))?;
    let effective_cache = if start_cache == CACHE_POS_INHERIT {
        start_rope
    } else {
        start_cache
    };
    if len_i32 > 0 {
        let last_offset = len_i32 - 1;
        start_rope.checked_add(last_offset).ok_or_else(|| {
            anyhow!("Qwen3.6 persistent RoPE timeline overflows i32: start={start_rope} len={len}")
        })?;
        let last_cache = effective_cache.checked_add(last_offset).ok_or_else(|| {
            anyhow!(
                "Qwen3.6 persistent cache timeline overflows i32: \
                 start={effective_cache} len={len}"
            )
        })?;
        for kv in full_attn_kv_capacities {
            let layer_idx = kv.layer_idx;
            let capacity = kv.capacity;
            if capacity < 0 || last_cache >= capacity {
                anyhow::bail!(
                    "Qwen3.6 persistent KV capacity exceeded at full-attention layer \
                     {layer_idx}: last cache slot {last_cache}, capacity is {capacity}"
                );
            }
        }
    }
    Ok(PersistentPositionPlan {
        start_rope,
        start_cache: effective_cache,
        len,
    })
}

pub(crate) fn validate_persistent_embedding_request(
    expected_backend: Backend,
    expected_ordinal: usize,
    hidden: usize,
    embedding: PersistentEmbeddingMetadata<'_>,
    tokens: &[u32],
) -> Result<()> {
    if embedding.backend != expected_backend {
        anyhow::bail!(
            "Qwen3.6 persistent embedding backend mismatch: got {:?}, expected {expected_backend:?}",
            embedding.backend
        );
    }
    if embedding.ordinal != expected_ordinal {
        anyhow::bail!(
            "Qwen3.6 persistent embedding device ordinal mismatch: got {}, expected {expected_ordinal}",
            embedding.ordinal
        );
    }
    if embedding.dtype != ScalarType::BF16 {
        anyhow::bail!(
            "Qwen3.6 persistent embedding dtype mismatch: got {:?}, expected BF16",
            embedding.dtype
        );
    }
    let &[rows, row_width] = embedding.shape else {
        anyhow::bail!(
            "Qwen3.6 persistent embedding must have shape [vocab, hidden], got {:?}",
            embedding.shape
        );
    };
    if row_width != hidden {
        anyhow::bail!(
            "Qwen3.6 persistent embedding row width mismatch: got {row_width}, expected {hidden}"
        );
    }
    let required_bytes = rows
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow!("Qwen3.6 persistent embedding size overflow"))?;
    if embedding.len_bytes < required_bytes {
        anyhow::bail!(
            "Qwen3.6 persistent embedding buffer too small: got {} bytes, need {required_bytes}",
            embedding.len_bytes
        );
    }
    for (token_idx, &token) in tokens.iter().enumerate() {
        if token > i32::MAX as u32 {
            anyhow::bail!("Qwen3.6 persistent token {token} at index {token_idx} exceeds i32::MAX");
        }
        if token as usize >= rows {
            anyhow::bail!(
                "Qwen3.6 persistent token {token} at index {token_idx} is outside embedding \
                 vocab rows {rows}"
            );
        }
    }
    Ok(())
}

pub(crate) fn validate_sparse_prefetch_policy(
    sparse_owner: bool,
    has_prefetch_policy: bool,
) -> Result<()> {
    match (sparse_owner, has_prefetch_policy) {
        (true, false) => {
            anyhow::bail!("Qwen3.6 sparse expert residency requires an expert prefetch policy")
        }
        (false, true) => {
            anyhow::bail!("Qwen3.6 expert prefetch policy requires sparse expert residency")
        }
        _ => Ok(()),
    }
}

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

    #[cfg(test)]
    pub(crate) fn attach_test_sparse_expert_residency(
        &mut self,
        manager: MoeExpertResidencyManager,
    ) {
        self.moe_expert_residency = Some(manager);
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

    fn validate_persistent_owner_policy(&self) -> Result<()> {
        validate_sparse_prefetch_policy(self.has_sparse_expert_residency(), false)
    }

    pub fn run_dense_prefill_tokens_from_device_embedding(
        &mut self,
        ordinal: usize,
        embed_w: &GpuBuffer,
        tokens: &[u32],
        start_position: i32,
        start_cache_pos: i32,
    ) -> anyhow::Result<std::time::Duration> {
        self.validate_persistent_owner_policy()?;
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
        self.validate_persistent_owner_policy()?;
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
        execution: &Qwen36ExecutionOptions,
    ) -> anyhow::Result<crate::qwen36_moe::types::DecodeOutputs> {
        self.validate_persistent_owner_policy()?;
        self.persistent_scratch
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.6 persistent decode is not enabled"))?
            .run_segmented_profile(ordinal, initial_hidden, position, cache_pos, execution)
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

    #[test]
    fn persistent_position_plan_accepts_only_inherit_or_nonnegative_cache_positions() {
        assert_eq!(
            validate_persistent_position_plan(
                4,
                CACHE_POS_INHERIT,
                3,
                &[PersistentKvCapacity {
                    layer_idx: 3,
                    capacity: 8,
                }],
            )
            .expect("inherit cache timeline"),
            PersistentPositionPlan {
                start_rope: 4,
                start_cache: 4,
                len: 3,
            }
        );
        assert_eq!(
            validate_persistent_position_plan(
                7,
                2,
                2,
                &[PersistentKvCapacity {
                    layer_idx: 3,
                    capacity: 4,
                }],
            )
            .expect("split cache timeline"),
            PersistentPositionPlan {
                start_rope: 7,
                start_cache: 2,
                len: 2,
            }
        );

        for (rope, cache, len) in [(0, -2, 1), (-1, 0, 1), (i32::MAX, 0, 2)] {
            assert!(
                validate_persistent_position_plan(
                    rope,
                    cache,
                    len,
                    &[PersistentKvCapacity {
                        layer_idx: 3,
                        capacity: 8,
                    }],
                )
                .is_err(),
                "rope={rope} cache={cache} len={len}"
            );
        }
    }

    #[test]
    fn persistent_position_plan_checks_every_full_attention_cache() {
        let err = validate_persistent_position_plan(
            10,
            3,
            2,
            &[
                PersistentKvCapacity {
                    layer_idx: 3,
                    capacity: 8,
                },
                PersistentKvCapacity {
                    layer_idx: 7,
                    capacity: 4,
                },
            ],
        )
        .expect_err("second KV cache must reject slot 4");

        assert!(err.to_string().contains("layer 7"));
        assert!(err.to_string().contains("capacity is 4"));
    }

    #[test]
    fn persistent_embedding_contract_checks_backend_device_dtype_shape_and_tokens() {
        let valid = PersistentEmbeddingMetadata {
            backend: Backend::Hip,
            ordinal: 0,
            dtype: ScalarType::BF16,
            shape: &[16, 8],
            len_bytes: 16 * 8 * 2,
        };
        validate_persistent_embedding_request(Backend::Hip, 0, 8, valid, &[0, 15])
            .expect("valid embedding request");

        let cases = [
            PersistentEmbeddingMetadata {
                backend: Backend::Metal,
                ..valid
            },
            PersistentEmbeddingMetadata {
                ordinal: 1,
                ..valid
            },
            PersistentEmbeddingMetadata {
                dtype: ScalarType::F32,
                ..valid
            },
            PersistentEmbeddingMetadata {
                shape: &[16, 7],
                ..valid
            },
            PersistentEmbeddingMetadata {
                len_bytes: 15 * 8 * 2,
                ..valid
            },
        ];
        for metadata in cases {
            assert!(
                validate_persistent_embedding_request(Backend::Hip, 0, 8, metadata, &[0]).is_err()
            );
        }
        assert!(validate_persistent_embedding_request(Backend::Hip, 0, 8, valid, &[16]).is_err());
        assert!(
            validate_persistent_embedding_request(Backend::Hip, 0, 8, valid, &[u32::MAX]).is_err()
        );
    }

    #[test]
    fn sparse_owner_requires_exactly_one_prefetch_policy() {
        validate_sparse_prefetch_policy(false, false).expect("dense owner without callback");
        validate_sparse_prefetch_policy(true, true).expect("sparse owner with callback");

        assert!(validate_sparse_prefetch_policy(true, false)
            .expect_err("sparse owner without callback must fail")
            .to_string()
            .contains("requires an expert prefetch policy"));
        assert!(validate_sparse_prefetch_policy(false, true)
            .expect_err("dense owner with callback must fail")
            .to_string()
            .contains("requires sparse expert residency"));
    }
}
