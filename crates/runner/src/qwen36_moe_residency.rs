//! Sparse VMM residency policy for Qwen3.6-MoE routed expert slabs.
//!
//! The decode descriptors need stable base pointers for the fused
//! `experts.gate_up_proj` and `experts.down_proj` tensors. This manager
//! reserves those full virtual address ranges up front, but only uploads and
//! maps the expert slices that are explicitly pinned resident.

use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{ScalarType, VirtualAllocationRole, VirtualArena, VirtualBacking};
use model_store::BakedStore;

use crate::qwen36_moe_decode::ResidentWeight;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeExpertProjection {
    GateUp,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeExpertKey {
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub projection: MoeExpertProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeExpertResidencyConfig {
    /// Maximum number of logical expert slices marked resident at once.
    ///
    /// HIP maps at the VMM page granularity, so resident physical bytes can be
    /// wider than this logical count implies. The policy tracks the logical
    /// slices because the router naturally speaks in `(layer, expert)` terms.
    pub max_resident_slices: usize,
}

impl MoeExpertResidencyConfig {
    pub fn new(max_resident_slices: usize) -> Result<Self> {
        if max_resident_slices == 0 {
            return Err(anyhow!("max_resident_slices must be > 0"));
        }
        Ok(Self {
            max_resident_slices,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoeExpertResidencyStats {
    pub registered_tensors: usize,
    pub resident_slices: usize,
    pub hits: u64,
    pub misses: u64,
    pub evicted_slices: u64,
    pub uploaded_bytes: usize,
    pub unmapped_bytes: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MoeExpertTensorReservation {
    pub allocation_id: usize,
    pub ptr: *const c_void,
    pub dtype: ScalarType,
    pub shape: Vec<usize>,
    pub len_bytes: usize,
    pub expert_count: usize,
    pub expert_bytes: usize,
}

#[derive(Debug, Clone)]
struct ExpertTensor {
    name: String,
    allocation_id: usize,
    ptr: *const c_void,
    dtype: ScalarType,
    shape: Vec<usize>,
    len_bytes: usize,
    expert_count: usize,
    expert_bytes: usize,
    page_bytes: usize,
}

#[derive(Debug, Clone)]
struct ResidentSlice {
    tensor_idx: usize,
    logical_offset: usize,
    logical_len: usize,
    page_offset: usize,
    page_len: usize,
    last_used: u64,
}

pub struct MoeExpertResidencyManager {
    arena: VirtualArena,
    config: MoeExpertResidencyConfig,
    tensors: Vec<ExpertTensor>,
    tensor_by_layer_projection: HashMap<(usize, MoeExpertProjection), usize>,
    resident: HashMap<MoeExpertKey, ResidentSlice>,
    clock: u64,
    hits: u64,
    misses: u64,
    evicted_slices: u64,
    uploaded_bytes: usize,
    unmapped_bytes: usize,
}

impl MoeExpertResidencyManager {
    pub fn new(device_ordinal: usize, config: MoeExpertResidencyConfig) -> Self {
        Self {
            arena: VirtualArena::new(device_ordinal, VirtualBacking::Discard),
            config,
            tensors: Vec::new(),
            tensor_by_layer_projection: HashMap::new(),
            resident: HashMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            evicted_slices: 0,
            uploaded_bytes: 0,
            unmapped_bytes: 0,
        }
    }

    pub fn arena(&self) -> &VirtualArena {
        &self.arena
    }

    #[allow(dead_code)]
    pub fn arena_mut(&mut self) -> &mut VirtualArena {
        &mut self.arena
    }

    pub fn stats(&self) -> MoeExpertResidencyStats {
        MoeExpertResidencyStats {
            registered_tensors: self.tensors.len(),
            resident_slices: self.resident.len(),
            hits: self.hits,
            misses: self.misses,
            evicted_slices: self.evicted_slices,
            uploaded_bytes: self.uploaded_bytes,
            unmapped_bytes: self.unmapped_bytes,
        }
    }

    pub fn register_tensor(
        &mut self,
        store: &BakedStore,
        layer_idx: usize,
        projection: MoeExpertProjection,
        name: &str,
        expert_count: usize,
    ) -> Result<MoeExpertTensorReservation> {
        if expert_count == 0 {
            return Err(anyhow!("expert_count must be > 0 for {name}"));
        }
        if self
            .tensor_by_layer_projection
            .contains_key(&(layer_idx, projection))
        {
            return Err(anyhow!(
                "duplicate MoE expert tensor registration for layer {layer_idx} {projection:?}"
            ));
        }

        let allocation_id = store
            .reserve_virtual_arena(&mut self.arena, name, VirtualAllocationRole::MoeExpert)
            .with_context(|| format!("reserve virtual MoE expert tensor {name}"))?;
        let allocation = self
            .arena
            .allocation(allocation_id)
            .ok_or_else(|| anyhow!("virtual allocation id {allocation_id} missing"))?;
        let buffer = allocation.buffer();
        let len_bytes = buffer.len_bytes();
        if len_bytes % expert_count != 0 {
            return Err(anyhow!(
                "MoE expert tensor {name} has {len_bytes} bytes, not divisible by {expert_count} experts"
            ));
        }

        let tensor = ExpertTensor {
            name: name.to_string(),
            allocation_id,
            ptr: buffer.as_ptr(),
            dtype: buffer.dtype(),
            shape: buffer.shape().to_vec(),
            len_bytes,
            expert_count,
            expert_bytes: len_bytes / expert_count,
            page_bytes: buffer.granularity(),
        };
        let reservation = tensor.reservation();
        let tensor_idx = self.tensors.len();
        self.tensors.push(tensor);
        self.tensor_by_layer_projection
            .insert((layer_idx, projection), tensor_idx);
        Ok(reservation)
    }

    pub fn resident_weight(
        &self,
        layer_idx: usize,
        projection: MoeExpertProjection,
    ) -> Result<ResidentWeight> {
        let tensor = self.tensor(layer_idx, projection)?;
        Ok(ResidentWeight::Virtual {
            allocation_id: tensor.allocation_id,
            ptr: tensor.ptr,
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            len_bytes: tensor.len_bytes,
        })
    }

    #[allow(dead_code)]
    pub fn is_resident(&self, key: MoeExpertKey) -> bool {
        self.resident.contains_key(&key)
    }

    pub fn ensure_resident(&mut self, store: &BakedStore, key: MoeExpertKey) -> Result<()> {
        let tensor_idx = self.tensor_idx(key.layer_idx, key.projection)?;
        let (name, allocation_id, logical_offset, logical_len, page_offset, page_len) = {
            let tensor = &self.tensors[tensor_idx];
            if key.expert_idx >= tensor.expert_count {
                return Err(anyhow!(
                    "expert {} out of range for layer {} {:?} (expert_count={})",
                    key.expert_idx,
                    key.layer_idx,
                    key.projection,
                    tensor.expert_count
                ));
            }
            let logical_offset = key.expert_idx * tensor.expert_bytes;
            let logical_len = tensor.expert_bytes;
            let (page_offset, page_len) =
                page_range(tensor.page_bytes, logical_offset, logical_len);
            (
                tensor.name.clone(),
                tensor.allocation_id,
                logical_offset,
                logical_len,
                page_offset,
                page_len,
            )
        };

        self.clock += 1;
        if let Some(entry) = self.resident.get_mut(&key) {
            entry.last_used = self.clock;
            self.hits += 1;
            return Ok(());
        }

        self.misses += 1;
        while self.resident.len() >= self.config.max_resident_slices {
            self.evict_lru_slice()?;
        }

        store
            .load_range_to_virtual_arena(
                &mut self.arena,
                allocation_id,
                &name,
                logical_offset,
                logical_len,
            )
            .with_context(|| {
                format!(
                    "load MoE expert slice layer={} expert={} projection={:?}",
                    key.layer_idx, key.expert_idx, key.projection
                )
            })?;
        self.uploaded_bytes += logical_len;
        self.resident.insert(
            key,
            ResidentSlice {
                tensor_idx,
                logical_offset,
                logical_len,
                page_offset,
                page_len,
                last_used: self.clock,
            },
        );
        Ok(())
    }

    fn evict_lru_slice(&mut self) -> Result<()> {
        let Some((victim, entry)) = self
            .resident
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, entry)| (*key, entry.clone()))
        else {
            return Ok(());
        };

        let tensor = &self.tensors[entry.tensor_idx];
        let allocation = self
            .arena
            .allocation_mut(tensor.allocation_id)
            .ok_or_else(|| anyhow!("virtual allocation id {} missing", tensor.allocation_id))?;
        let removed_ranges = allocation
            .buffer_mut()
            .unmap_range_discard(entry.logical_offset, entry.logical_len)
            .with_context(|| {
                format!(
                    "evict MoE expert slice layer={} expert={} projection={:?}",
                    victim.layer_idx, victim.expert_idx, victim.projection
                )
            })?;
        self.unmapped_bytes += removed_ranges.iter().map(|(_, len)| *len).sum::<usize>();
        if removed_ranges.is_empty() {
            self.resident.remove(&victim);
            self.evicted_slices += 1;
            return Ok(());
        }

        let before = self.resident.len();
        self.resident.retain(|_, resident| {
            resident.tensor_idx != entry.tensor_idx
                || !removed_ranges.iter().any(|(offset, len)| {
                    ranges_overlap(
                        resident.page_offset,
                        resident.page_offset + resident.page_len,
                        *offset,
                        *offset + *len,
                    )
                })
        });
        self.evicted_slices += (before - self.resident.len()) as u64;
        Ok(())
    }

    fn tensor(&self, layer_idx: usize, projection: MoeExpertProjection) -> Result<&ExpertTensor> {
        let idx = self.tensor_idx(layer_idx, projection)?;
        Ok(&self.tensors[idx])
    }

    fn tensor_idx(&self, layer_idx: usize, projection: MoeExpertProjection) -> Result<usize> {
        self.tensor_by_layer_projection
            .get(&(layer_idx, projection))
            .copied()
            .ok_or_else(|| {
                anyhow!("no MoE expert tensor registered for layer {layer_idx} {projection:?}")
            })
    }
}

impl ExpertTensor {
    fn reservation(&self) -> MoeExpertTensorReservation {
        MoeExpertTensorReservation {
            allocation_id: self.allocation_id,
            ptr: self.ptr,
            dtype: self.dtype,
            shape: self.shape.clone(),
            len_bytes: self.len_bytes,
            expert_count: self.expert_count,
            expert_bytes: self.expert_bytes,
        }
    }
}

fn page_range(page_bytes: usize, offset: usize, len: usize) -> (usize, usize) {
    let end = offset + len;
    let page_start = offset / page_bytes * page_bytes;
    let page_end = end.div_ceil(page_bytes) * page_bytes;
    (page_start, page_end - page_start)
}

fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_hal::{Backend, VirtualAllocationRole};
    use model_store::manifest::{LayoutTag, Manifest, TensorMeta, FORMAT_VERSION};

    fn synthetic_store(expert_count: usize, expert_bytes: usize) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bake_dir = tmp.path();
        let total_bytes = expert_count * expert_bytes;
        let mut weights = vec![0u8; total_bytes];
        for expert in 0..expert_count {
            let fill = (expert as u8).wrapping_add(1);
            weights[expert * expert_bytes..(expert + 1) * expert_bytes].fill(fill);
        }
        std::fs::write(model_store::weights_bin_path(bake_dir), &weights)
            .expect("write weights.bin");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            converter_version: 1,
            model_family: "test-qwen36-moe".to_string(),
            quant_profile: None,
            source_format: None,
            source_quant: None,
            tensors: vec![TensorMeta {
                name: "model.layers.0.mlp.experts.gate_up_proj".to_string(),
                shape: vec![total_bytes],
                dtype: "u8".to_string(),
                layout: LayoutTag::Int4Quantized,
                offset: 0,
                byte_len: total_bytes as u64,
            }],
        };
        std::fs::write(
            model_store::manifest_path(bake_dir),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        tmp
    }

    fn vmm_backends() -> [Backend; 2] {
        [Backend::Hip, Backend::Cuda]
    }

    fn with_supported_vmm_backend(test_name: &str, mut f: impl FnMut(Backend)) {
        for backend in vmm_backends() {
            gpu_hal::set_backend(backend);
            if !gpu_hal::vmm_is_supported(backend, 0) {
                eprintln!("skip: {backend} VMM unsupported on this device/runtime for {test_name}");
                continue;
            }
            f(backend);
        }
    }

    #[test]
    fn reserves_expert_slab_without_resident_pages() {
        with_supported_vmm_backend("reserves_expert_slab_without_resident_pages", |_backend| {
            let tmp = synthetic_store(4, 4096);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(2).unwrap());
            let reservation = manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    4,
                )
                .expect("register tensor");

            assert_eq!(reservation.expert_count, 4);
            assert_eq!(reservation.expert_bytes, 4096);
            let arena_stats = manager.arena().stats();
            assert_eq!(arena_stats.allocations, 1);
            assert_eq!(arena_stats.logical_bytes, 16 * 1024);
            assert_eq!(arena_stats.logical_resident_bytes, 0);
            assert_eq!(arena_stats.mapping_count, 0);
        });
    }

    #[test]
    fn lru_eviction_invalidates_page_overlapping_slices() {
        with_supported_vmm_backend(
            "lru_eviction_invalidates_page_overlapping_slices",
            |_backend| {
                let tmp = synthetic_store(4, 4096);
                let store = BakedStore::open(tmp.path()).expect("open synthetic store");
                let mut manager =
                    MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
                manager
                    .register_tensor(
                        &store,
                        0,
                        MoeExpertProjection::GateUp,
                        "model.layers.0.mlp.experts.gate_up_proj",
                        4,
                    )
                    .expect("register tensor");

                let e0 = MoeExpertKey {
                    layer_idx: 0,
                    expert_idx: 0,
                    projection: MoeExpertProjection::GateUp,
                };
                let e1 = MoeExpertKey {
                    layer_idx: 0,
                    expert_idx: 1,
                    projection: MoeExpertProjection::GateUp,
                };
                manager.ensure_resident(&store, e0).expect("load expert 0");
                assert!(manager.is_resident(e0));
                assert_eq!(manager.stats().resident_slices, 1);

                manager.ensure_resident(&store, e1).expect("load expert 1");
                assert!(!manager.is_resident(e0));
                assert!(manager.is_resident(e1));
                assert_eq!(manager.stats().resident_slices, 1);
                assert_eq!(manager.stats().misses, 2);
                assert_eq!(manager.stats().evicted_slices, 1);

                let allocation_id = manager
                    .resident_weight(0, MoeExpertProjection::GateUp)
                    .expect("resident weight")
                    .allocation_id()
                    .expect("virtual allocation");
                let bytes = manager
                    .arena()
                    .allocation(allocation_id)
                    .expect("allocation")
                    .buffer()
                    .to_host_range_bytes(4096, 4096)
                    .expect("read expert 1");
                assert!(bytes.iter().all(|b| *b == 2));
            },
        );
    }

    #[test]
    fn store_range_upload_keeps_stable_virtual_pointer() {
        with_supported_vmm_backend(
            "store_range_upload_keeps_stable_virtual_pointer",
            |_backend| {
                let tmp = synthetic_store(2, 4096);
                let store = BakedStore::open(tmp.path()).expect("open synthetic store");
                let mut arena = VirtualArena::new(0, VirtualBacking::Discard);
                let id = store
                    .reserve_virtual_arena(
                        &mut arena,
                        "model.layers.0.mlp.experts.gate_up_proj",
                        VirtualAllocationRole::MoeExpert,
                    )
                    .expect("reserve tensor");
                let ptr = arena.allocation(id).expect("allocation").buffer().as_ptr();
                store
                    .load_range_to_virtual_arena(
                        &mut arena,
                        id,
                        "model.layers.0.mlp.experts.gate_up_proj",
                        4096,
                        4096,
                    )
                    .expect("upload expert 1");
                let allocation = arena.allocation(id).expect("allocation after upload");
                assert_eq!(allocation.buffer().as_ptr(), ptr);
                let bytes = allocation
                    .buffer()
                    .to_host_range_bytes(4096, 4096)
                    .expect("read expert 1");
                assert!(bytes.iter().all(|b| *b == 2));
            },
        );
    }
}
