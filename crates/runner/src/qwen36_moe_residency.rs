//! Sparse VMM residency policy for Qwen3.6-MoE routed expert slabs.
//!
//! The decode descriptors need stable base pointers for the fused
//! `experts.gate_up_proj` and `experts.down_proj` tensors. This manager
//! reserves those full virtual address ranges up front, but only uploads and
//! maps the expert pages that are explicitly pinned resident.

use std::collections::{HashMap, HashSet};
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
    /// Maximum number of VMM backing pages resident at once.
    ///
    /// The environment knob is still expressed in experts because the router
    /// naturally speaks in `(layer, expert)` terms, but physical residency is
    /// page-granular on HIP/CUDA. The engine converts that expert cap into this
    /// page budget conservatively.
    pub max_resident_pages: usize,
}

impl MoeExpertResidencyConfig {
    pub fn new(max_resident_pages: usize) -> Result<Self> {
        if max_resident_pages == 0 {
            return Err(anyhow!("max_resident_pages must be > 0"));
        }
        Ok(Self { max_resident_pages })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoeExpertResidencyStats {
    pub registered_tensors: usize,
    pub resident_slices: usize,
    pub resident_pages: usize,
    pub page_backed_slices: usize,
    pub hits: u64,
    pub misses: u64,
    pub page_hits: u64,
    pub page_misses: u64,
    pub evicted_slices: u64,
    pub evicted_pages: u64,
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
    page_offset: usize,
    page_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResidentPageKey {
    tensor_idx: usize,
    page_offset: usize,
}

#[derive(Debug, Clone)]
struct ResidentPage {
    tensor_idx: usize,
    page_offset: usize,
    page_len: usize,
    last_used: u64,
}

#[derive(Debug, Clone, Copy)]
struct PageSpan {
    offset: usize,
    len: usize,
    copy_len: usize,
}

pub struct MoeExpertResidencyManager {
    arena: VirtualArena,
    config: MoeExpertResidencyConfig,
    tensors: Vec<ExpertTensor>,
    tensor_by_layer_projection: HashMap<(usize, MoeExpertProjection), usize>,
    resident: HashMap<MoeExpertKey, ResidentSlice>,
    resident_pages: HashMap<ResidentPageKey, ResidentPage>,
    clock: u64,
    hits: u64,
    misses: u64,
    page_hits: u64,
    page_misses: u64,
    evicted_slices: u64,
    evicted_pages: u64,
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
            resident_pages: HashMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            page_hits: 0,
            page_misses: 0,
            evicted_slices: 0,
            evicted_pages: 0,
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
            resident_pages: self.resident_pages.len(),
            page_backed_slices: self.page_backed_slices(),
            hits: self.hits,
            misses: self.misses,
            page_hits: self.page_hits,
            page_misses: self.page_misses,
            evicted_slices: self.evicted_slices,
            evicted_pages: self.evicted_pages,
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
        let (name, allocation_id, logical_offset, logical_len, pages) = {
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
            let pages = page_spans(
                tensor.page_bytes,
                logical_offset,
                logical_len,
                tensor.len_bytes,
            );
            (
                tensor.name.clone(),
                tensor.allocation_id,
                logical_offset,
                logical_len,
                pages,
            )
        };
        if pages.len() > self.config.max_resident_pages {
            return Err(anyhow!(
                "MoE expert slice layer={} expert={} projection={:?} spans {} VMM pages, \
                 exceeding max_resident_pages={}",
                key.layer_idx,
                key.expert_idx,
                key.projection,
                pages.len(),
                self.config.max_resident_pages
            ));
        }
        let slice_page_offset = pages
            .first()
            .map(|page| page.offset)
            .unwrap_or(logical_offset);
        let slice_page_end = pages
            .last()
            .map(|page| page.offset + page.len)
            .unwrap_or(logical_offset + logical_len);

        self.clock += 1;
        if self.resident.contains_key(&key) {
            let all_pages_resident = pages.iter().all(|span| {
                self.resident_pages.contains_key(&ResidentPageKey {
                    tensor_idx,
                    page_offset: span.offset,
                })
            });
            if all_pages_resident {
                self.hits += 1;
                for span in &pages {
                    self.page_hits += 1;
                    if let Some(page) = self.resident_pages.get_mut(&ResidentPageKey {
                        tensor_idx,
                        page_offset: span.offset,
                    }) {
                        page.last_used = self.clock;
                    }
                }
                return Ok(());
            }
            self.resident.remove(&key);
        }

        self.misses += 1;
        let mut missing_pages = Vec::new();
        for span in &pages {
            let page_key = ResidentPageKey {
                tensor_idx,
                page_offset: span.offset,
            };
            if let Some(page) = self.resident_pages.get_mut(&page_key) {
                page.last_used = self.clock;
                self.page_hits += 1;
            } else {
                missing_pages.push(*span);
            }
        }

        self.page_misses += missing_pages.len() as u64;
        for span in missing_pages {
            while self.resident_pages.len() >= self.config.max_resident_pages {
                self.evict_lru_page()?;
            }

            store
                .load_range_to_virtual_arena(
                    &mut self.arena,
                    allocation_id,
                    &name,
                    span.offset,
                    span.copy_len,
                )
                .with_context(|| {
                    format!(
                        "load MoE expert page layer={} expert={} projection={:?} offset={} len={}",
                        key.layer_idx, key.expert_idx, key.projection, span.offset, span.copy_len
                    )
                })?;
            self.uploaded_bytes += span.copy_len;
            self.resident_pages.insert(
                ResidentPageKey {
                    tensor_idx,
                    page_offset: span.offset,
                },
                ResidentPage {
                    tensor_idx,
                    page_offset: span.offset,
                    page_len: span.len,
                    last_used: self.clock,
                },
            );
        }
        self.resident.insert(
            key,
            ResidentSlice {
                tensor_idx,
                page_offset: slice_page_offset,
                page_len: slice_page_end - slice_page_offset,
            },
        );
        Ok(())
    }

    fn evict_lru_page(&mut self) -> Result<()> {
        let Some((victim, page)) = self
            .resident_pages
            .iter()
            .min_by_key(|(_, page)| page.last_used)
            .map(|(key, page)| (*key, page.clone()))
        else {
            return Ok(());
        };

        let tensor = &self.tensors[page.tensor_idx];
        let allocation = self
            .arena
            .allocation_mut(tensor.allocation_id)
            .ok_or_else(|| anyhow!("virtual allocation id {} missing", tensor.allocation_id))?;
        let removed_ranges = allocation
            .buffer_mut()
            .unmap_range_discard(page.page_offset, page.page_len)
            .with_context(|| {
                format!(
                    "evict MoE expert page tensor={} offset={} len={}",
                    tensor.name, page.page_offset, page.page_len
                )
            })?;
        self.unmapped_bytes += removed_ranges.iter().map(|(_, len)| *len).sum::<usize>();
        if removed_ranges.is_empty() {
            self.resident_pages.remove(&victim);
            let removed_slices = self.remove_resident_slices_overlapping(
                page.tensor_idx,
                page.page_offset,
                page.page_offset + page.page_len,
            );
            self.evicted_pages += 1;
            self.evicted_slices += removed_slices as u64;
            return Ok(());
        }

        let removed_pages =
            self.remove_resident_pages_overlapping(page.tensor_idx, &removed_ranges);
        let removed_slices =
            self.remove_resident_slices_overlapping_ranges(page.tensor_idx, &removed_ranges);
        self.evicted_pages += removed_pages as u64;
        self.evicted_slices += removed_slices as u64;
        Ok(())
    }

    fn remove_resident_pages_overlapping(
        &mut self,
        tensor_idx: usize,
        ranges: &[(usize, usize)],
    ) -> usize {
        let before = self.resident_pages.len();
        self.resident_pages.retain(|_, page| {
            page.tensor_idx != tensor_idx
                || !ranges.iter().any(|(offset, len)| {
                    ranges_overlap(
                        page.page_offset,
                        page.page_offset + page.page_len,
                        *offset,
                        *offset + *len,
                    )
                })
        });
        before - self.resident_pages.len()
    }

    fn remove_resident_slices_overlapping_ranges(
        &mut self,
        tensor_idx: usize,
        ranges: &[(usize, usize)],
    ) -> usize {
        let before = self.resident.len();
        self.resident.retain(|_, resident| {
            resident.tensor_idx != tensor_idx
                || !ranges.iter().any(|(offset, len)| {
                    ranges_overlap(
                        resident.page_offset,
                        resident.page_offset + resident.page_len,
                        *offset,
                        *offset + *len,
                    )
                })
        });
        before - self.resident.len()
    }

    fn remove_resident_slices_overlapping(
        &mut self,
        tensor_idx: usize,
        offset: usize,
        end: usize,
    ) -> usize {
        let before = self.resident.len();
        self.resident.retain(|_, resident| {
            resident.tensor_idx != tensor_idx
                || !ranges_overlap(
                    resident.page_offset,
                    resident.page_offset + resident.page_len,
                    offset,
                    end,
                )
        });
        before - self.resident.len()
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

    fn page_backed_slices(&self) -> usize {
        let mut backed = HashSet::new();
        for page in self.resident_pages.values() {
            let tensor = &self.tensors[page.tensor_idx];
            if tensor.expert_bytes == 0 || page.page_len == 0 {
                continue;
            }
            let page_end = page.page_offset + page.page_len;
            let first = page.page_offset / tensor.expert_bytes;
            let last = (page_end - 1) / tensor.expert_bytes;
            for expert_idx in first..=last.min(tensor.expert_count.saturating_sub(1)) {
                let offset = expert_idx * tensor.expert_bytes;
                let end = offset + tensor.expert_bytes;
                if ranges_overlap(offset, end, page.page_offset, page_end) {
                    backed.insert((page.tensor_idx, expert_idx));
                }
            }
        }
        backed.len()
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

fn page_spans(page_bytes: usize, offset: usize, len: usize, total_len: usize) -> Vec<PageSpan> {
    let end = offset + len;
    let mut cursor = offset / page_bytes * page_bytes;
    let page_end = end.div_ceil(page_bytes) * page_bytes;
    let mut spans = Vec::new();
    while cursor < page_end {
        let len = page_bytes.min(page_end - cursor);
        let copy_len = len.min(total_len.saturating_sub(cursor));
        spans.push(PageSpan {
            offset: cursor,
            len,
            copy_len,
        });
        cursor += len;
    }
    spans
}

fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_hal::{Backend, VirtualAllocationRole};
    use model_store::manifest::{LayoutTag, Manifest, TensorMeta, FORMAT_VERSION};
    use std::sync::Mutex;

    static VMM_BACKEND_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct BackendRestore(Backend);

    impl Drop for BackendRestore {
        fn drop(&mut self) {
            gpu_hal::set_backend(self.0);
        }
    }

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
        let _lock = VMM_BACKEND_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _restore_backend = BackendRestore(gpu_hal::current_backend());

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
    fn shared_page_reuses_backing_for_neighbor_slice() {
        with_supported_vmm_backend(
            "shared_page_reuses_backing_for_neighbor_slice",
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
                assert_eq!(manager.stats().resident_pages, 1);
                assert_eq!(manager.stats().page_misses, 1);

                manager.ensure_resident(&store, e1).expect("load expert 1");
                assert!(manager.is_resident(e0));
                assert!(manager.is_resident(e1));
                assert_eq!(manager.stats().resident_slices, 2);
                assert_eq!(manager.stats().resident_pages, 1);
                assert!(manager.stats().page_backed_slices >= 2);
                assert_eq!(manager.stats().misses, 2);
                assert_eq!(manager.stats().page_hits, 1);
                assert_eq!(manager.stats().page_misses, 1);
                assert_eq!(manager.stats().evicted_pages, 0);
                assert_eq!(manager.stats().evicted_slices, 0);

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
    fn lru_eviction_invalidates_whole_pages() {
        with_supported_vmm_backend("lru_eviction_invalidates_whole_pages", |_backend| {
            let probe_tmp = synthetic_store(1, 4096);
            let probe_store = BakedStore::open(probe_tmp.path()).expect("open probe store");
            let mut probe =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            probe
                .register_tensor(
                    &probe_store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    1,
                )
                .expect("register probe tensor");
            let expert_bytes = probe.tensors[0].page_bytes;

            let tmp = synthetic_store(3, expert_bytes);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    3,
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
            assert_eq!(manager.stats().resident_pages, 1);

            manager.ensure_resident(&store, e1).expect("load expert 1");
            assert!(!manager.is_resident(e0));
            assert!(manager.is_resident(e1));
            assert_eq!(manager.stats().resident_pages, 1);
            assert_eq!(manager.stats().page_misses, 2);
            assert!(manager.stats().evicted_pages >= 1);
            assert!(manager.stats().evicted_slices >= 1);

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
                .to_host_range_bytes(expert_bytes, 4096)
                .expect("read expert 1");
            assert!(bytes.iter().all(|b| *b == 2));
        });
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
