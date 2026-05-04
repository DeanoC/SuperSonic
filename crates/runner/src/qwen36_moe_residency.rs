//! Sparse VMM residency policy for Qwen3.6-MoE routed expert slabs.
//!
//! The decode descriptors need stable base pointers for the fused
//! `experts.gate_up_proj` and `experts.down_proj` tensors. This manager
//! reserves those full virtual address ranges up front, but only uploads and
//! maps the expert pages that are explicitly pinned resident.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{
    copy_h2d_async, memset_zeros_async, Backend, GpuEvent, GpuStream, PinnedHostBuffer, ScalarType,
    VirtualAllocationRole, VirtualArena, VirtualBacking,
};
use model_store::BakedStore;

use crate::qwen36_moe_residency_pages::{
    oldest_protected_page, page_spans, prune_protected_pages, ranges_overlap,
    remove_pages_overlapping, remove_slices_overlapping, remove_slices_overlapping_ranges,
    select_lru_resident_page, AsyncPageIn, AsyncStagingSlot, PageSpan, PendingPage, ResidentPage,
    ResidentPageKey, ResidentSlice,
};
use crate::qwen36_moe_types::ResidentWeight;

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
    pub prefetch_requests: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub prefetch_page_hits: u64,
    pub prefetch_page_misses: u64,
    pub prefetch_skipped: u64,
    pub prefetch_skipped_pages: u64,
    pub prefetch_uploaded_bytes: usize,
    pub protected_pages: usize,
    pub protected_page_budget: usize,
    pub protect_requests: u64,
    pub protect_hits: u64,
    pub protect_misses: u64,
    pub protect_demotions: u64,
    pub protected_evicted_pages: u64,
    pub async_scheduled_pages: u64,
    pub async_completed_pages: u64,
    pub async_waited_pages: u64,
    pub async_skipped_no_slot: u64,
    pub async_skipped_no_capacity: u64,
    pub async_uploaded_bytes: usize,
    pub async_pending_pages_peak: usize,
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
    projection: MoeExpertProjection,
    allocation_id: usize,
    ptr: *const c_void,
    dtype: ScalarType,
    shape: Vec<usize>,
    len_bytes: usize,
    expert_count: usize,
    expert_bytes: usize,
    page_bytes: usize,
}

pub struct MoeExpertResidencyManager {
    arena: VirtualArena,
    config: MoeExpertResidencyConfig,
    tensors: Vec<ExpertTensor>,
    tensor_by_layer_projection: HashMap<(usize, MoeExpertProjection), usize>,
    resident: HashMap<MoeExpertKey, ResidentSlice>,
    resident_pages: HashMap<ResidentPageKey, ResidentPage>,
    protected_pages: HashMap<ResidentPageKey, u64>,
    max_protected_pages: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    page_hits: u64,
    page_misses: u64,
    evicted_slices: u64,
    evicted_pages: u64,
    uploaded_bytes: usize,
    unmapped_bytes: usize,
    prefetch_requests: u64,
    prefetch_hits: u64,
    prefetch_misses: u64,
    prefetch_page_hits: u64,
    prefetch_page_misses: u64,
    prefetch_skipped: u64,
    prefetch_skipped_pages: u64,
    prefetch_uploaded_bytes: usize,
    protect_requests: u64,
    protect_hits: u64,
    protect_misses: u64,
    protect_demotions: u64,
    protected_evicted_pages: u64,
    async_page_in: Option<AsyncPageIn>,
    pending_pages: HashMap<ResidentPageKey, PendingPage>,
    async_scheduled_pages: u64,
    async_completed_pages: u64,
    async_waited_pages: u64,
    async_skipped_no_slot: u64,
    async_skipped_no_capacity: u64,
    async_uploaded_bytes: usize,
    async_pending_pages_peak: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidencyAccessKind {
    Demand,
    Prefetch,
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
            protected_pages: HashMap::new(),
            max_protected_pages: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            page_hits: 0,
            page_misses: 0,
            evicted_slices: 0,
            evicted_pages: 0,
            uploaded_bytes: 0,
            unmapped_bytes: 0,
            prefetch_requests: 0,
            prefetch_hits: 0,
            prefetch_misses: 0,
            prefetch_page_hits: 0,
            prefetch_page_misses: 0,
            prefetch_skipped: 0,
            prefetch_skipped_pages: 0,
            prefetch_uploaded_bytes: 0,
            protect_requests: 0,
            protect_hits: 0,
            protect_misses: 0,
            protect_demotions: 0,
            protected_evicted_pages: 0,
            async_page_in: None,
            pending_pages: HashMap::new(),
            async_scheduled_pages: 0,
            async_completed_pages: 0,
            async_waited_pages: 0,
            async_skipped_no_slot: 0,
            async_skipped_no_capacity: 0,
            async_uploaded_bytes: 0,
            async_pending_pages_peak: 0,
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
            prefetch_requests: self.prefetch_requests,
            prefetch_hits: self.prefetch_hits,
            prefetch_misses: self.prefetch_misses,
            prefetch_page_hits: self.prefetch_page_hits,
            prefetch_page_misses: self.prefetch_page_misses,
            prefetch_skipped: self.prefetch_skipped,
            prefetch_skipped_pages: self.prefetch_skipped_pages,
            prefetch_uploaded_bytes: self.prefetch_uploaded_bytes,
            protected_pages: self.protected_pages.len(),
            protected_page_budget: self.max_protected_pages,
            protect_requests: self.protect_requests,
            protect_hits: self.protect_hits,
            protect_misses: self.protect_misses,
            protect_demotions: self.protect_demotions,
            protected_evicted_pages: self.protected_evicted_pages,
            async_scheduled_pages: self.async_scheduled_pages,
            async_completed_pages: self.async_completed_pages,
            async_waited_pages: self.async_waited_pages,
            async_skipped_no_slot: self.async_skipped_no_slot,
            async_skipped_no_capacity: self.async_skipped_no_capacity,
            async_uploaded_bytes: self.async_uploaded_bytes,
            async_pending_pages_peak: self.async_pending_pages_peak,
        }
    }

    pub fn enable_async_prefetch(&mut self, staging_pages: usize) -> Result<()> {
        if staging_pages == 0 {
            return Err(anyhow!("async MoE staging page count must be > 0"));
        }
        if gpu_hal::current_backend() != Backend::Hip {
            return Err(anyhow!(
                "SUPERSONIC_MOE_ISLAND_ASYNC_PREFETCH=1 is HIP-only in v1"
            ));
        }
        let page_bytes = self
            .tensors
            .iter()
            .map(|tensor| tensor.page_bytes)
            .max()
            .ok_or_else(|| anyhow!("async MoE prefetch requires registered expert tensors"))?;
        let stream = GpuStream::new_nonblocking(self.arena.device_ordinal())
            .context("create MoE async prefetch stream")?;
        let mut slots = Vec::with_capacity(staging_pages);
        for _ in 0..staging_pages {
            slots.push(AsyncStagingSlot {
                buffer: PinnedHostBuffer::new(self.arena.device_ordinal(), page_bytes)
                    .context("allocate MoE async pinned staging page")?,
                event: GpuEvent::new(self.arena.device_ordinal())
                    .context("create MoE async prefetch event")?,
                pending: None,
            });
        }
        self.async_page_in = Some(AsyncPageIn { stream, slots });
        Ok(())
    }

    pub fn max_resident_pages(&self) -> usize {
        self.config.max_resident_pages
    }

    pub fn set_max_resident_pages(&mut self, max_resident_pages: usize) -> Result<()> {
        self.config = MoeExpertResidencyConfig::new(max_resident_pages)?;
        while self.resident_pages.len() + self.pending_pages.len() > self.config.max_resident_pages
        {
            self.evict_lru_page()?;
        }
        self.clamp_protected_pages();
        Ok(())
    }

    pub fn max_protected_pages(&self) -> usize {
        self.max_protected_pages
    }

    pub fn set_max_protected_pages(&mut self, max_protected_pages: usize) {
        self.max_protected_pages = max_protected_pages.min(self.config.max_resident_pages);
        self.clamp_protected_pages();
    }

    pub fn page_budget_for_routed_experts(&self, routed_experts: usize) -> Result<usize> {
        if routed_experts == 0 {
            return Err(anyhow!("routed_experts must be > 0"));
        }
        let mut pages_by_projection: HashMap<MoeExpertProjection, usize> = HashMap::new();
        for tensor in &self.tensors {
            let pages_per_slice = tensor.max_pages_per_expert_slice();
            pages_by_projection
                .entry(tensor.projection)
                .and_modify(|current| *current = (*current).max(pages_per_slice))
                .or_insert(pages_per_slice);
        }
        let pages_per_routed_expert = pages_by_projection
            .values()
            .try_fold(0usize, |acc, pages| acc.checked_add(*pages))
            .ok_or_else(|| anyhow!("routed expert page footprint overflows"))?;
        if pages_per_routed_expert == 0 {
            return Err(anyhow!(
                "cannot derive sparse MoE page budget before registering expert tensors"
            ));
        }
        routed_experts
            .checked_mul(pages_per_routed_expert)
            .ok_or_else(|| {
                anyhow!(
                    "sparse MoE page budget overflows: routed_experts={routed_experts} \
                     pages_per_routed_expert={pages_per_routed_expert}"
                )
            })
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
            projection,
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
        self.ensure_resident_with_kind(store, key, ResidencyAccessKind::Demand)
    }

    pub fn prefetch_resident(&mut self, store: &BakedStore, key: MoeExpertKey) -> Result<()> {
        self.ensure_resident_with_kind(store, key, ResidencyAccessKind::Prefetch)
    }

    pub fn protect_resident(&mut self, key: MoeExpertKey) -> Result<()> {
        self.protect_requests += 1;
        if self.max_protected_pages == 0 {
            self.protect_misses += 1;
            return Ok(());
        }
        let Some(resident) = self.resident.get(&key).cloned() else {
            self.protect_misses += 1;
            return Ok(());
        };
        let tensor = &self.tensors[resident.tensor_idx];
        let pages = page_spans(
            tensor.page_bytes,
            resident.page_offset,
            resident.page_len,
            tensor.len_bytes,
        );
        let mut protected = 0usize;
        self.clock += 1;
        for span in pages {
            let page_key = ResidentPageKey {
                tensor_idx: resident.tensor_idx,
                page_offset: span.offset,
            };
            if self.resident_pages.contains_key(&page_key) {
                self.protected_pages.insert(page_key, self.clock);
                protected += 1;
            }
        }
        if protected == 0 {
            self.protect_misses += 1;
        } else {
            self.protect_hits += 1;
        }
        self.clamp_protected_pages();
        Ok(())
    }

    fn ensure_resident_with_kind(
        &mut self,
        store: &BakedStore,
        key: MoeExpertKey,
        kind: ResidencyAccessKind,
    ) -> Result<()> {
        if kind == ResidencyAccessKind::Prefetch {
            self.prefetch_requests += 1;
        }
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

        self.promote_completed_pending_pages()?;
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
                if kind == ResidencyAccessKind::Prefetch {
                    self.prefetch_hits += 1;
                }
                for span in &pages {
                    self.page_hits += 1;
                    if kind == ResidencyAccessKind::Prefetch {
                        self.prefetch_page_hits += 1;
                    }
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
        if kind == ResidencyAccessKind::Prefetch {
            self.prefetch_misses += 1;
        }
        let mut missing_pages = Vec::new();
        for span in &pages {
            let page_key = ResidentPageKey {
                tensor_idx,
                page_offset: span.offset,
            };
            if let Some(page) = self.resident_pages.get_mut(&page_key) {
                page.last_used = self.clock;
                self.page_hits += 1;
                if kind == ResidencyAccessKind::Prefetch {
                    self.prefetch_page_hits += 1;
                }
            } else {
                let page_key = ResidentPageKey {
                    tensor_idx,
                    page_offset: span.offset,
                };
                if self.pending_pages.contains_key(&page_key) {
                    if kind == ResidencyAccessKind::Demand {
                        self.wait_pending_page(page_key)?;
                    }
                    self.page_hits += 1;
                    if kind == ResidencyAccessKind::Prefetch {
                        self.prefetch_page_hits += 1;
                    }
                } else {
                    missing_pages.push(*span);
                }
            }
        }

        if kind == ResidencyAccessKind::Prefetch {
            let free_pages = self
                .config
                .max_resident_pages
                .saturating_sub(self.resident_pages.len() + self.pending_pages.len());
            if missing_pages.len() > free_pages {
                self.prefetch_skipped += 1;
                self.prefetch_skipped_pages += missing_pages.len() as u64;
                if self.async_page_in.is_some() {
                    self.async_skipped_no_capacity += missing_pages.len() as u64;
                }
                return Ok(());
            }
        }
        self.page_misses += missing_pages.len() as u64;
        if kind == ResidencyAccessKind::Prefetch {
            self.prefetch_page_misses += missing_pages.len() as u64;
        }
        for span in missing_pages {
            if kind == ResidencyAccessKind::Prefetch
                && self.schedule_async_page(store, tensor_idx, allocation_id, &name, span)?
            {
                continue;
            }
            while self.resident_pages.len() + self.pending_pages.len()
                >= self.config.max_resident_pages
            {
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
            if kind == ResidencyAccessKind::Prefetch {
                self.prefetch_uploaded_bytes += span.copy_len;
            }
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
        if pages.iter().all(|span| {
            self.resident_pages.contains_key(&ResidentPageKey {
                tensor_idx,
                page_offset: span.offset,
            })
        }) {
            self.resident.insert(
                key,
                ResidentSlice {
                    tensor_idx,
                    page_offset: slice_page_offset,
                    page_len: slice_page_end - slice_page_offset,
                },
            );
        }
        Ok(())
    }

    fn schedule_async_page(
        &mut self,
        store: &BakedStore,
        tensor_idx: usize,
        allocation_id: usize,
        name: &str,
        span: PageSpan,
    ) -> Result<bool> {
        if self.async_page_in.is_none() {
            return Ok(false);
        }
        self.promote_completed_pending_pages()?;
        if self.resident_pages.len() + self.pending_pages.len() >= self.config.max_resident_pages {
            self.async_skipped_no_capacity += 1;
            return Ok(false);
        }
        let page_key = ResidentPageKey {
            tensor_idx,
            page_offset: span.offset,
        };
        if self.pending_pages.contains_key(&page_key) {
            return Ok(true);
        }
        let async_page_in = self.async_page_in.as_mut().expect("checked async_page_in");
        let Some((slot_idx, slot)) = async_page_in
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.pending.is_none())
        else {
            self.async_skipped_no_slot += 1;
            return Ok(false);
        };
        if span.copy_len > slot.buffer.len() {
            return Err(anyhow!(
                "MoE async staging page too small: copy_len={} staging_len={}",
                span.copy_len,
                slot.buffer.len()
            ));
        }
        let src = store
            .raw_byte_range(name, span.offset, span.copy_len)
            .with_context(|| {
                format!(
                    "read baked MoE expert page for async prefetch tensor={name} offset={} len={}",
                    span.offset, span.copy_len
                )
            })?;
        slot.buffer.as_mut_slice()[..span.copy_len].copy_from_slice(src);
        let ordinal = self.arena.device_ordinal();
        let allocation = self
            .arena
            .allocation_mut(allocation_id)
            .ok_or_else(|| anyhow!("virtual allocation id {allocation_id} missing"))?;
        let buffer = allocation.buffer_mut();
        buffer
            .map_range_bytes_no_sync(span.offset, span.len)
            .with_context(|| {
                format!(
                    "async map MoE expert page tensor={name} offset={}",
                    span.offset
                )
            })?;
        memset_zeros_async(
            ordinal,
            &async_page_in.stream,
            buffer.offset_mut_ptr(span.offset),
            span.len,
        )
        .context("async memset MoE expert page")?;
        copy_h2d_async(
            ordinal,
            &async_page_in.stream,
            buffer.offset_mut_ptr(span.offset),
            slot.buffer.as_ptr(),
            span.copy_len,
        )
        .context("async H2D MoE expert page")?;
        slot.event
            .record_on_stream(&async_page_in.stream)
            .context("record MoE async prefetch event")?;
        slot.pending = Some(page_key);
        self.pending_pages.insert(
            page_key,
            PendingPage {
                tensor_idx,
                page_offset: span.offset,
                page_len: span.len,
                copy_len: span.copy_len,
                last_used: self.clock,
                slot_idx,
            },
        );
        self.async_scheduled_pages += 1;
        self.async_uploaded_bytes += span.copy_len;
        self.prefetch_uploaded_bytes += span.copy_len;
        self.async_pending_pages_peak = self.async_pending_pages_peak.max(self.pending_pages.len());
        Ok(true)
    }

    fn promote_completed_pending_pages(&mut self) -> Result<()> {
        let Some(async_page_in) = self.async_page_in.as_mut() else {
            return Ok(());
        };
        let mut completed = Vec::new();
        for slot in &async_page_in.slots {
            if let Some(key) = slot.pending {
                if slot
                    .event
                    .query()
                    .context("query MoE async prefetch event")?
                {
                    completed.push(key);
                }
            }
        }
        for key in completed {
            self.finish_pending_page(key, None);
        }
        Ok(())
    }

    fn wait_pending_page(&mut self, key: ResidentPageKey) -> Result<()> {
        let Some(pending) = self.pending_pages.get(&key) else {
            return Ok(());
        };
        let slot_idx = pending.slot_idx;
        let Some(async_page_in) = self.async_page_in.as_mut() else {
            return Ok(());
        };
        async_page_in.slots[slot_idx]
            .event
            .synchronize()
            .context("wait MoE async prefetch page")?;
        self.async_waited_pages += 1;
        self.finish_pending_page(key, Some(self.clock));
        Ok(())
    }

    fn finish_pending_page(&mut self, key: ResidentPageKey, last_used: Option<u64>) {
        let Some(pending) = self.pending_pages.remove(&key) else {
            return;
        };
        if let Some(async_page_in) = self.async_page_in.as_mut() {
            if let Some(slot) = async_page_in.slots.get_mut(pending.slot_idx) {
                slot.pending = None;
            }
        }
        self.resident_pages.insert(
            key,
            ResidentPage {
                tensor_idx: pending.tensor_idx,
                page_offset: pending.page_offset,
                page_len: pending.page_len,
                last_used: last_used.unwrap_or(pending.last_used),
            },
        );
        self.uploaded_bytes += pending.copy_len;
        self.async_completed_pages += 1;
    }

    fn evict_lru_page(&mut self) -> Result<()> {
        let Some((victim, page)) =
            select_lru_resident_page(&self.resident_pages, &self.protected_pages)
        else {
            if let Some(key) = self.pending_pages.keys().next().copied() {
                self.wait_pending_page(key)?;
            }
            return Ok(());
        };
        let victim_was_protected = self.protected_pages.contains_key(&victim);

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
            if self.protected_pages.remove(&victim).is_some() {
                self.protected_evicted_pages += 1;
            }
            let removed_slices = remove_slices_overlapping(
                &mut self.resident,
                page.tensor_idx,
                page.page_offset,
                page.page_offset + page.page_len,
            );
            self.evicted_pages += 1;
            self.evicted_slices += removed_slices as u64;
            return Ok(());
        }

        let removed_pages =
            remove_pages_overlapping(&mut self.resident_pages, page.tensor_idx, &removed_ranges);
        let removed_slices =
            remove_slices_overlapping_ranges(&mut self.resident, page.tensor_idx, &removed_ranges);
        let demoted = prune_protected_pages(&mut self.protected_pages, &self.resident_pages);
        self.protect_demotions += demoted as u64;
        if victim_was_protected {
            self.protected_evicted_pages += 1;
        }
        self.evicted_pages += removed_pages as u64;
        self.evicted_slices += removed_slices as u64;
        Ok(())
    }

    fn clamp_protected_pages(&mut self) {
        while self.protected_pages.len() > self.max_protected_pages {
            let Some(victim) = oldest_protected_page(&self.protected_pages) else {
                return;
            };
            self.protected_pages.remove(&victim);
            self.protect_demotions += 1;
        }
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

    fn max_pages_per_expert_slice(&self) -> usize {
        (0..self.expert_count)
            .map(|expert_idx| {
                page_spans(
                    self.page_bytes,
                    expert_idx * self.expert_bytes,
                    self.expert_bytes,
                    self.len_bytes,
                )
                .len()
            })
            .max()
            .unwrap_or(0)
    }
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
    fn protected_pages_are_evicted_after_unprotected_pages() {
        with_supported_vmm_backend(
            "protected_pages_are_evicted_after_unprotected_pages",
            |_backend| {
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
                    MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(2).unwrap());
                manager.set_max_protected_pages(1);
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
                let e2 = MoeExpertKey {
                    layer_idx: 0,
                    expert_idx: 2,
                    projection: MoeExpertProjection::GateUp,
                };

                manager.ensure_resident(&store, e0).expect("load expert 0");
                manager.protect_resident(e0).expect("protect expert 0");
                manager.ensure_resident(&store, e1).expect("load expert 1");
                manager.ensure_resident(&store, e2).expect("load expert 2");

                assert!(manager.is_resident(e0));
                assert!(!manager.is_resident(e1));
                assert!(manager.is_resident(e2));
                assert_eq!(manager.stats().protected_pages, 1);
                assert_eq!(manager.stats().protect_hits, 1);
                assert!(manager.stats().evicted_pages >= 1);
            },
        );
    }

    #[test]
    fn prefetch_does_not_evict_when_page_budget_is_full() {
        with_supported_vmm_backend(
            "prefetch_does_not_evict_when_page_budget_is_full",
            |_backend| {
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

                let tmp = synthetic_store(2, expert_bytes);
                let store = BakedStore::open(tmp.path()).expect("open synthetic store");
                let mut manager =
                    MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
                manager
                    .register_tensor(
                        &store,
                        0,
                        MoeExpertProjection::GateUp,
                        "model.layers.0.mlp.experts.gate_up_proj",
                        2,
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
                manager
                    .prefetch_resident(&store, e1)
                    .expect("prefetch expert 1");

                assert!(manager.is_resident(e0));
                assert!(!manager.is_resident(e1));
                assert_eq!(manager.stats().resident_pages, 1);
                assert_eq!(manager.stats().evicted_pages, 0);
                assert_eq!(manager.stats().page_misses, 1);
                assert_eq!(manager.stats().prefetch_requests, 1);
                assert_eq!(manager.stats().prefetch_misses, 1);
                assert_eq!(manager.stats().prefetch_page_misses, 0);
                assert_eq!(manager.stats().prefetch_skipped, 1);
                assert_eq!(manager.stats().prefetch_skipped_pages, 1);
            },
        );
    }

    #[test]
    fn async_prefetch_promotes_before_demand() {
        with_supported_vmm_backend("async_prefetch_promotes_before_demand", |backend| {
            if backend != Backend::Hip {
                return;
            }
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

            let tmp = synthetic_store(2, expert_bytes);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    2,
                )
                .expect("register tensor");
            manager
                .enable_async_prefetch(1)
                .expect("enable async prefetch");

            let e1 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 1,
                projection: MoeExpertProjection::GateUp,
            };
            manager
                .prefetch_resident(&store, e1)
                .expect("async prefetch expert 1");
            assert_eq!(manager.stats().async_scheduled_pages, 1);
            manager
                .ensure_resident(&store, e1)
                .expect("demand async expert 1");
            assert!(manager.is_resident(e1));
            assert_eq!(manager.stats().async_completed_pages, 1);
            let promoted = manager
                .resident_pages
                .get(&ResidentPageKey {
                    tensor_idx: 0,
                    page_offset: expert_bytes,
                })
                .expect("promoted pending page");
            assert_eq!(promoted.last_used, manager.clock);

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
    fn page_budget_uses_registered_projection_footprint() {
        with_supported_vmm_backend(
            "page_budget_uses_registered_projection_footprint",
            |_backend| {
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
                let page_bytes = probe.tensors[0].page_bytes;

                let expert_bytes = page_bytes + 1;
                let tmp = synthetic_store(2, expert_bytes);
                let store = BakedStore::open(tmp.path()).expect("open synthetic store");
                let mut manager =
                    MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
                manager
                    .register_tensor(
                        &store,
                        0,
                        MoeExpertProjection::GateUp,
                        "model.layers.0.mlp.experts.gate_up_proj",
                        2,
                    )
                    .expect("register gate/up tensor");
                manager
                    .register_tensor(
                        &store,
                        0,
                        MoeExpertProjection::Down,
                        "model.layers.0.mlp.experts.gate_up_proj",
                        2,
                    )
                    .expect("register down tensor");

                assert_eq!(
                    manager
                        .page_budget_for_routed_experts(3)
                        .expect("derive page budget"),
                    12
                );
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
