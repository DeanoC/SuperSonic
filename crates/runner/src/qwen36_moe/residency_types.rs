//! Shared types for Qwen3.6-MoE sparse residency.

use std::ffi::c_void;

use gpu_hal::ScalarType;

use crate::qwen36_moe_residency_pages::page_spans;

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
    /// Allow lookahead prefetch to evict least-recently-used resident pages
    /// when the page budget is full. Disabled by default so prefetch remains
    /// non-disruptive unless an experiment opts in.
    pub prefetch_evict: bool,
}

impl MoeExpertResidencyConfig {
    pub fn new(max_resident_pages: usize) -> anyhow::Result<Self> {
        if max_resident_pages == 0 {
            anyhow::bail!("max_resident_pages must be > 0");
        }
        Ok(Self {
            max_resident_pages,
            prefetch_evict: false,
        })
    }

    pub fn with_prefetch_evict(mut self, prefetch_evict: bool) -> Self {
        self.prefetch_evict = prefetch_evict;
        self
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
    pub prefetch_evicted_pages: u64,
    pub prefetch_uploaded_bytes: usize,
    pub protected_pages: usize,
    pub protected_page_budget: usize,
    pub protect_requests: u64,
    pub protect_hits: u64,
    pub protect_misses: u64,
    pub protect_demotions: u64,
    pub protected_evicted_pages: u64,
    pub fixed_hot_pages: usize,
    pub fixed_hot_page_budget: usize,
    pub fixed_hot_requests: u64,
    pub fixed_hot_hits: u64,
    pub fixed_hot_misses: u64,
    pub fixed_hot_skipped: u64,
    pub fixed_hot_evicted_pages: u64,
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
pub(crate) struct ExpertTensor {
    pub(crate) name: String,
    pub(crate) projection: MoeExpertProjection,
    pub(crate) allocation_id: usize,
    pub(crate) ptr: *const c_void,
    pub(crate) dtype: ScalarType,
    pub(crate) shape: Vec<usize>,
    pub(crate) len_bytes: usize,
    pub(crate) expert_count: usize,
    pub(crate) expert_bytes: usize,
    pub(crate) page_bytes: usize,
}

impl ExpertTensor {
    pub(crate) fn reservation(&self) -> MoeExpertTensorReservation {
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

    pub(crate) fn max_pages_per_expert_slice(&self) -> usize {
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
