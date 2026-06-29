//! Runtime contracts for Qwen3.6-MoE sparse expert residency.

use std::ffi::c_void;

use gpu_hal::ScalarType;

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
    pub max_resident_pages: usize,
    /// Allow lookahead prefetch to evict least-recently-used resident pages.
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

#[cfg(test)]
mod tests {
    use super::{
        MoeExpertKey, MoeExpertProjection, MoeExpertResidencyConfig, MoeExpertResidencyStats,
    };

    #[test]
    fn residency_config_rejects_zero_page_budget() {
        assert!(MoeExpertResidencyConfig::new(0).is_err());
        assert_eq!(
            MoeExpertResidencyConfig::new(8)
                .unwrap()
                .with_prefetch_evict(true),
            MoeExpertResidencyConfig {
                max_resident_pages: 8,
                prefetch_evict: true,
            }
        );
    }

    #[test]
    fn expert_keys_are_hashable_runtime_contracts() {
        let key = MoeExpertKey {
            layer_idx: 3,
            expert_idx: 17,
            projection: MoeExpertProjection::GateUp,
        };

        let mut set = std::collections::HashSet::new();
        set.insert(key);

        assert!(set.contains(&key));
        assert_ne!(MoeExpertProjection::GateUp, MoeExpertProjection::Down);
    }

    #[test]
    fn residency_stats_default_to_empty() {
        let stats = MoeExpertResidencyStats::default();

        assert_eq!(stats.registered_tensors, 0);
        assert_eq!(stats.resident_pages, 0);
        assert_eq!(stats.async_pending_pages_peak, 0);
    }
}
