//! Shared types for Qwen3.6-MoE sparse residency.

use std::ffi::c_void;

use gpu_hal::ScalarType;
pub use supersonic_runtime::qwen36_moe::residency::{
    MoeExpertKey, MoeExpertProjection, MoeExpertResidencyConfig, MoeExpertResidencyStats,
    MoeExpertTensorReservation,
};

use crate::qwen36_moe_residency_pages::page_spans;

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
