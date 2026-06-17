//! Page-level bookkeeping helpers for Qwen3.6-MoE sparse residency.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use gpu_hal::{GpuEvent, GpuStream, PinnedHostBuffer};

#[derive(Debug, Clone)]
pub(crate) struct ResidentSlice {
    pub(crate) tensor_idx: usize,
    pub(crate) page_offset: usize,
    pub(crate) page_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ResidentPageKey {
    pub(crate) tensor_idx: usize,
    pub(crate) page_offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ResidentPage {
    pub(crate) tensor_idx: usize,
    pub(crate) page_offset: usize,
    pub(crate) page_len: usize,
    pub(crate) last_used: u64,
}

pub(crate) struct PendingPage {
    pub(crate) tensor_idx: usize,
    pub(crate) page_offset: usize,
    pub(crate) page_len: usize,
    pub(crate) copy_len: usize,
    pub(crate) last_used: u64,
    pub(crate) slot_idx: usize,
}

pub(crate) struct AsyncStagingSlot {
    pub(crate) buffer: PinnedHostBuffer,
    pub(crate) event: GpuEvent,
    pub(crate) pending: Option<ResidentPageKey>,
}

pub(crate) struct AsyncPageIn {
    pub(crate) stream: GpuStream,
    pub(crate) slots: Vec<AsyncStagingSlot>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PageSpan {
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) copy_len: usize,
}

pub(crate) fn page_spans(
    page_bytes: usize,
    offset: usize,
    len: usize,
    total_len: usize,
) -> Vec<PageSpan> {
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

pub(crate) fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

pub(crate) fn select_lru_resident_page(
    resident_pages: &HashMap<ResidentPageKey, ResidentPage>,
    protected_pages: &HashMap<ResidentPageKey, u64>,
    fixed_hot_pages: &HashSet<ResidentPageKey>,
    unevictable_pages: &HashSet<ResidentPageKey>,
) -> Option<(ResidentPageKey, ResidentPage)> {
    resident_pages
        .iter()
        .filter(|(key, _)| !unevictable_pages.contains(key))
        .min_by_key(|(key, page)| {
            (
                fixed_hot_pages.contains(key),
                protected_pages.contains_key(key),
                page.last_used,
                page.tensor_idx,
                page.page_offset,
            )
        })
        .map(|(key, page)| (*key, page.clone()))
}

pub(crate) fn oldest_protected_page(
    protected_pages: &HashMap<ResidentPageKey, u64>,
) -> Option<ResidentPageKey> {
    protected_pages
        .iter()
        .min_by_key(|(_, protected_at)| **protected_at)
        .map(|(key, _)| *key)
}

pub(crate) fn prune_protected_pages(
    protected_pages: &mut HashMap<ResidentPageKey, u64>,
    resident_pages: &HashMap<ResidentPageKey, ResidentPage>,
) -> usize {
    let before = protected_pages.len();
    protected_pages.retain(|key, _| resident_pages.contains_key(key));
    before - protected_pages.len()
}

pub(crate) fn remove_pages_overlapping(
    resident_pages: &mut HashMap<ResidentPageKey, ResidentPage>,
    tensor_idx: usize,
    ranges: &[(usize, usize)],
) -> usize {
    let before = resident_pages.len();
    resident_pages.retain(|_, page| {
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
    before - resident_pages.len()
}

pub(crate) fn remove_slices_overlapping_ranges<K: Eq + Hash>(
    resident_slices: &mut HashMap<K, ResidentSlice>,
    tensor_idx: usize,
    ranges: &[(usize, usize)],
) -> usize {
    let before = resident_slices.len();
    resident_slices.retain(|_, resident| {
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
    before - resident_slices.len()
}

pub(crate) fn remove_slices_overlapping<K: Eq + Hash>(
    resident_slices: &mut HashMap<K, ResidentSlice>,
    tensor_idx: usize,
    offset: usize,
    end: usize,
) -> usize {
    let before = resident_slices.len();
    resident_slices.retain(|_, resident| {
        resident.tensor_idx != tensor_idx
            || !ranges_overlap(
                resident.page_offset,
                resident.page_offset + resident.page_len,
                offset,
                end,
            )
    });
    before - resident_slices.len()
}
