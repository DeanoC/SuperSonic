use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

use crate::backend::{current_backend, Backend};
use crate::error::{backend_error, GpuError, Result};
use crate::hip_sys::*;
use crate::ops::{self, hal_profile_time};
use crate::scalar_type::ScalarType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualBacking {
    CpuBackup,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualBufferStats {
    /// Logical tensor byte length requested by the caller.
    pub logical_bytes: usize,
    /// Reserved virtual address bytes. This is page-aligned and can exceed the
    /// logical tensor size, especially on HIP where we use a 2 MiB page floor.
    pub reserved_bytes: usize,
    /// Physical bytes currently mapped into the reservation.
    pub resident_bytes: usize,
    /// Logical bytes covered by resident mappings.
    pub logical_resident_bytes: usize,
    /// Logical bytes held in the CPU backup image.
    pub logical_backup_bytes: usize,
    pub mapping_count: usize,
    pub page_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualAllocationRole {
    KvCache,
    Weights,
    MoeExpert,
    Scratch,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualAllocationStats {
    pub name: String,
    pub role: VirtualAllocationRole,
    pub buffer: VirtualBufferStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtualArenaStats {
    pub allocations: usize,
    pub logical_bytes: usize,
    pub reserved_bytes: usize,
    pub resident_bytes: usize,
    pub logical_resident_bytes: usize,
    pub logical_backup_bytes: usize,
    pub mapping_count: usize,
}

enum PhysicalHandle {
    Hip(Option<HipMemGenericAllocationHandle>),
}

struct Mapping {
    offset: usize,
    len: usize,
    handle: PhysicalHandle,
}

struct BackupChunk {
    offset: usize,
    data: Vec<u8>,
}

/// Reserved device virtual address range with lazily mapped physical pages.
///
/// The base pointer is stable for the lifetime of the reservation. Callers map
/// ranges as their logical capacity grows; descriptors can keep using the same
/// base pointer while newly touched pages become resident.
///
/// `len_bytes()` is the logical tensor length. `reserved_bytes()`,
/// `mapped_bytes()`, and `resident_bytes()` are physical/page-aligned VMM
/// counters. Use `logical_resident_bytes()` and `logical_backup_bytes()` when
/// reasoning about model data rather than page residency.
pub struct VirtualBuffer {
    ptr: NonNull<c_void>,
    reserved_bytes: usize,
    mapped_bytes: usize,
    dtype: ScalarType,
    shape: Vec<usize>,
    device_ordinal: usize,
    backend: Backend,
    granularity: usize,
    mappings: Vec<Mapping>,
    backing: VirtualBacking,
    backup: Option<Vec<BackupChunk>>,
}

pub struct VirtualAllocation {
    name: String,
    role: VirtualAllocationRole,
    buffer: VirtualBuffer,
}

pub struct VirtualArena {
    device_ordinal: usize,
    backing: VirtualBacking,
    allocations: Vec<VirtualAllocation>,
}

unsafe impl Send for VirtualBuffer {}
unsafe impl Sync for VirtualBuffer {}

impl std::fmt::Debug for VirtualBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VirtualBuffer({:?}, {:?}, mapped={} reserved={} device:{})",
            self.dtype, self.shape, self.mapped_bytes, self.reserved_bytes, self.device_ordinal
        )
    }
}

impl std::fmt::Debug for VirtualAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualAllocation")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl std::fmt::Debug for VirtualArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualArena")
            .field("device_ordinal", &self.device_ordinal)
            .field("backing", &self.backing)
            .field("allocations", &self.allocations.len())
            .finish()
    }
}

impl VirtualAllocation {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn role(&self) -> VirtualAllocationRole {
        self.role
    }

    pub fn buffer(&self) -> &VirtualBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut VirtualBuffer {
        &mut self.buffer
    }

    pub fn stats(&self) -> VirtualAllocationStats {
        VirtualAllocationStats {
            name: self.name.clone(),
            role: self.role,
            buffer: self.buffer.stats(),
        }
    }
}

impl VirtualArena {
    pub fn new(device_ordinal: usize, backing: VirtualBacking) -> Self {
        Self {
            device_ordinal,
            backing,
            allocations: Vec::new(),
        }
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn backing(&self) -> VirtualBacking {
        self.backing
    }

    pub fn reserve(
        &mut self,
        name: impl Into<String>,
        role: VirtualAllocationRole,
        dtype: ScalarType,
        shape: &[usize],
    ) -> Result<usize> {
        let buffer = VirtualBuffer::reserve(self.device_ordinal, dtype, shape, self.backing)?;
        let id = self.allocations.len();
        self.allocations.push(VirtualAllocation {
            name: name.into(),
            role,
            buffer,
        });
        Ok(id)
    }

    pub fn allocation(&self, id: usize) -> Option<&VirtualAllocation> {
        self.allocations.get(id)
    }

    pub fn allocation_mut(&mut self, id: usize) -> Option<&mut VirtualAllocation> {
        self.allocations.get_mut(id)
    }

    pub fn allocations(&self) -> &[VirtualAllocation] {
        &self.allocations
    }

    pub fn allocation_stats(&self) -> Vec<VirtualAllocationStats> {
        self.allocations
            .iter()
            .map(VirtualAllocation::stats)
            .collect()
    }

    pub fn stats(&self) -> VirtualArenaStats {
        self.allocations
            .iter()
            .map(|allocation| allocation.buffer.stats())
            .fold(
                VirtualArenaStats {
                    allocations: self.allocations.len(),
                    ..VirtualArenaStats::default()
                },
                |mut acc, stats| {
                    acc.logical_bytes += stats.logical_bytes;
                    acc.reserved_bytes += stats.reserved_bytes;
                    acc.resident_bytes += stats.resident_bytes;
                    acc.logical_resident_bytes += stats.logical_resident_bytes;
                    acc.logical_backup_bytes += stats.logical_backup_bytes;
                    acc.mapping_count += stats.mapping_count;
                    acc
                },
            )
    }
}

impl Drop for VirtualBuffer {
    fn drop(&mut self) {
        let _ = self.unmap_all_discard();
        match self.backend {
            Backend::Hip => {
                let _ = ops::with_device_impl(self.backend, self.device_ordinal, || {
                    let status =
                        unsafe { hipMemAddressFree(self.ptr.as_ptr(), self.reserved_bytes) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipMemAddressFree", status));
                    }
                    Ok(())
                });
            }
        }
    }
}

impl VirtualBuffer {
    pub fn reserve(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        backing: VirtualBacking,
    ) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let len_bytes = ops::byte_len(dtype, elems);
        if len_bytes == 0 {
            return Err(GpuError::InvalidArg(
                "virtual buffer reservation size must be > 0".into(),
            ));
        }
        let backend = current_backend();
        if !vmm_is_supported(backend, ordinal) {
            return Err(GpuError::Unsupported(format!(
                "{backend} VMM is not available on device {ordinal}"
            )));
        }
        let api_granularity = vmm_granularity(backend, ordinal)?;
        let granularity = effective_vmm_mapping_granularity(backend, api_granularity);
        let reserved_bytes = align_up(len_bytes, granularity);
        let ptr = hal_profile_time("vmm_reserve", reserved_bytes, || {
            reserve_address_range(backend, ordinal, reserved_bytes, granularity)
        })?;
        Ok(Self {
            ptr,
            reserved_bytes,
            mapped_bytes: 0,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
            backend,
            granularity,
            mappings: Vec::new(),
            backing,
            backup: None,
        })
    }

    pub fn reserve_and_map_prefix(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        prefix_bytes: usize,
        backing: VirtualBacking,
    ) -> Result<Self> {
        let mut buf = Self::reserve(ordinal, dtype, shape, backing)?;
        buf.map_prefix_bytes(prefix_bytes)?;
        Ok(buf)
    }

    pub fn map_prefix_bytes(&mut self, needed_bytes: usize) -> Result<()> {
        if needed_bytes > self.reserved_bytes {
            return Err(GpuError::InvalidArg(format!(
                "virtual map prefix {needed_bytes} exceeds reservation {}",
                self.reserved_bytes
            )));
        }
        self.map_range_bytes(0, needed_bytes)?;
        self.mapped_bytes = self
            .mapped_bytes
            .max(align_up(needed_bytes, self.granularity));
        Ok(())
    }

    pub fn map_range_bytes(&mut self, offset: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let end = offset.checked_add(len).ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "virtual map range overflows: offset={offset} len={len}"
            ))
        })?;
        if end > self.reserved_bytes {
            return Err(GpuError::InvalidArg(format!(
                "virtual map range [{offset}, {end}) exceeds reservation {}",
                self.reserved_bytes
            )));
        }
        let aligned_offset = align_down(offset, self.granularity);
        let aligned_end = align_up(end, self.granularity);
        if !self
            .missing_segments(aligned_offset, aligned_end)
            .is_empty()
        {
            ops::sync(self.device_ordinal)?;
        }
        for (seg_offset, seg_len) in self.missing_segments(aligned_offset, aligned_end) {
            let mapping = hal_profile_time("vmm_map", seg_len, || {
                map_physical(
                    self.backend,
                    self.device_ordinal,
                    self.ptr,
                    seg_offset,
                    seg_len,
                    seg_offset,
                    seg_len,
                )
            })?;
            self.mappings.push(mapping);
            self.mapped_bytes = self.mapped_bytes.max(seg_offset + seg_len);
            ops::memset_zeros(
                self.device_ordinal,
                unsafe { (self.ptr.as_ptr() as *mut u8).add(seg_offset) as *mut c_void },
                seg_len,
            )?;
            ops::sync(self.device_ordinal)?;
        }
        Ok(())
    }

    /// Map a byte range without synchronizing the device or clearing the new
    /// physical pages. This is only for callers that can prove no in-flight
    /// kernel can touch the newly mapped range and will initialize it before
    /// publishing it as resident.
    pub fn map_range_bytes_no_sync(&mut self, offset: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let end = offset.checked_add(len).ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "virtual map range overflows: offset={offset} len={len}"
            ))
        })?;
        if end > self.reserved_bytes {
            return Err(GpuError::InvalidArg(format!(
                "virtual map range [{offset}, {end}) exceeds reservation {}",
                self.reserved_bytes
            )));
        }
        let aligned_offset = align_down(offset, self.granularity);
        let aligned_end = align_up(end, self.granularity);
        for (seg_offset, seg_len) in self.missing_segments(aligned_offset, aligned_end) {
            let mapping = hal_profile_time("vmm_map_no_sync", seg_len, || {
                map_physical(
                    self.backend,
                    self.device_ordinal,
                    self.ptr,
                    seg_offset,
                    seg_len,
                    seg_offset,
                    seg_len,
                )
            })?;
            self.mappings.push(mapping);
            self.mapped_bytes = self.mapped_bytes.max(seg_offset + seg_len);
        }
        Ok(())
    }

    /// Copy a source file range into this virtual buffer without staging
    /// through a pageable host slice.
    ///
    /// Unsupported backends return before mapping any virtual pages, so callers
    /// cannot accidentally get a pageable H2D fallback when they requested a
    /// storage-direct transfer.
    pub fn copy_storage_range_bytes_no_sync(
        &mut self,
        dst_offset: usize,
        source_path: &Path,
        source_offset: u64,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        ops::ensure_storage_to_device_supported(self.backend, source_path, source_offset, len)?;
        self.map_range_bytes_no_sync(dst_offset, len)?;
        ops::copy_storage_to_device(
            self.device_ordinal,
            self.offset_mut_ptr(dst_offset),
            source_path,
            source_offset,
            len,
        )?;
        ops::sync(self.device_ordinal)?;
        Ok(())
    }

    pub fn unmap_all(&mut self) -> Result<()> {
        if self.mapped_bytes == 0 && self.mappings.is_empty() {
            return Ok(());
        }
        if self.backing == VirtualBacking::CpuBackup {
            self.backup_mapped_to_host()?;
        }
        self.unmap_all_discard()
    }

    pub fn evict_to_host(&mut self) -> Result<()> {
        if self.mapped_bytes == 0 && self.mappings.is_empty() {
            return Ok(());
        }
        if self.backing != VirtualBacking::CpuBackup {
            return Err(GpuError::InvalidArg(
                "virtual buffer eviction requires CpuBackup backing".into(),
            ));
        }
        ops::sync(self.device_ordinal)?;
        self.backup_mapped_to_host()?;
        ops::sync(self.device_ordinal)?;
        self.unmap_all_discard()
    }

    pub fn evict_discard(&mut self) -> Result<()> {
        ops::sync(self.device_ordinal)?;
        self.unmap_all_discard()
    }

    /// Unmap any resident physical mappings that overlap the requested byte
    /// range, without first copying data to the CPU backup image.
    ///
    /// The actual unmapped ranges are VMM-page aligned and may be wider than
    /// the logical range. Callers that maintain their own backing store should
    /// treat every returned range as non-resident.
    pub fn unmap_range_discard(
        &mut self,
        offset: usize,
        len: usize,
    ) -> Result<Vec<(usize, usize)>> {
        if len == 0 || self.mappings.is_empty() {
            return Ok(Vec::new());
        }
        let end = offset.checked_add(len).ok_or_else(|| {
            GpuError::InvalidArg(format!(
                "virtual unmap range overflows: offset={offset} len={len}"
            ))
        })?;
        if end > self.reserved_bytes {
            return Err(GpuError::InvalidArg(format!(
                "virtual unmap range [{offset}, {end}) exceeds reservation {}",
                self.reserved_bytes
            )));
        }

        let aligned_offset = align_down(offset, self.granularity);
        let aligned_end = align_up(end, self.granularity);
        ops::sync(self.device_ordinal)?;

        let mut kept = Vec::with_capacity(self.mappings.len());
        let mut removed = Vec::new();
        for mapping in self.mappings.drain(..) {
            let map_end = mapping.offset + mapping.len;
            if ranges_overlap(mapping.offset, map_end, aligned_offset, aligned_end) {
                let removed_range = (mapping.offset, mapping.len);
                hal_profile_time("vmm_unmap", mapping.len, || {
                    unmap_and_release(self.backend, self.device_ordinal, self.ptr, mapping)
                })?;
                removed.push(removed_range);
            } else {
                kept.push(mapping);
            }
        }
        self.mappings = kept;
        self.mapped_bytes = self
            .mappings
            .iter()
            .map(|mapping| mapping.offset + mapping.len)
            .max()
            .unwrap_or(0);
        Ok(removed)
    }

    fn unmap_all_discard(&mut self) -> Result<()> {
        if self.mapped_bytes == 0 && self.mappings.is_empty() {
            return Ok(());
        }
        for mapping in self.mappings.drain(..).rev() {
            hal_profile_time("vmm_unmap", mapping.len, || {
                unmap_and_release(self.backend, self.device_ordinal, self.ptr, mapping)
            })?;
        }
        self.mapped_bytes = 0;
        Ok(())
    }

    pub fn backup_mapped_to_host(&mut self) -> Result<()> {
        if self.backing != VirtualBacking::CpuBackup || self.mapped_bytes == 0 {
            return Ok(());
        }
        let logical_len = self.len_bytes();
        let mut chunks = Vec::with_capacity(self.mappings.len());
        let mut ranges: Vec<(usize, usize)> = self
            .mappings
            .iter()
            .filter_map(|mapping| {
                if mapping.offset >= logical_len {
                    return None;
                }
                let len = mapping.len.min(logical_len - mapping.offset);
                (len > 0).then_some((mapping.offset, len))
            })
            .collect();
        ranges.sort_by_key(|(offset, _)| *offset);
        for (offset, len) in ranges {
            let mut data = vec![0u8; len];
            let mut copied = 0usize;
            while copied < len {
                let chunk_len = (len - copied).min(VMM_BACKUP_COPY_CHUNK_BYTES);
                hal_profile_time("vmm_backup", chunk_len, || {
                    ops::copy_d2h(
                        self.device_ordinal,
                        unsafe { data.as_mut_ptr().add(copied) as *mut c_void },
                        unsafe {
                            (self.ptr.as_ptr() as *const u8).add(offset + copied) as *const c_void
                        },
                        chunk_len,
                    )
                })?;
                copied += chunk_len;
            }
            chunks.push(BackupChunk { offset, data });
        }
        ops::sync(self.device_ordinal)?;
        self.backup = Some(chunks);
        Ok(())
    }

    pub fn restore_backup(&mut self) -> Result<()> {
        self.map_backup_ranges_for_restore()?;
        self.copy_backup_to_mapped()
    }

    pub fn map_backup_ranges_for_restore(&mut self) -> Result<()> {
        let Some(chunks) = self.backup.as_ref() else {
            return Ok(());
        };
        let ranges = chunks
            .iter()
            .map(|chunk| (chunk.offset, chunk.data.len()))
            .collect::<Vec<_>>();
        for (offset, len) in ranges {
            self.map_range_bytes(offset, len)?;
        }
        Ok(())
    }

    pub fn copy_backup_to_mapped(&mut self) -> Result<()> {
        let Some(chunks) = self.backup.as_ref() else {
            return Ok(());
        };
        for chunk in chunks {
            let mut restored = false;
            for _attempt in 0..3 {
                let mut copied = 0usize;
                while copied < chunk.data.len() {
                    let chunk_len = (chunk.data.len() - copied).min(VMM_BACKUP_COPY_CHUNK_BYTES);
                    hal_profile_time("vmm_restore", chunk_len, || {
                        ops::copy_h2d(
                            self.device_ordinal,
                            unsafe {
                                (self.ptr.as_ptr() as *mut u8).add(chunk.offset + copied)
                                    as *mut c_void
                            },
                            unsafe { chunk.data.as_ptr().add(copied) as *const c_void },
                            chunk_len,
                        )
                    })?;
                    copied += chunk_len;
                }
                ops::sync(self.device_ordinal)?;
                let mut verify = vec![0u8; chunk.data.len()];
                ops::copy_d2h(
                    self.device_ordinal,
                    verify.as_mut_ptr() as *mut c_void,
                    unsafe { (self.ptr.as_ptr() as *const u8).add(chunk.offset) as *const c_void },
                    verify.len(),
                )?;
                ops::sync(self.device_ordinal)?;
                if verify == chunk.data {
                    restored = true;
                    break;
                }
            }
            if !restored {
                return Err(GpuError::backend(
                    self.backend,
                    format!(
                        "virtual buffer restore verification failed at offset {} len {}",
                        chunk.offset,
                        chunk.data.len()
                    ),
                ));
            }
        }
        ops::sync(self.device_ordinal)?;
        Ok(())
    }

    #[doc(hidden)]
    pub fn backup_chunks_for_debug(&self) -> Option<Vec<(usize, Vec<u8>)>> {
        self.backup.as_ref().map(|chunks| {
            chunks
                .iter()
                .map(|chunk| (chunk.offset, chunk.data.clone()))
                .collect()
        })
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }

    pub fn logical_resident_bytes(&self) -> usize {
        let logical_len = self.len_bytes();
        self.mappings
            .iter()
            .map(|mapping| {
                if mapping.offset >= logical_len {
                    0
                } else {
                    mapping.len.min(logical_len - mapping.offset)
                }
            })
            .sum()
    }

    pub fn resident_bytes(&self) -> usize {
        self.mappings.iter().map(|mapping| mapping.len).sum()
    }

    pub fn logical_backup_bytes(&self) -> usize {
        self.backup
            .as_ref()
            .map(|chunks| chunks.iter().map(|chunk| chunk.data.len()).sum())
            .unwrap_or(0)
    }

    pub fn stats(&self) -> VirtualBufferStats {
        VirtualBufferStats {
            logical_bytes: self.len_bytes(),
            reserved_bytes: self.reserved_bytes,
            resident_bytes: self.resident_bytes(),
            logical_resident_bytes: self.logical_resident_bytes(),
            logical_backup_bytes: self.logical_backup_bytes(),
            mapping_count: self.mapping_count(),
            page_bytes: self.granularity,
        }
    }

    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    pub fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    pub fn granularity(&self) -> usize {
        self.granularity
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn offset_ptr(&self, byte_offset: usize) -> *const c_void {
        debug_assert!(byte_offset <= self.reserved_bytes);
        unsafe { (self.ptr.as_ptr() as *const u8).add(byte_offset) as *const c_void }
    }

    pub fn offset_mut_ptr(&mut self, byte_offset: usize) -> *mut c_void {
        debug_assert!(byte_offset <= self.reserved_bytes);
        unsafe { (self.ptr.as_ptr() as *mut u8).add(byte_offset) as *mut c_void }
    }

    pub fn len_bytes(&self) -> usize {
        ops::byte_len(self.dtype, ops::elem_count(&self.shape))
    }

    pub fn dtype(&self) -> ScalarType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn to_host_bytes(&self) -> Result<Vec<u8>> {
        self.to_host_prefix_bytes(self.len_bytes())
    }

    pub fn to_host_prefix_bytes(&self, len: usize) -> Result<Vec<u8>> {
        if !self.is_range_mapped(0, len) {
            return Err(GpuError::InvalidArg(format!(
                "virtual D2H prefix {len} is not fully mapped"
            )));
        }
        ops::sync(self.device_ordinal)?;
        let mut buf = vec![0u8; len];
        ops::copy_d2h(
            self.device_ordinal,
            buf.as_mut_ptr() as *mut c_void,
            self.ptr.as_ptr(),
            len,
        )?;
        ops::sync(self.device_ordinal)?;
        Ok(buf)
    }

    pub fn to_host_range_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        if !self.is_range_mapped(offset, len) {
            return Err(GpuError::InvalidArg(format!(
                "virtual D2H range [{}, {}) is not fully mapped",
                offset,
                offset.saturating_add(len)
            )));
        }
        ops::sync(self.device_ordinal)?;
        let mut buf = vec![0u8; len];
        ops::copy_d2h(
            self.device_ordinal,
            buf.as_mut_ptr() as *mut c_void,
            unsafe { (self.ptr.as_ptr() as *const u8).add(offset) as *const c_void },
            len,
        )?;
        ops::sync(self.device_ordinal)?;
        Ok(buf)
    }

    fn missing_segments(&self, offset: usize, end: usize) -> Vec<(usize, usize)> {
        let mut covered: Vec<(usize, usize)> = self
            .mappings
            .iter()
            .filter_map(|mapping| {
                let map_start = mapping.offset.max(offset);
                let map_end = (mapping.offset + mapping.len).min(end);
                (map_start < map_end).then_some((map_start, map_end))
            })
            .collect();
        covered.sort_by_key(|(start, _)| *start);

        let mut segments = Vec::new();
        let mut cursor = offset;
        for (start, stop) in covered {
            if cursor < start {
                segments.push((cursor, start - cursor));
            }
            cursor = cursor.max(stop);
        }
        if cursor < end {
            segments.push((cursor, end - cursor));
        }
        segments
    }

    fn is_range_mapped(&self, offset: usize, len: usize) -> bool {
        let Some(end) = offset.checked_add(len) else {
            return false;
        };
        if end > self.reserved_bytes {
            return false;
        }
        self.missing_segments(offset, end).is_empty()
    }
}

pub fn vmm_is_supported(backend: Backend, ordinal: usize) -> bool {
    if std::env::var_os("SUPERSONIC_FORCE_VMM_UNSUPPORTED").is_some() {
        return false;
    }
    if backend == Backend::Hip {
        {
            let mut supported = 0;
            let status = unsafe {
                hipDeviceGetAttribute(
                    &mut supported,
                    HIP_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                    ordinal as i32,
                )
            };
            if status != 0 || supported == 0 {
                return false;
            }
        }
    }
    backend == Backend::Hip && vmm_granularity(backend, ordinal).is_ok()
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0);
    value.div_ceil(alignment) * alignment
}

fn align_down(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0);
    (value / alignment) * alignment
}

fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

const VMM_BACKUP_COPY_CHUNK_BYTES: usize = usize::MAX;
const HIP_VMM_MIN_STABLE_MAPPING_BYTES: usize = 2 * 1024 * 1024;

fn effective_vmm_mapping_granularity(backend: Backend, api_granularity: usize) -> usize {
    if backend != Backend::Hip {
        return api_granularity;
    }
    let floor = std::env::var("SUPERSONIC_HIP_VMM_MIN_MAP_BYTES")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(HIP_VMM_MIN_STABLE_MAPPING_BYTES);
    if floor == 0 {
        return api_granularity;
    }
    // HIP can report 4 KiB VMM granularity on ROCm systems where remap/restore
    // of many sub-2 MiB mappings corrupts earlier mappings. Match the
    // conservative small-page size used by mature VMM allocators on ROCm.
    api_granularity.max(align_up(floor, api_granularity))
}

fn allocation_prop_hip(ordinal: usize) -> HipMemAllocationProp {
    HipMemAllocationProp {
        type_: HIP_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_type: HIP_MEM_HANDLE_TYPE_NONE,
        location: HipMemLocation {
            type_: HIP_MEM_LOCATION_TYPE_DEVICE,
            id: ordinal as i32,
        },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: HipMemAllocationPropAllocFlags {
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
        },
    }
}

fn vmm_granularity(backend: Backend, ordinal: usize) -> Result<usize> {
    match backend {
        Backend::Hip => {
            let prop = allocation_prop_hip(ordinal);
            let mut granularity = 0usize;
            let status = unsafe {
                hipMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
                )
            };
            if status != 0 {
                return Err(backend_error(
                    Backend::Hip,
                    "hipMemGetAllocationGranularity",
                    status,
                ));
            }
            if granularity == 0 {
                return Err(GpuError::Unsupported(
                    "HIP VMM returned zero allocation granularity".into(),
                ));
            }
            Ok(granularity)
        }
    }
}

fn reserve_address_range(
    backend: Backend,
    ordinal: usize,
    len: usize,
    alignment: usize,
) -> Result<NonNull<c_void>> {
    ops::with_device_impl(backend, ordinal, || match backend {
        Backend::Hip => {
            let mut ptr = std::ptr::null_mut();
            let status =
                unsafe { hipMemAddressReserve(&mut ptr, len, alignment, std::ptr::null_mut(), 0) };
            if status != 0 {
                return Err(backend_error(Backend::Hip, "hipMemAddressReserve", status));
            }
            NonNull::new(ptr).ok_or_else(|| {
                GpuError::backend(Backend::Hip, "hipMemAddressReserve returned null".into())
            })
        }
    })
}

fn map_physical(
    backend: Backend,
    ordinal: usize,
    base: NonNull<c_void>,
    offset: usize,
    len: usize,
    access_offset: usize,
    access_len: usize,
) -> Result<Mapping> {
    ops::with_device_impl(backend, ordinal, || match backend {
        Backend::Hip => {
            let prop = allocation_prop_hip(ordinal);
            let mut handle = std::ptr::null_mut();
            let status = unsafe { hipMemCreate(&mut handle, len, &prop, 0) };
            if status != 0 {
                return Err(backend_error(Backend::Hip, "hipMemCreate", status));
            }
            let ptr = unsafe { (base.as_ptr() as *mut u8).add(offset) as *mut c_void };
            let status = unsafe { hipMemMap(ptr, len, 0, handle, 0) };
            if status != 0 {
                let _ = unsafe { hipMemRelease(handle) };
                return Err(backend_error(Backend::Hip, "hipMemMap", status));
            }
            let access = HipMemAccessDesc {
                location: HipMemLocation {
                    type_: HIP_MEM_LOCATION_TYPE_DEVICE,
                    id: ordinal as i32,
                },
                flags: HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE,
            };
            let access_ptr =
                unsafe { (base.as_ptr() as *mut u8).add(access_offset) as *mut c_void };
            let status = unsafe { hipMemSetAccess(access_ptr, access_len, &access, 1) };
            if status != 0 {
                let _ = unsafe { hipMemUnmap(ptr, len) };
                let _ = unsafe { hipMemRelease(handle) };
                return Err(backend_error(Backend::Hip, "hipMemSetAccess", status));
            }
            let status = unsafe { hipMemRelease(handle) };
            if status != 0 {
                let _ = unsafe { hipMemUnmap(ptr, len) };
                return Err(backend_error(Backend::Hip, "hipMemRelease", status));
            }
            Ok(Mapping {
                offset,
                len,
                handle: PhysicalHandle::Hip(None),
            })
        }
    })
}

fn unmap_and_release(
    backend: Backend,
    ordinal: usize,
    base: NonNull<c_void>,
    mapping: Mapping,
) -> Result<()> {
    ops::with_device_impl(backend, ordinal, || match (backend, mapping.handle) {
        (Backend::Hip, PhysicalHandle::Hip(handle)) => {
            let ptr = unsafe { (base.as_ptr() as *mut u8).add(mapping.offset) as *mut c_void };
            let status = unsafe { hipMemUnmap(ptr, mapping.len) };
            if status != 0 {
                return Err(backend_error(Backend::Hip, "hipMemUnmap", status));
            }
            if let Some(handle) = handle {
                let status = unsafe { hipMemRelease(handle) };
                if status != 0 {
                    return Err(backend_error(Backend::Hip, "hipMemRelease", status));
                }
            }
            ops::sync(ordinal)?;
            Ok(())
        }
    })
}
