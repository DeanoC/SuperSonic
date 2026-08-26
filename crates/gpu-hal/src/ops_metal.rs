use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::backend::{current_backend, current_strategy_for, Backend, BufferKind, DeviceInfo};
use crate::error::{backend_error, GpuError, Result};
use crate::metal_sys::*;
use crate::scalar_type::ScalarType;

static HAL_PROFILE_EXPLICIT_ENABLED: AtomicBool = AtomicBool::new(false);
static HAL_PROFILE_ACTIVE_CAPTURES: AtomicUsize = AtomicUsize::new(0);
static HAL_PROFILE: OnceLock<Mutex<HalProfileAccumulator>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct HalProfileEntry {
    pub op: String,
    pub calls: u64,
    pub total_ms: f64,
    pub max_ms: f64,
    pub total_bytes: u64,
}

impl HalProfileEntry {
    pub fn mean_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_ms / self.calls as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HalProfileSnapshot {
    pub total_calls: u64,
    pub total_ms: f64,
    pub alloc_calls: u64,
    pub alloc_bytes: u64,
    pub free_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub d2d_bytes: u64,
    pub memset_bytes: u64,
    pub sync_calls: u64,
    pub entries: Vec<HalProfileEntry>,
}

#[derive(Debug)]
#[must_use = "finish the capture to retain its HAL profile snapshot"]
pub struct HalProfileCapture {
    id: u64,
    active: bool,
}

#[derive(Debug, Default)]
struct HalProfileAccumulator {
    entries: BTreeMap<String, HalProfileEntry>,
    captures: BTreeMap<u64, BTreeMap<String, HalProfileEntry>>,
    next_capture_id: u64,
}

impl HalProfileCapture {
    pub fn begin() -> Self {
        let profile = HAL_PROFILE.get_or_init(|| Mutex::new(HalProfileAccumulator::default()));
        let mut profile = profile.lock().expect("HAL profile mutex poisoned");
        profile.next_capture_id = profile
            .next_capture_id
            .checked_add(1)
            .expect("HAL profile capture id exhausted");
        let id = profile.next_capture_id;
        profile.captures.insert(id, BTreeMap::new());
        HAL_PROFILE_ACTIVE_CAPTURES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                active.checked_add(1)
            })
            .expect("HAL profile capture count exhausted");
        Self { id, active: true }
    }

    pub fn finish(mut self) -> HalProfileSnapshot {
        self.close()
    }

    fn close(&mut self) -> HalProfileSnapshot {
        if !self.active {
            return HalProfileSnapshot::default();
        }
        let profile = HAL_PROFILE.get_or_init(|| Mutex::new(HalProfileAccumulator::default()));
        let mut profile = profile.lock().expect("HAL profile mutex poisoned");
        let entries = profile
            .captures
            .remove(&self.id)
            .unwrap_or_default()
            .into_values()
            .collect();
        self.active = false;
        HAL_PROFILE_ACTIVE_CAPTURES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                active.checked_sub(1)
            })
            .expect("HAL profile capture count underflow");
        drop(profile);
        hal_profile_snapshot_from_entries(entries)
    }
}

impl Drop for HalProfileCapture {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub fn hal_profile_set_enabled(enabled: bool) {
    HAL_PROFILE_EXPLICIT_ENABLED.store(enabled, Ordering::Release);
}

pub fn hal_profile_enabled() -> bool {
    HAL_PROFILE_EXPLICIT_ENABLED.load(Ordering::Acquire)
        || HAL_PROFILE_ACTIVE_CAPTURES.load(Ordering::Acquire) > 0
        || std::env::var_os("SUPERSONIC_HAL_PROFILE").is_some()
}

pub fn hal_profile_reset() {
    if let Some(profile) = HAL_PROFILE.get() {
        profile
            .lock()
            .expect("HAL profile mutex poisoned")
            .entries
            .clear();
    }
}

pub fn hal_profile_snapshot() -> HalProfileSnapshot {
    let Some(profile) = HAL_PROFILE.get() else {
        return HalProfileSnapshot::default();
    };
    let entries = profile
        .lock()
        .expect("HAL profile mutex poisoned")
        .entries
        .values()
        .cloned()
        .collect();
    hal_profile_snapshot_from_entries(entries)
}

fn hal_profile_snapshot_from_entries(mut entries: Vec<HalProfileEntry>) -> HalProfileSnapshot {
    let mut snapshot = HalProfileSnapshot::default();
    entries.sort_by(|lhs, rhs| {
        rhs.total_ms
            .partial_cmp(&lhs.total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.op.cmp(&rhs.op))
    });
    for entry in &entries {
        snapshot.total_calls += entry.calls;
        snapshot.total_ms += entry.total_ms;
        match entry.op.as_str() {
            "alloc" => {
                snapshot.alloc_calls += entry.calls;
                snapshot.alloc_bytes += entry.total_bytes;
            }
            "free" => snapshot.free_calls += entry.calls,
            "copy_h2d" => snapshot.h2d_bytes += entry.total_bytes,
            "copy_d2h" => snapshot.d2h_bytes += entry.total_bytes,
            "copy_d2d" => snapshot.d2d_bytes += entry.total_bytes,
            "memset_zeros" => snapshot.memset_bytes += entry.total_bytes,
            "sync" => snapshot.sync_calls += entry.calls,
            _ => {}
        }
    }
    snapshot.entries = entries;
    snapshot
}

fn update_hal_profile_entry(
    entries: &mut BTreeMap<String, HalProfileEntry>,
    op: &'static str,
    bytes: usize,
    elapsed_ms: f64,
) {
    let entry = entries
        .entry(op.to_string())
        .or_insert_with(|| HalProfileEntry {
            op: op.to_string(),
            calls: 0,
            total_ms: 0.0,
            max_ms: 0.0,
            total_bytes: 0,
        });
    entry.calls += 1;
    entry.total_ms += elapsed_ms;
    entry.max_ms = entry.max_ms.max(elapsed_ms);
    entry.total_bytes += bytes as u64;
}

fn record_enabled_hal_profile_sample(op: &'static str, bytes: usize, elapsed_ms: f64) {
    let profile = HAL_PROFILE.get_or_init(|| Mutex::new(HalProfileAccumulator::default()));
    let mut profile = profile.lock().expect("HAL profile mutex poisoned");
    update_hal_profile_entry(&mut profile.entries, op, bytes, elapsed_ms);
    for capture in profile.captures.values_mut() {
        update_hal_profile_entry(capture, op, bytes, elapsed_ms);
    }
}

pub(crate) fn hal_profile_time<T, F>(op: &'static str, bytes: usize, f: F) -> T
where
    F: FnOnce() -> T,
{
    if !hal_profile_enabled() {
        return f();
    }
    let start = Instant::now();
    let result = f();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    record_enabled_hal_profile_sample(op, bytes, elapsed_ms);
    result
}

fn ensure_device_ordinal(ordinal: usize) -> Result<()> {
    if ordinal != 0 {
        return Err(GpuError::InvalidArg(format!(
            "Metal backend supports only device ordinal 0 (got {ordinal})"
        )));
    }
    Ok(())
}

pub(crate) fn with_device_impl<T>(
    backend: Backend,
    ordinal: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _ = backend;
    ensure_device_ordinal(ordinal)?;
    f()
}

pub fn set_device(ordinal: usize) -> Result<()> {
    ensure_device_ordinal(ordinal)
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AllocatorKind {
    Discrete,
}

pub(crate) fn alloc(
    ordinal: usize,
    len_bytes: usize,
    kind: BufferKind,
) -> Result<(NonNull<c_void>, AllocatorKind)> {
    let _ = (current_strategy_for(kind),);
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg("allocation size must be > 0".into()));
    }
    hal_profile_time("alloc", len_bytes, || {
        with_device_impl(current_backend(), ordinal, || {
            let mut ptr = std::ptr::null_mut();
            let status = unsafe { supersonic_metal_alloc(len_bytes, &mut ptr) };
            if status != 0 {
                return Err(backend_error(
                    Backend::Metal,
                    "supersonic_metal_alloc",
                    status,
                ));
            }
            let nn = NonNull::new(ptr).ok_or_else(|| {
                GpuError::backend(Backend::Metal, "supersonic_metal_alloc returned null".into())
            })?;
            Ok((nn, AllocatorKind::Discrete))
        })
    })
}

pub fn alloc_host_pinned(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    alloc(ordinal, len_bytes, BufferKind::Scratch).map(|(ptr, _)| ptr)
}

pub fn host_pinned_device_ptr(
    backend: Backend,
    ordinal: usize,
    ptr: *mut c_void,
) -> Result<NonNull<c_void>> {
    let _ = (backend, ordinal);
    NonNull::new(ptr).ok_or_else(|| {
        GpuError::InvalidArg("host_pinned_device_ptr: null host pointer".into())
    })
}

pub fn free_host_pinned(backend: Backend, ordinal: usize, ptr: *mut c_void, _len_bytes: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = with_device_impl(backend, ordinal, || {
        let status = unsafe { supersonic_metal_free(ptr) };
        if status != 0 {
            return Err(backend_error(Backend::Metal, "supersonic_metal_free", status));
        }
        Ok(())
    });
}

pub struct PinnedHostBuffer {
    backend: Backend,
    ordinal: usize,
    ptr: NonNull<c_void>,
    len_bytes: usize,
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub fn new(ordinal: usize, len_bytes: usize) -> Result<Self> {
        let backend = current_backend();
        let ptr = alloc_host_pinned(ordinal, len_bytes)?;
        Ok(Self {
            backend,
            ordinal,
            ptr,
            len_bytes,
        })
    }

    pub fn len(&self) -> usize {
        self.len_bytes
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len_bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len_bytes) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        free_host_pinned(
            self.backend,
            self.ordinal,
            self.ptr.as_ptr(),
            self.len_bytes,
        );
    }
}

pub struct RegisteredHostBuffer {
    ptr: NonNull<c_void>,
    len_bytes: usize,
}

unsafe impl Send for RegisteredHostBuffer {}
unsafe impl Sync for RegisteredHostBuffer {}

impl RegisteredHostBuffer {
    pub unsafe fn new(ordinal: usize, ptr: *mut c_void, len_bytes: usize) -> Result<Self> {
        ensure_device_ordinal(ordinal)?;
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            GpuError::InvalidArg("RegisteredHostBuffer::new: null host pointer".into())
        })?;
        if len_bytes == 0 {
            return Err(GpuError::InvalidArg(
                "RegisteredHostBuffer::new: len_bytes must be > 0".into(),
            ));
        }
        Ok(Self { ptr, len_bytes })
    }

    pub fn len(&self) -> usize {
        self.len_bytes
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }
}

impl Drop for RegisteredHostBuffer {
    fn drop(&mut self) {}
}

pub(crate) fn alloc_zeros(
    ordinal: usize,
    len_bytes: usize,
    kind: BufferKind,
) -> Result<(NonNull<c_void>, AllocatorKind)> {
    let (ptr, allocator) = alloc(ordinal, len_bytes, kind)?;
    memset_zeros(ordinal, ptr.as_ptr(), len_bytes)?;
    Ok((ptr, allocator))
}

pub(crate) fn free(
    backend: Backend,
    ordinal: usize,
    dev_ptr: *mut c_void,
    _allocator: AllocatorKind,
) {
    if dev_ptr.is_null() {
        return;
    }
    let _ = hal_profile_time("free", 0, || {
        with_device_impl(backend, ordinal, || {
            let status = unsafe { supersonic_metal_free(dev_ptr) };
            if status != 0 {
                return Err(backend_error(Backend::Metal, "supersonic_metal_free", status));
            }
            Ok(())
        })
    });
}

fn memcpy_profile(op: &'static str, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(GpuError::InvalidArg(format!("{op}: null pointer")));
    }
    hal_profile_time(op, len, || {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
        }
        Ok(())
    })
}

pub fn copy_h2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    with_device_impl(current_backend(), ordinal, || memcpy_profile("copy_h2d", dst, src, len))
}

pub fn copy_h2d_async(
    ordinal: usize,
    stream: &GpuStream,
    dst: *mut c_void,
    src: *const c_void,
    len: usize,
) -> Result<()> {
    if stream.ordinal != ordinal {
        return Err(GpuError::InvalidArg(
            "copy_h2d_async requires stream on matching device".into(),
        ));
    }
    copy_h2d(ordinal, dst, src, len)
}

pub fn storage_to_device_is_supported(_backend: Backend) -> bool {
    false
}

pub(crate) fn ensure_storage_to_device_supported(
    backend: Backend,
    _ordinal: usize,
) -> Result<()> {
    Err(GpuError::Unsupported(format!(
        "direct storage reads are not supported on {backend}"
    )))
}

pub fn copy_storage_to_device(
    _ordinal: usize,
    _path: &Path,
    _dst: *mut c_void,
    _source_offset: u64,
    _len: usize,
) -> Result<()> {
    Err(GpuError::Unsupported(
        "copy_storage_to_device is not supported on Metal".into(),
    ))
}

pub fn copy_d2h(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    with_device_impl(current_backend(), ordinal, || {
        metal_dispatch_wait()?;
        memcpy_profile("copy_d2h", dst, src, len)
    })
}

pub fn copy_d2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    with_device_impl(current_backend(), ordinal, || memcpy_profile("copy_d2d", dst, src, len))
}

pub fn memset_zeros(ordinal: usize, dst: *mut c_void, len: usize) -> Result<()> {
    if dst.is_null() {
        return Err(GpuError::InvalidArg("memset_zeros: null pointer".into()));
    }
    with_device_impl(current_backend(), ordinal, || {
        hal_profile_time("memset_zeros", len, || {
            unsafe {
                std::ptr::write_bytes(dst as *mut u8, 0, len);
            }
            Ok(())
        })
    })
}

pub fn memset_zeros_async(
    ordinal: usize,
    stream: &GpuStream,
    dst: *mut c_void,
    len: usize,
) -> Result<()> {
    if stream.ordinal != ordinal {
        return Err(GpuError::InvalidArg(
            "memset_zeros_async requires stream on matching device".into(),
        ));
    }
    memset_zeros(ordinal, dst, len)
}

pub fn sync(ordinal: usize) -> Result<()> {
    with_device_impl(current_backend(), ordinal, || {
        hal_profile_time("sync", 0, || metal_dispatch_wait())
    })
}

fn metal_dispatch_wait() -> Result<()> {
    unsafe {
        supersonic_metal_dispatch_wait();
    }
    Ok(())
}

pub struct GpuEvent {
    backend: Backend,
    ordinal: usize,
}

unsafe impl Send for GpuEvent {}

pub struct GpuStream {
    backend: Backend,
    ordinal: usize,
}

unsafe impl Send for GpuStream {}

impl GpuStream {
    pub fn new_nonblocking(ordinal: usize) -> Result<Self> {
        ensure_device_ordinal(ordinal)?;
        Ok(Self {
            backend: current_backend(),
            ordinal,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        sync(self.ordinal)
    }

    pub fn wait_event(&self, event: &GpuEvent) -> Result<()> {
        if self.backend != event.backend || self.ordinal != event.ordinal {
            return Err(GpuError::InvalidArg(
                "GpuStream::wait_event requires matching backend/device".into(),
            ));
        }
        Ok(())
    }
}

impl Drop for GpuStream {
    fn drop(&mut self) {}
}

impl GpuEvent {
    pub fn new(ordinal: usize) -> Result<Self> {
        ensure_device_ordinal(ordinal)?;
        Ok(Self {
            backend: current_backend(),
            ordinal,
        })
    }

    pub fn record(&self) -> Result<()> {
        Ok(())
    }

    pub fn record_on_stream(&self, stream: &GpuStream) -> Result<()> {
        if self.backend != stream.backend || self.ordinal != stream.ordinal {
            return Err(GpuError::InvalidArg(
                "GpuEvent::record_on_stream requires matching backend/device".into(),
            ));
        }
        Ok(())
    }

    pub fn query(&self) -> Result<bool> {
        Ok(true)
    }

    pub fn synchronize(&self) -> Result<()> {
        sync(self.ordinal)
    }

    pub fn elapsed_ms(_start: &GpuEvent, _end: &GpuEvent) -> Result<f32> {
        Ok(0.0)
    }
}

impl Drop for GpuEvent {
    fn drop(&mut self) {}
}

pub fn query_device_info(backend: Backend, ordinal: usize) -> Result<DeviceInfo> {
    ensure_device_ordinal(ordinal)?;
    let mut arch_name = vec![0i8; 128];
    let mut total_vram = 0u64;
    let mut warp_size = 0u32;
    let mut clock_rate_khz = 0u32;
    let status = unsafe {
        supersonic_metal_query_device_info(
            ordinal,
            arch_name.as_mut_ptr(),
            arch_name.len(),
            &mut total_vram,
            &mut warp_size,
            &mut clock_rate_khz,
        )
    };
    if status != 0 {
        return Err(backend_error(
            Backend::Metal,
            "supersonic_metal_query_device_info",
            status,
        ));
    }
    let arch_bytes: Vec<u8> = arch_name
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte as u8)
        .collect();
    let arch_name = String::from_utf8_lossy(&arch_bytes).into_owned();
    let _ = backend;
    Ok(DeviceInfo {
        arch_name,
        total_vram_bytes: total_vram,
        warp_size,
        clock_rate_khz,
    })
}

pub fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn byte_len(dtype: ScalarType, elems: usize) -> usize {
    elems * dtype.size_in_bytes()
}
