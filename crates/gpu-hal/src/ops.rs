use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::BTreeMap;
#[cfg(supersonic_backend_hipfile)]
use std::ffi::{c_char, CStr, CString};
use std::ffi::{c_int, c_void};
#[cfg(all(unix, supersonic_backend_hipfile))]
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[cfg(supersonic_backend_hip)]
use crate::backend::AllocStrategy;
use crate::backend::{current_backend, current_strategy_for, Backend, BufferKind, DeviceInfo};
use crate::error::{backend_error, GpuError, Result};
#[cfg(supersonic_backend_hip)]
use crate::hip_sys::*;
use crate::scalar_type::ScalarType;

#[cfg(supersonic_backend_hipfile)]
unsafe extern "C" {
    fn supersonic_hipfile_read_to_device(
        ordinal: c_int,
        path: *const c_char,
        dst: *mut c_void,
        source_offset: u64,
        len: usize,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> c_int;
}

static HAL_PROFILE_EXPLICIT_ENABLED: AtomicBool = AtomicBool::new(false);
static HAL_PROFILE_ACTIVE_CAPTURES: AtomicUsize = AtomicUsize::new(0);
static HAL_PROFILE: OnceLock<Mutex<HalProfileAccumulator>> = OnceLock::new();
const STORAGE_DIRECT_BLOCK_ALIGNMENT: usize = 4096;

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
            "free" => {
                snapshot.free_calls += entry.calls;
            }
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

#[cfg(test)]
fn record_hal_profile_sample(op: &'static str, bytes: usize, elapsed_ms: f64) {
    if !hal_profile_enabled() {
        return;
    }
    record_enabled_hal_profile_sample(op, bytes, elapsed_ms);
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

pub(crate) fn with_device_impl<T>(
    backend: Backend,
    ordinal: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    #[allow(unused_mut)]
    let mut prev = 0;
    match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            {
                let status = unsafe { hipGetDevice(&mut prev) };
                if status != 0 {
                    return Err(backend_error(Backend::Hip, "hipGetDevice", status));
                }
            }
            #[cfg(not(supersonic_backend_hip))]
            return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
        }
        Backend::Cuda => {
            #[cfg(any())]
            {
                let status = unsafe { cudaGetDevice(&mut prev) };
                if status != 0 {
                    return Err(backend_error(Backend::Cuda, "cudaGetDevice", status));
                }
            }
            #[cfg(not(any()))]
            return Err(GpuError::InvalidArg("CUDA backend not compiled".into()));
        }
        Backend::Metal => {
            if ordinal != 0 {
                return Err(GpuError::InvalidArg(
                    "Metal backend currently supports only device ordinal 0".into(),
                ));
            }
        }
    }
    let restore = if prev != ordinal_i32 {
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipSetDevice(ordinal_i32)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(any())]
                unsafe {
                    cudaSetDevice(ordinal_i32)
                }
                #[cfg(not(any()))]
                1
            }
            Backend::Metal => 0,
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => backend_error(Backend::Hip, "hipSetDevice", status),
                Backend::Cuda => backend_error(Backend::Cuda, "cudaSetDevice", status),
                Backend::Metal => backend_error(Backend::Metal, "metalSetDevice", status),
            });
        }
        Some(prev)
    } else {
        None
    };
    let result = f();
    if let Some(prev) = restore {
        let _ = prev;
        let status = match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    hipSetDevice(prev)
                }
                #[cfg(not(supersonic_backend_hip))]
                1
            }
            Backend::Cuda => {
                #[cfg(any())]
                unsafe {
                    cudaSetDevice(prev)
                }
                #[cfg(not(any()))]
                1
            }
            Backend::Metal => 0,
        };
        if status != 0 {
            return Err(match backend {
                Backend::Hip => backend_error(Backend::Hip, "hipSetDevice(restore)", status),
                Backend::Cuda => backend_error(Backend::Cuda, "cudaSetDevice(restore)", status),
                Backend::Metal => backend_error(Backend::Metal, "metalSetDevice(restore)", status),
            });
        }
    }
    result
}

pub fn set_device(ordinal: usize) -> Result<()> {
    let backend = current_backend();
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    let _ = ordinal_i32;
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                hipSetDevice(ordinal_i32)
            }
            #[cfg(not(supersonic_backend_hip))]
            1
        }
        Backend::Cuda => {
            #[cfg(any())]
            unsafe {
                cudaSetDevice(ordinal_i32)
            }
            #[cfg(not(any()))]
            1
        }
        Backend::Metal => {
            if ordinal == 0 {
                0
            } else {
                1
            }
        }
    };
    if status != 0 {
        return Err(match backend {
            Backend::Hip => backend_error(Backend::Hip, "hipSetDevice", status),
            Backend::Cuda => backend_error(Backend::Cuda, "cudaSetDevice", status),
            Backend::Metal => GpuError::InvalidArg(
                "Metal backend currently supports only device ordinal 0".into(),
            ),
        });
    }
    Ok(())
}

/// Distinguishes which underlying allocator produced a buffer pointer, so the
/// matching `free` call can be issued at drop time. Internal coordination
/// type between [`alloc`] and [`free`].
///
/// `UnifiedHost` carries the original host pointer separately from the
/// device-mapped pointer: `hipHostMalloc` returns a host pointer, then
/// `hipHostGetDevicePointer` produces a device-addressable pointer that may
/// or may not equal the host one. The buffer stores the device pointer for
/// kernel ops; the host pointer is what `hipHostFree` needs at drop time.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AllocatorKind {
    /// Classic device-pointer allocation: `hipMalloc` / `cudaMalloc` /
    /// `supersonic_metal_alloc`. Free with the matching `*Free` on the
    /// device pointer.
    Discrete,
    /// HIP host-mapped allocation (`hipHostMalloc` with
    /// `HIP_HOST_MALLOC_MAPPED`, *no* coherent flag — coherence-protocol
    /// traffic is the bottleneck on RDNA3.5 APU's bandwidth-bound decode
    /// path, and weights are write-once-from-baker / read-many-from-decode
    /// so coherence buys nothing). The pointer addresses system RAM directly.
    /// Free with `hipHostFree(host_ptr)`.
    #[allow(dead_code)]
    UnifiedHost { host_ptr: NonNull<c_void> },
}

/// Allocate `len_bytes` of device-addressable memory, returning a non-null
/// pointer plus the allocator kind that produced it. The active
/// `BufferPolicy` resolves the supplied `BufferKind` to an `AllocStrategy`:
///
/// - `AllocStrategy::Default` → `hipMalloc` / `cudaMalloc` /
///   `supersonic_metal_alloc`. Device-resident, GPU-cacheable, classic.
/// - `AllocStrategy::HostMapped` (HIP only) → `hipHostMalloc(MAPPED) +
///   hipHostGetDevicePointer`. System-RAM-resident, zero-copy from host,
///   but **bypasses GPU L2 on RDNA3.5 APUs** — only choose this for
///   one-shot scratch buffers with no cache reuse. See
///   `docs/gfx1150-l2-bypass.md` for the measurement.
///
/// Default policy is `{Persistent: Default, Scratch: Default}` until
/// startup wiring installs the per-arch table; this preserves classic
/// alloc behavior for any code path that runs early.
pub(crate) fn alloc(
    ordinal: usize,
    len_bytes: usize,
    kind: BufferKind,
) -> Result<(NonNull<c_void>, AllocatorKind)> {
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg("allocation size must be > 0".into()));
    }
    #[allow(unused_variables)]
    let strategy = current_strategy_for(kind);
    hal_profile_time("alloc", len_bytes, || {
        let backend = current_backend();
        with_device_impl(backend, ordinal, || match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                unsafe {
                    if strategy == AllocStrategy::HostMapped {
                        let mut host_ptr = std::ptr::null_mut();
                        let status =
                            hipHostMalloc(&mut host_ptr, len_bytes, HIP_HOST_MALLOC_MAPPED);
                        if status != 0 {
                            return Err(backend_error(
                                Backend::Hip,
                                "hipHostMalloc(unified)",
                                status,
                            ));
                        }
                        let host_nn = NonNull::new(host_ptr).ok_or_else(|| {
                            GpuError::backend(Backend::Hip, "hipHostMalloc returned null".into())
                        })?;
                        let mut dev_ptr = std::ptr::null_mut();
                        let status = hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0);
                        if status != 0 {
                            // Roll back the host alloc so we don't leak.
                            let _ = hipHostFree(host_ptr);
                            return Err(backend_error(
                                Backend::Hip,
                                "hipHostGetDevicePointer",
                                status,
                            ));
                        }
                        let dev_nn = NonNull::new(dev_ptr).ok_or_else(|| {
                            // Same rollback on the unlikely null device ptr.
                            let _ = hipHostFree(host_ptr);
                            GpuError::backend(
                                Backend::Hip,
                                "hipHostGetDevicePointer returned null".into(),
                            )
                        })?;
                        return Ok((dev_nn, AllocatorKind::UnifiedHost { host_ptr: host_nn }));
                    }
                    let mut ptr = std::ptr::null_mut();
                    let status = hipMalloc(&mut ptr, len_bytes);
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipMalloc", status));
                    }
                    let nn = NonNull::new(ptr).ok_or_else(|| {
                        GpuError::backend(Backend::Hip, "hipMalloc returned null".into())
                    })?;
                    Ok((nn, AllocatorKind::Discrete))
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => {
                #[cfg(any())]
                unsafe {
                    let mut ptr = std::ptr::null_mut();
                    let status = cudaMalloc(&mut ptr, len_bytes);
                    if status != 0 {
                        return Err(backend_error(Backend::Cuda, "cudaMalloc", status));
                    }
                    let nn = NonNull::new(ptr).ok_or_else(|| {
                        GpuError::backend(Backend::Cuda, "cudaMalloc returned null".into())
                    })?;
                    Ok((nn, AllocatorKind::Discrete))
                }
                #[cfg(not(any()))]
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
            Backend::Metal => {
                #[cfg(any())]
                unsafe {
                    let mut ptr = std::ptr::null_mut();
                    let status = supersonic_metal_alloc(len_bytes, &mut ptr);
                    if status != 0 {
                        return Err(backend_error(Backend::Metal, "metalAlloc", status));
                    }
                    let nn = NonNull::new(ptr).ok_or_else(|| {
                        GpuError::backend(Backend::Metal, "metalAlloc returned null".into())
                    })?;
                    Ok((nn, AllocatorKind::Discrete))
                }
                #[cfg(not(any()))]
                Err(GpuError::InvalidArg("Metal backend not compiled".into()))
            }
        })
    })
}

/// Allocate host memory suitable for fast host-to-device page-in.
pub fn alloc_host_pinned(ordinal: usize, len_bytes: usize) -> Result<NonNull<c_void>> {
    let backend = current_backend();
    if len_bytes == 0 {
        return Err(GpuError::InvalidArg(
            "host allocation size must be > 0".into(),
        ));
    }
    match backend {
        Backend::Cuda => with_device_impl(backend, ordinal, || {
            #[cfg(any())]
            {
                let mut ptr = std::ptr::null_mut();
                const CUDA_HOST_ALLOC_MAPPED: u32 = 0x02;
                let status = unsafe { cudaHostAlloc(&mut ptr, len_bytes, CUDA_HOST_ALLOC_MAPPED) };
                if status != 0 {
                    return Err(backend_error(Backend::Cuda, "cudaHostAlloc", status));
                }
                NonNull::new(ptr).ok_or_else(|| {
                    GpuError::backend(Backend::Cuda, "cudaHostAlloc returned null".into())
                })
            }
            #[cfg(not(any()))]
            Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
        }),
        Backend::Hip => with_device_impl(backend, ordinal, || {
            #[cfg(supersonic_backend_hip)]
            {
                let mut ptr = std::ptr::null_mut();
                let status = unsafe { hipHostMalloc(&mut ptr, len_bytes, 0) };
                if status != 0 {
                    return Err(backend_error(Backend::Hip, "hipHostMalloc", status));
                }
                NonNull::new(ptr).ok_or_else(|| {
                    GpuError::backend(Backend::Hip, "hipHostMalloc returned null".into())
                })
            }
            #[cfg(not(supersonic_backend_hip))]
            Err(GpuError::InvalidArg("HIP backend not compiled".into()))
        }),
        Backend::Metal => {
            let layout = Layout::from_size_align(len_bytes, 64)
                .map_err(|e| GpuError::InvalidArg(format!("host allocation layout failed: {e}")))?;
            let ptr = unsafe { alloc_zeroed(layout) as *mut c_void };
            NonNull::new(ptr).ok_or_else(|| {
                GpuError::backend(Backend::Metal, "host allocation returned null".into())
            })
        }
    }
}

/// Return the device-visible pointer for mapped pinned host memory.
pub fn host_pinned_device_ptr(
    backend: Backend,
    ordinal: usize,
    ptr: *mut c_void,
) -> Result<NonNull<c_void>> {
    if ptr.is_null() {
        return Err(GpuError::InvalidArg(
            "host_pinned_device_ptr: null host pointer".into(),
        ));
    }
    match backend {
        Backend::Cuda => with_device_impl(backend, ordinal, || {
            #[cfg(any())]
            {
                let mut device_ptr = std::ptr::null_mut();
                let status = unsafe { cudaHostGetDevicePointer(&mut device_ptr, ptr, 0) };
                if status != 0 {
                    return Err(backend_error(
                        Backend::Cuda,
                        "cudaHostGetDevicePointer",
                        status,
                    ));
                }
                NonNull::new(device_ptr).ok_or_else(|| {
                    GpuError::backend(
                        Backend::Cuda,
                        "cudaHostGetDevicePointer returned null".into(),
                    )
                })
            }
            #[cfg(not(any()))]
            Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
        }),
        Backend::Hip | Backend::Metal => NonNull::new(ptr)
            .ok_or_else(|| GpuError::InvalidArg("host_pinned_device_ptr returned null".into())),
    }
}

/// Free host memory allocated by `alloc_host_pinned`.
pub fn free_host_pinned(backend: Backend, ordinal: usize, ptr: *mut c_void, len_bytes: usize) {
    if ptr.is_null() {
        return;
    }
    match backend {
        Backend::Cuda => {
            let _: Result<()> = with_device_impl(backend, ordinal, || {
                #[cfg(any())]
                {
                    let status = unsafe { cudaFreeHost(ptr) };
                    if status != 0 {
                        return Err(backend_error(Backend::Cuda, "cudaFreeHost", status));
                    }
                    Ok(())
                }
                #[cfg(not(any()))]
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            });
        }
        Backend::Hip => {
            let _: Result<()> = with_device_impl(backend, ordinal, || {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipHostFree(ptr) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipHostFree", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            });
        }
        Backend::Metal => {
            if let Ok(layout) = Layout::from_size_align(len_bytes, 64) {
                unsafe { dealloc(ptr as *mut u8, layout) };
            }
        }
    }
}

/// Safe RAII wrapper around backend pinned host memory.
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

/// RAII wrapper around externally owned host memory registered for faster H2D.
///
/// The wrapper does not own the underlying bytes. It only owns the backend
/// registration and unregisters the exact pointer at drop time.
pub struct RegisteredHostBuffer {
    backend: Backend,
    ordinal: usize,
    ptr: NonNull<c_void>,
    len_bytes: usize,
}

unsafe impl Send for RegisteredHostBuffer {}
unsafe impl Sync for RegisteredHostBuffer {}

impl RegisteredHostBuffer {
    /// Register an externally owned host range with the active backend.
    ///
    /// # Safety
    ///
    /// `ptr..ptr+len_bytes` must stay valid and mapped for the lifetime of the
    /// returned wrapper, and it must satisfy the backend's host-registration
    /// alignment requirements. The same range must not be concurrently
    /// unregistered elsewhere.
    pub unsafe fn new(ordinal: usize, ptr: *mut c_void, len_bytes: usize) -> Result<Self> {
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            GpuError::InvalidArg("RegisteredHostBuffer::new: null host pointer".into())
        })?;
        if len_bytes == 0 {
            return Err(GpuError::InvalidArg(
                "RegisteredHostBuffer::new: len_bytes must be > 0".into(),
            ));
        }
        let backend = current_backend();
        hal_profile_time("host_register", len_bytes, || {
            with_device_impl(backend, ordinal, || match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    {
                        let status = unsafe { hipHostRegister(ptr.as_ptr(), len_bytes, 0) };
                        if status != 0 {
                            return Err(backend_error(Backend::Hip, "hipHostRegister", status));
                        }
                        Ok(())
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    Err(GpuError::InvalidArg("HIP backend not compiled".into()))
                }
                Backend::Cuda => Err(GpuError::Unsupported(
                    "RegisteredHostBuffer is not implemented for CUDA yet".into(),
                )),
                Backend::Metal => Err(GpuError::Unsupported(
                    "RegisteredHostBuffer is not implemented for Metal".into(),
                )),
            })
        })?;
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
}

impl Drop for RegisteredHostBuffer {
    fn drop(&mut self) {
        let backend = self.backend;
        let ordinal = self.ordinal;
        let ptr = self.ptr.as_ptr();
        let len_bytes = self.len_bytes;
        let _ = hal_profile_time("host_unregister", len_bytes, || {
            with_device_impl(backend, ordinal, || match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    {
                        let status = unsafe { hipHostUnregister(ptr) };
                        if status != 0 {
                            return Err(backend_error(Backend::Hip, "hipHostUnregister", status));
                        }
                        Ok(())
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    Err(GpuError::InvalidArg("HIP backend not compiled".into()))
                }
                Backend::Cuda | Backend::Metal => Ok(()),
            })
        });
    }
}

/// Allocate `len_bytes` of device memory, zeroed. Same allocator-dispatch
/// behavior as [`alloc`].
pub(crate) fn alloc_zeros(
    ordinal: usize,
    len_bytes: usize,
    kind: BufferKind,
) -> Result<(NonNull<c_void>, AllocatorKind)> {
    let (ptr, allocator) = alloc(ordinal, len_bytes, kind)?;
    memset_zeros(ordinal, ptr.as_ptr(), len_bytes)?;
    Ok((ptr, allocator))
}

/// Free a buffer allocated by [`alloc`]. Dispatches based on the recorded
/// allocator kind: `Discrete` frees the device pointer with `hipFree` /
/// `cudaFree` / metal-free; `UnifiedHost` frees the original host pointer
/// (carried in the kind) with `hipHostFree` and ignores the device-mapped
/// pointer. No-op on null.
pub(crate) fn free(
    backend: Backend,
    ordinal: usize,
    dev_ptr: *mut c_void,
    allocator: AllocatorKind,
) {
    if dev_ptr.is_null() {
        return;
    }
    hal_profile_time("free", 0, || {
        let _ = with_device_impl(backend, ordinal, || {
            let status = match (backend, allocator) {
                (Backend::Hip, AllocatorKind::UnifiedHost { host_ptr }) => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipHostFree(host_ptr.as_ptr())
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    {
                        let _ = host_ptr;
                        1
                    }
                }
                (Backend::Hip, AllocatorKind::Discrete) => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipFree(dev_ptr)
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    1
                }
                (Backend::Cuda, _) => {
                    #[cfg(any())]
                    unsafe {
                        cudaFree(dev_ptr)
                    }
                    #[cfg(not(any()))]
                    1
                }
                (Backend::Metal, _) => {
                    #[cfg(any())]
                    unsafe {
                        supersonic_metal_free(dev_ptr)
                    }
                    #[cfg(not(any()))]
                    1
                }
            };
            if status != 0 {
                return Err(match (backend, allocator) {
                    (Backend::Hip, AllocatorKind::UnifiedHost { .. }) => {
                        backend_error(Backend::Hip, "hipHostFree", status)
                    }
                    (Backend::Hip, AllocatorKind::Discrete) => {
                        backend_error(Backend::Hip, "hipFree", status)
                    }
                    (Backend::Cuda, _) => backend_error(Backend::Cuda, "cudaFree", status),
                    (Backend::Metal, _) => backend_error(Backend::Metal, "metalFree", status),
                });
            }
            Ok(())
        });
    });
}

/// Copy from host memory to device memory.
pub fn copy_h2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_h2d: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("copy_h2d", len, || {
        let backend = current_backend();
        with_device_impl(backend, ordinal, || {
            let status = match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipMemcpy(dst, src, len, HIP_MEMCPY_HOST_TO_DEVICE)
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    1
                }
                Backend::Cuda => {
                    #[cfg(any())]
                    unsafe {
                        cudaMemcpy(dst, src, len, CUDA_MEMCPY_HOST_TO_DEVICE)
                    }
                    #[cfg(not(any()))]
                    1
                }
                Backend::Metal => {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
                    }
                    0
                }
            };
            if status != 0 {
                return Err(match backend {
                    Backend::Hip => backend_error(Backend::Hip, "hipMemcpy(H2D)", status),
                    Backend::Cuda => backend_error(Backend::Cuda, "cudaMemcpy(H2D)", status),
                    Backend::Metal => backend_error(Backend::Metal, "metalMemcpy(H2D)", status),
                });
            }
            Ok(())
        })
    })
}

pub fn copy_h2d_async(
    ordinal: usize,
    stream: &GpuStream,
    dst: *mut c_void,
    src: *const c_void,
    len: usize,
) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_h2d_async: null pointer or zero len".into(),
        ));
    }
    if stream.ordinal != ordinal {
        return Err(GpuError::InvalidArg(
            "copy_h2d_async requires stream on matching device".into(),
        ));
    }
    hal_profile_time("copy_h2d_async", len, || {
        with_device_impl(stream.backend, ordinal, || match stream.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe {
                        hipMemcpyAsync(dst, src, len, HIP_MEMCPY_HOST_TO_DEVICE, stream.raw)
                    };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipMemcpyAsync(H2D)", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "copy_h2d_async is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "copy_h2d_async is not implemented for Metal".into(),
            )),
        })
    })
}

pub fn storage_to_device_is_supported(backend: Backend) -> bool {
    match backend {
        Backend::Hip => hipfile_is_compiled(),
        Backend::Cuda | Backend::Metal => false,
    }
}

#[cfg(supersonic_backend_hipfile)]
fn hipfile_is_compiled() -> bool {
    true
}

#[cfg(not(supersonic_backend_hipfile))]
fn hipfile_is_compiled() -> bool {
    false
}

pub(crate) fn ensure_storage_to_device_supported(
    backend: Backend,
    source_path: &Path,
    source_offset: u64,
    len: usize,
) -> Result<()> {
    if storage_to_device_is_supported(backend) {
        return Ok(());
    }
    let reason = match backend {
        Backend::Hip => {
            "hipFile support is not compiled; ROCm >= 7.2 with hipfile.h and libhipfile is required"
        }
        Backend::Cuda => "CUDA storage-to-device transfer is not implemented yet",
        Backend::Metal => "Metal storage-to-device transfer is not implemented",
    };
    Err(GpuError::Unsupported(format!(
        "GPU-direct storage-to-device transfer is not available for {backend}: {reason} \
         (source={} offset={} len={})",
        source_path.display(),
        source_offset,
        len
    )))
}

pub fn copy_storage_to_device(
    ordinal: usize,
    dst: *mut c_void,
    source_path: &Path,
    source_offset: u64,
    len: usize,
) -> Result<()> {
    if dst.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_storage_to_device: null pointer or zero len".into(),
        ));
    }
    if source_path.as_os_str().is_empty() {
        return Err(GpuError::InvalidArg(
            "copy_storage_to_device: source path must not be empty".into(),
        ));
    }
    if source_offset % STORAGE_DIRECT_BLOCK_ALIGNMENT as u64 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "copy_storage_to_device: source offset {source_offset} is not {STORAGE_DIRECT_BLOCK_ALIGNMENT}-byte aligned"
        )));
    }
    if len % STORAGE_DIRECT_BLOCK_ALIGNMENT != 0 {
        return Err(GpuError::InvalidArg(format!(
            "copy_storage_to_device: length {len} is not {STORAGE_DIRECT_BLOCK_ALIGNMENT}-byte aligned"
        )));
    }
    let backend = current_backend();
    ensure_storage_to_device_supported(backend, source_path, source_offset, len)?;
    hal_profile_time("copy_storage_to_device", len, || {
        with_device_impl(backend, ordinal, || match backend {
            Backend::Hip => {
                copy_storage_to_device_hipfile(ordinal, dst, source_path, source_offset, len)
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "copy_storage_to_device is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "copy_storage_to_device is not implemented for Metal".into(),
            )),
        })
    })
}

#[cfg(supersonic_backend_hipfile)]
fn copy_storage_to_device_hipfile(
    ordinal: usize,
    dst: *mut c_void,
    source_path: &Path,
    source_offset: u64,
    len: usize,
) -> Result<()> {
    #[cfg(unix)]
    let path_bytes = source_path.as_os_str().as_bytes();
    #[cfg(not(unix))]
    let path_bytes = source_path
        .to_str()
        .ok_or_else(|| GpuError::InvalidArg("hipFile source path is not valid UTF-8".into()))?
        .as_bytes();
    let path = CString::new(path_bytes).map_err(|_| {
        GpuError::InvalidArg(format!(
            "hipFile source path contains an interior NUL: {}",
            source_path.display()
        ))
    })?;
    let mut err_buf = vec![0i8; 512];
    let status = unsafe {
        supersonic_hipfile_read_to_device(
            ordinal as c_int,
            path.as_ptr(),
            dst,
            source_offset,
            len,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(GpuError::backend(
        Backend::Hip,
        format!("hipFile storage-to-device transfer failed: {message}"),
    ))
}

#[cfg(not(supersonic_backend_hipfile))]
fn copy_storage_to_device_hipfile(
    _ordinal: usize,
    _dst: *mut c_void,
    _source_path: &Path,
    _source_offset: u64,
    _len: usize,
) -> Result<()> {
    Err(GpuError::Unsupported(
        "hipFile support is not compiled; ROCm >= 7.2 with hipfile.h and libhipfile is required"
            .into(),
    ))
}

/// Copy from device memory to host memory.
pub fn copy_d2h(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_d2h: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("copy_d2h", len, || {
        let backend = current_backend();
        with_device_impl(backend, ordinal, || {
            let status = match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipMemcpy(dst, src, len, HIP_MEMCPY_DEVICE_TO_HOST)
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    1
                }
                Backend::Cuda => {
                    #[cfg(any())]
                    unsafe {
                        cudaMemcpy(dst, src, len, CUDA_MEMCPY_DEVICE_TO_HOST)
                    }
                    #[cfg(not(any()))]
                    1
                }
                Backend::Metal => {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len);
                    }
                    0
                }
            };
            if status != 0 {
                return Err(match backend {
                    Backend::Hip => backend_error(Backend::Hip, "hipMemcpy(D2H)", status),
                    Backend::Cuda => backend_error(Backend::Cuda, "cudaMemcpy(D2H)", status),
                    Backend::Metal => backend_error(Backend::Metal, "metalMemcpy(D2H)", status),
                });
            }
            Ok(())
        })
    })
}

/// Copy from device memory to device memory.
pub fn copy_d2d(ordinal: usize, dst: *mut c_void, src: *const c_void, len: usize) -> Result<()> {
    if dst.is_null() || src.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "copy_d2d: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("copy_d2d", len, || {
        let backend = current_backend();
        with_device_impl(backend, ordinal, || {
            let status = match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipMemcpy(dst, src, len, HIP_MEMCPY_DEVICE_TO_DEVICE)
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    1
                }
                Backend::Cuda => {
                    #[cfg(any())]
                    unsafe {
                        cudaMemcpy(dst, src, len, CUDA_MEMCPY_DEVICE_TO_DEVICE)
                    }
                    #[cfg(not(any()))]
                    1
                }
                Backend::Metal => {
                    unsafe {
                        std::ptr::copy(src as *const u8, dst as *mut u8, len);
                    }
                    0
                }
            };
            if status != 0 {
                return Err(match backend {
                    Backend::Hip => backend_error(Backend::Hip, "hipMemcpy(D2D)", status),
                    Backend::Cuda => backend_error(Backend::Cuda, "cudaMemcpy(D2D)", status),
                    Backend::Metal => backend_error(Backend::Metal, "metalMemcpy(D2D)", status),
                });
            }
            Ok(())
        })
    })
}

/// Set device memory to zero.
pub fn memset_zeros(ordinal: usize, dst: *mut c_void, len: usize) -> Result<()> {
    if dst.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "memset_zeros: null pointer or zero len".into(),
        ));
    }
    hal_profile_time("memset_zeros", len, || {
        let backend = current_backend();
        with_device_impl(backend, ordinal, || {
            let status = match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipMemset(dst, 0, len)
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    1
                }
                Backend::Cuda => {
                    #[cfg(any())]
                    unsafe {
                        cudaMemset(dst, 0, len)
                    }
                    #[cfg(not(any()))]
                    1
                }
                Backend::Metal => {
                    unsafe {
                        std::ptr::write_bytes(dst as *mut u8, 0, len);
                    }
                    0
                }
            };
            if status != 0 {
                return Err(match backend {
                    Backend::Hip => backend_error(Backend::Hip, "hipMemset", status),
                    Backend::Cuda => backend_error(Backend::Cuda, "cudaMemset", status),
                    Backend::Metal => backend_error(Backend::Metal, "metalMemset", status),
                });
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
    if dst.is_null() || len == 0 {
        return Err(GpuError::InvalidArg(
            "memset_zeros_async: null pointer or zero len".into(),
        ));
    }
    if stream.ordinal != ordinal {
        return Err(GpuError::InvalidArg(
            "memset_zeros_async requires stream on matching device".into(),
        ));
    }
    hal_profile_time("memset_zeros_async", len, || {
        with_device_impl(stream.backend, ordinal, || match stream.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipMemsetAsync(dst, 0, len, stream.raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipMemsetAsync", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "memset_zeros_async is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "memset_zeros_async is not implemented for Metal".into(),
            )),
        })
    })
}

/// Synchronize the device (block until all pending work completes).
pub fn sync(ordinal: usize) -> Result<()> {
    hal_profile_time("sync", 0, || {
        let backend = current_backend();
        with_device_impl(backend, ordinal, || {
            let status = match backend {
                Backend::Hip => {
                    #[cfg(supersonic_backend_hip)]
                    unsafe {
                        hipDeviceSynchronize()
                    }
                    #[cfg(not(supersonic_backend_hip))]
                    1
                }
                Backend::Cuda => {
                    #[cfg(any())]
                    unsafe {
                        cudaDeviceSynchronize()
                    }
                    #[cfg(not(any()))]
                    1
                }
                Backend::Metal => 0,
            };
            if status != 0 {
                return Err(match backend {
                    Backend::Hip => backend_error(Backend::Hip, "hipDeviceSynchronize", status),
                    Backend::Cuda => backend_error(Backend::Cuda, "cudaDeviceSynchronize", status),
                    Backend::Metal => {
                        backend_error(Backend::Metal, "metalDeviceSynchronize", status)
                    }
                });
            }
            Ok(())
        })
    })
}

/// RAII wrapper around a backend timing event.
///
/// Timing events are currently implemented only for HIP. On CUDA builds this
/// returns an explicit error until the matching runtime bindings are added.
pub struct GpuEvent {
    backend: Backend,
    ordinal: usize,
    raw: *mut c_void,
}

// SAFETY: `GpuEvent` exclusively owns its backend event handle. Operations and
// destruction reselect the recorded device before touching that handle, so
// transferring exclusive ownership to another thread is valid. It is not Sync.
unsafe impl Send for GpuEvent {}

/// RAII wrapper around a non-blocking backend stream.
pub struct GpuStream {
    backend: Backend,
    ordinal: usize,
    raw: *mut c_void,
}

// SAFETY: `GpuStream` exclusively owns its backend stream handle. Operations
// and destruction reselect the recorded device before touching that handle, so
// transferring exclusive ownership to another thread is valid. It is not Sync.
unsafe impl Send for GpuStream {}

impl GpuStream {
    pub fn new_nonblocking(ordinal: usize) -> Result<Self> {
        let backend = current_backend();
        let raw = with_device_impl(backend, ordinal, || match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let mut raw: *mut c_void = std::ptr::null_mut();
                    let status =
                        unsafe { hipStreamCreateWithFlags(&mut raw, HIP_STREAM_NON_BLOCKING) };
                    if status != 0 {
                        return Err(backend_error(
                            Backend::Hip,
                            "hipStreamCreateWithFlags",
                            status,
                        ));
                    }
                    Ok(raw)
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "GpuStream is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "GpuStream is not implemented for Metal".into(),
            )),
        })?;
        Ok(Self {
            backend,
            ordinal,
            raw,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipStreamSynchronize(self.raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipStreamSynchronize", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "GpuStream is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "GpuStream is not implemented for Metal".into(),
            )),
        })
    }

    pub fn wait_event(&self, event: &GpuEvent) -> Result<()> {
        if self.backend != event.backend || self.ordinal != event.ordinal {
            return Err(GpuError::InvalidArg(
                "GpuStream::wait_event requires matching backend/device".into(),
            ));
        }
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipStreamWaitEvent(self.raw, event.raw, 0) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipStreamWaitEvent", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "GpuStream::wait_event is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "GpuStream::wait_event is not implemented for Metal".into(),
            )),
        })
    }
}

impl Drop for GpuStream {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let _ = with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipStreamDestroy(self.raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipStreamDestroy", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda | Backend::Metal => Ok(()),
        });
    }
}

impl GpuEvent {
    pub fn new(ordinal: usize) -> Result<Self> {
        let backend = current_backend();
        #[allow(unused_mut)]
        let mut raw: *mut c_void = std::ptr::null_mut();
        with_device_impl(backend, ordinal, || match backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventCreate(&mut raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipEventCreate", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })?;
        Ok(Self {
            backend,
            ordinal,
            raw,
        })
    }

    pub fn record(&self) -> Result<()> {
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventRecord(self.raw, std::ptr::null_mut()) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipEventRecord", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })
    }

    pub fn record_on_stream(&self, stream: &GpuStream) -> Result<()> {
        if self.backend != stream.backend || self.ordinal != stream.ordinal {
            return Err(GpuError::InvalidArg(
                "GpuEvent::record_on_stream requires matching backend/device".into(),
            ));
        }
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventRecord(self.raw, stream.raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipEventRecord", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })
    }

    pub fn query(&self) -> Result<bool> {
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventQuery(self.raw) };
                    if status == 0 {
                        Ok(true)
                    } else if status == HIP_ERROR_NOT_READY {
                        Ok(false)
                    } else {
                        Err(backend_error(Backend::Hip, "hipEventQuery", status))
                    }
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::Unsupported(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::Unsupported(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventSynchronize(self.raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipEventSynchronize", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        })
    }

    pub fn elapsed_ms(start: &GpuEvent, end: &GpuEvent) -> Result<f32> {
        if start.backend != end.backend || start.ordinal != end.ordinal {
            return Err(GpuError::InvalidArg(
                "GpuEvent::elapsed_ms requires matching backend/device".into(),
            ));
        }
        match start.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let mut ms: f32 = 0.0;
                    with_device_impl(start.backend, start.ordinal, || {
                        let status = unsafe { hipEventElapsedTime(&mut ms, start.raw, end.raw) };
                        if status != 0 {
                            return Err(backend_error(Backend::Hip, "hipEventElapsedTime", status));
                        }
                        Ok(())
                    })?;
                    Ok(ms)
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for CUDA yet".into(),
            )),
            Backend::Metal => Err(GpuError::InvalidArg(
                "GpuEvent is not implemented for Metal yet".into(),
            )),
        }
    }
}

impl Drop for GpuEvent {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let _ = with_device_impl(self.backend, self.ordinal, || match self.backend {
            Backend::Hip => {
                #[cfg(supersonic_backend_hip)]
                {
                    let status = unsafe { hipEventDestroy(self.raw) };
                    if status != 0 {
                        return Err(backend_error(Backend::Hip, "hipEventDestroy", status));
                    }
                    Ok(())
                }
                #[cfg(not(supersonic_backend_hip))]
                Err(GpuError::InvalidArg("HIP backend not compiled".into()))
            }
            Backend::Cuda => Ok(()),
            Backend::Metal => Ok(()),
        });
    }
}

pub fn query_device_info(backend: Backend, ordinal: usize) -> Result<DeviceInfo> {
    let ordinal_i32 = c_int::try_from(ordinal)
        .map_err(|_| GpuError::InvalidArg(format!("device ordinal {ordinal} overflows c_int")))?;
    let _ = ordinal_i32;
    match backend {
        Backend::Hip => Err(GpuError::InvalidArg(
            "HIP device query is provided by the HIP kernel bridge, not gpu-hal".into(),
        )),
        Backend::Cuda => {
            #[cfg(any())]
            {
                let mut props = unsafe { std::mem::zeroed::<CudaDeviceProp>() };
                let status = unsafe { cudaGetDeviceProperties(&mut props, ordinal_i32) };
                if status != 0 {
                    return Err(backend_error(
                        Backend::Cuda,
                        "cudaGetDeviceProperties",
                        status,
                    ));
                }
                let arch_name = format!("sm{}{}", props.major, props.minor);
                Ok(DeviceInfo {
                    arch_name,
                    total_vram_bytes: props.totalGlobalMem as u64,
                    warp_size: props.warpSize as u32,
                    clock_rate_khz: props.clockRate as u32,
                })
            }
            #[cfg(not(any()))]
            {
                Err(GpuError::InvalidArg("CUDA backend not compiled".into()))
            }
        }
        Backend::Metal => {
            #[cfg(any())]
            {
                let mut arch_name = vec![0i8; 64];
                let mut total_vram_bytes = 0u64;
                let mut warp_size = 0u32;
                let mut clock_rate_khz = 0u32;
                let status = unsafe {
                    supersonic_metal_query_device_info(
                        ordinal,
                        arch_name.as_mut_ptr(),
                        arch_name.len(),
                        &mut total_vram_bytes,
                        &mut warp_size,
                        &mut clock_rate_khz,
                    )
                };
                if status != 0 {
                    return Err(backend_error(
                        Backend::Metal,
                        "metalQueryDeviceInfo",
                        status,
                    ));
                }
                let nul_pos = arch_name
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(arch_name.len());
                let arch_name = String::from_utf8_lossy(
                    &arch_name[..nul_pos]
                        .iter()
                        .map(|&c| c as u8)
                        .collect::<Vec<_>>(),
                )
                .to_string();
                Ok(DeviceInfo {
                    arch_name,
                    total_vram_bytes,
                    warp_size,
                    clock_rate_khz,
                })
            }
            #[cfg(not(any()))]
            {
                Err(GpuError::InvalidArg("Metal backend not compiled".into()))
            }
        }
    }
}

#[cfg(any())]
fn metal_runtime_compile_smoke() -> Result<()> {
    let status = unsafe { supersonic_metal_compile_shader_smoke() };
    if status != 0 {
        return Err(backend_error(
            Backend::Metal,
            "metalCompileShaderSmoke",
            status,
        ));
    }
    Ok(())
}

/// Dtype-aware element count from shape.
pub fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Byte size for a given dtype and element count.
pub fn byte_len(dtype: ScalarType, elems: usize) -> usize {
    elems * dtype.size_in_bytes()
}

#[cfg(test)]
mod hal_profile_tests {
    use std::ffi::OsString;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    const HAL_PROFILE_ENV: &str = "SUPERSONIC_HAL_PROFILE";
    static PROFILE_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct ProfileTestGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        env_value: Option<OsString>,
    }

    impl ProfileTestGuard {
        fn new() -> Self {
            let lock = PROFILE_TEST_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let env_value = std::env::var_os(HAL_PROFILE_ENV);
            std::env::remove_var(HAL_PROFILE_ENV);
            hal_profile_set_enabled(false);
            hal_profile_reset();
            Self {
                _lock: lock,
                env_value,
            }
        }
    }

    impl Drop for ProfileTestGuard {
        fn drop(&mut self) {
            hal_profile_set_enabled(false);
            hal_profile_reset();
            match &self.env_value {
                Some(value) => std::env::set_var(HAL_PROFILE_ENV, value),
                None => std::env::remove_var(HAL_PROFILE_ENV),
            }
        }
    }

    fn entry<'a>(snapshot: &'a HalProfileSnapshot, op: &str) -> &'a HalProfileEntry {
        snapshot
            .entries
            .iter()
            .find(|entry| entry.op == op)
            .unwrap_or_else(|| panic!("missing HAL profile entry {op}"))
    }

    #[test]
    fn overlapping_captures_stay_enabled_until_both_drop_in_either_order() {
        let _guard = ProfileTestGuard::new();

        let first = HalProfileCapture::begin();
        let second = HalProfileCapture::begin();
        drop(first);
        assert!(
            hal_profile_enabled(),
            "dropping the first capture disabled the second"
        );
        drop(second);
        assert!(!hal_profile_enabled());

        let first = HalProfileCapture::begin();
        let second = HalProfileCapture::begin();
        drop(second);
        assert!(
            hal_profile_enabled(),
            "dropping the second capture disabled the first"
        );
        drop(first);
        assert!(!hal_profile_enabled());
    }

    #[test]
    fn surviving_capture_receives_samples_after_first_capture_drops() {
        let _guard = ProfileTestGuard::new();
        let first = HalProfileCapture::begin();
        let survivor = HalProfileCapture::begin();

        record_hal_profile_sample("copy_h2d", 4, 0.25);
        drop(first);
        assert!(hal_profile_enabled());
        record_hal_profile_sample("copy_d2h", 8, 0.75);
        let snapshot = survivor.finish();

        assert!(!hal_profile_enabled());
        assert_eq!(snapshot.total_calls, 2);
        assert_eq!(snapshot.total_ms, 1.0);
        assert_eq!(snapshot.h2d_bytes, 4);
        assert_eq!(snapshot.d2h_bytes, 8);
        assert_eq!(entry(&snapshot, "copy_h2d").calls, 1);
        assert_eq!(entry(&snapshot, "copy_d2h").calls, 1);
    }

    #[test]
    fn finish_and_drop_each_release_exactly_one_capture_owner() {
        let _guard = ProfileTestGuard::new();
        let finished = HalProfileCapture::begin();
        let dropped = HalProfileCapture::begin();
        record_hal_profile_sample("alloc", 32, 0.5);

        let snapshot = finished.finish();
        assert_eq!(snapshot.alloc_calls, 1);
        assert!(
            hal_profile_enabled(),
            "finish and its subsequent Drop released more than one owner"
        );
        drop(dropped);
        assert!(!hal_profile_enabled());
    }

    #[test]
    fn explicit_process_enablement_outlives_capture_ownership() {
        let _guard = ProfileTestGuard::new();
        hal_profile_set_enabled(true);
        let finished = HalProfileCapture::begin();
        let dropped = HalProfileCapture::begin();

        let _ = finished.finish();
        assert!(hal_profile_enabled());
        drop(dropped);
        assert!(hal_profile_enabled());
        record_hal_profile_sample("sync", 0, 0.5);
        assert_eq!(hal_profile_snapshot().sync_calls, 1);

        hal_profile_set_enabled(false);
        assert!(!hal_profile_enabled());
    }

    #[test]
    fn active_capture_remains_effective_after_explicit_enablement_is_cleared() {
        let _guard = ProfileTestGuard::new();
        hal_profile_set_enabled(true);
        let capture = HalProfileCapture::begin();

        hal_profile_set_enabled(false);
        assert!(hal_profile_enabled());
        record_hal_profile_sample("copy_h2d", 16, 0.5);
        let snapshot = capture.finish();

        assert_eq!(snapshot.h2d_bytes, 16);
        assert!(!hal_profile_enabled());
    }

    #[test]
    fn active_capture_remains_effective_when_environment_enablement_is_removed() {
        let _guard = ProfileTestGuard::new();
        std::env::set_var(HAL_PROFILE_ENV, "1");
        let capture = HalProfileCapture::begin();

        std::env::remove_var(HAL_PROFILE_ENV);
        assert!(hal_profile_enabled());
        record_hal_profile_sample("copy_d2h", 16, 0.5);
        let snapshot = capture.finish();

        assert_eq!(snapshot.d2h_bytes, 16);
        assert!(!hal_profile_enabled());

        std::env::set_var(HAL_PROFILE_ENV, "1");
        let capture = HalProfileCapture::begin();
        let _ = capture.finish();
        assert!(hal_profile_enabled());
    }

    #[test]
    fn concurrent_capture_lifetimes_preserve_survivors_and_release_all_owners() {
        const CAPTURE_THREADS: usize = 4;

        let _guard = ProfileTestGuard::new();
        let start = Arc::new(Barrier::new(CAPTURE_THREADS + 1));
        let (capture_tx, capture_rx) = std::sync::mpsc::channel();
        let mut begin_threads = Vec::new();
        for index in 0..CAPTURE_THREADS {
            let start = Arc::clone(&start);
            let capture_tx = capture_tx.clone();
            begin_threads.push(thread::spawn(move || {
                start.wait();
                capture_tx
                    .send((index, HalProfileCapture::begin()))
                    .expect("send concurrent capture");
            }));
        }
        drop(capture_tx);
        start.wait();

        let mut captures = capture_rx.into_iter().collect::<Vec<_>>();
        for begin_thread in begin_threads {
            begin_thread.join().expect("concurrent capture begin");
        }
        captures.sort_by_key(|(index, _)| *index);
        while captures.len() > 1 {
            let (_, capture) = captures.remove(0);
            thread::spawn(move || drop(capture))
                .join()
                .expect("cross-thread capture drop");
            assert!(
                hal_profile_enabled(),
                "a concurrent capture close disabled a live peer"
            );
        }

        let (_, last_concurrent_capture) = captures.pop().expect("remaining capture");
        let survivor = HalProfileCapture::begin();
        thread::spawn(move || drop(last_concurrent_capture))
            .join()
            .expect("last concurrent capture drop");
        assert!(
            hal_profile_enabled(),
            "the last concurrent capture disabled a newer survivor"
        );
        record_hal_profile_sample("copy_d2d", 8, 0.5);
        let survivor_snapshot = survivor.finish();
        assert_eq!(survivor_snapshot.d2d_bytes, 8);
        assert!(!hal_profile_enabled());

        let ready = Arc::new(Barrier::new(CAPTURE_THREADS + 1));
        let finish = Arc::new(Barrier::new(CAPTURE_THREADS + 1));
        let mut finish_threads = Vec::new();
        for _ in 0..CAPTURE_THREADS {
            let ready = Arc::clone(&ready);
            let finish = Arc::clone(&finish);
            finish_threads.push(thread::spawn(move || {
                let capture = HalProfileCapture::begin();
                ready.wait();
                finish.wait();
                capture.finish()
            }));
        }
        ready.wait();
        assert!(hal_profile_enabled());
        record_hal_profile_sample("memset_zeros", 32, 0.25);
        finish.wait();
        for finish_thread in finish_threads {
            let snapshot = finish_thread.join().expect("concurrent capture finish");
            assert_eq!(snapshot.memset_bytes, 32);
        }
        assert!(!hal_profile_enabled());
    }

    #[test]
    fn unwinding_capture_releases_only_its_own_enablement() {
        let _guard = ProfileTestGuard::new();
        let mut survivor = None;
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _unwinding = HalProfileCapture::begin();
            survivor = Some(HalProfileCapture::begin());
            panic!("expected capture unwind");
        }));

        assert!(unwind.is_err());
        assert!(
            hal_profile_enabled(),
            "unwinding capture disabled its surviving peer"
        );
        record_hal_profile_sample("copy_h2d", 8, 0.5);
        let snapshot = survivor.take().expect("surviving capture").finish();
        assert_eq!(snapshot.h2d_bytes, 8);
        assert!(!hal_profile_enabled());
    }

    #[test]
    fn nested_capture_tracks_exact_max_while_outer_profile_continues() {
        let _guard = ProfileTestGuard::new();
        hal_profile_set_enabled(true);

        record_hal_profile_sample("copy_h2d", 64, 10.0);
        let capture = HalProfileCapture::begin();
        record_hal_profile_sample("copy_h2d", 16, 1.0);
        let nested = capture.finish();
        record_hal_profile_sample("copy_d2h", 8, 2.0);
        let outer = hal_profile_snapshot();

        let nested_h2d = entry(&nested, "copy_h2d");
        assert_eq!(nested.total_calls, 1);
        assert_eq!(nested.total_ms, 1.0);
        assert_eq!(nested.h2d_bytes, 16);
        assert_eq!(nested.d2h_bytes, 0);
        assert_eq!(nested_h2d.calls, 1);
        assert_eq!(nested_h2d.total_bytes, 16);
        assert_eq!(nested_h2d.total_ms, 1.0);
        assert_eq!(nested_h2d.max_ms, 1.0);

        let outer_h2d = entry(&outer, "copy_h2d");
        assert_eq!(outer.total_calls, 3);
        assert_eq!(outer.total_ms, 13.0);
        assert_eq!(outer.h2d_bytes, 80);
        assert_eq!(outer.d2h_bytes, 8);
        assert_eq!(outer_h2d.calls, 2);
        assert_eq!(outer_h2d.total_bytes, 80);
        assert_eq!(outer_h2d.total_ms, 11.0);
        assert_eq!(outer_h2d.max_ms, 10.0);
        assert_eq!(entry(&outer, "copy_d2h").max_ms, 2.0);
    }
}

#[cfg(all(test, target_os = "macos", any()))]
mod tests {
    use super::*;
    use crate::{set_backend, Backend, GpuBuffer, ScalarType};

    fn use_metal_backend() {
        set_backend(Backend::Metal);
    }

    #[test]
    fn metal_device_info_reports_expected_shape() {
        use_metal_backend();
        let info = query_device_info(Backend::Metal, 0).expect("query metal device info");
        assert!(
            info.arch_name.contains("apple"),
            "unexpected metal arch name: {}",
            info.arch_name
        );
        assert!(info.total_vram_bytes > 0, "missing working set budget");
        assert_eq!(info.warp_size, 32);
    }

    #[test]
    fn metal_buffer_round_trip_copy_zero_and_sync() {
        use_metal_backend();
        let ordinal = 0usize;
        let host = [1.0f32, -2.5, 3.25, 4.5];
        let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let src = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[host.len()], &host_bytes)
            .expect("upload source buffer");
        let mut dst = GpuBuffer::zeros(ordinal, ScalarType::F32, &[host.len()])
            .expect("allocate zero destination");

        copy_d2d(ordinal, dst.as_mut_ptr(), src.as_ptr(), src.len_bytes()).expect("copy_d2d");
        sync(ordinal).expect("sync after copy_d2d");
        let copied = dst.to_host_bytes().expect("download copied bytes");
        assert_eq!(copied, host_bytes);

        memset_zeros(ordinal, dst.as_mut_ptr(), dst.len_bytes()).expect("memset zeros");
        sync(ordinal).expect("sync after memset");
        let zeroed = dst.to_host_bytes().expect("download zeroed bytes");
        assert!(
            zeroed.iter().all(|&b| b == 0),
            "destination buffer not zeroed"
        );
    }

    #[test]
    fn metal_runtime_shader_compile_smoke_succeeds() {
        use_metal_backend();
        metal_runtime_compile_smoke().expect("runtime Metal shader compilation should succeed");
    }

    #[test]
    fn metal_rejects_nonzero_ordinal() {
        use_metal_backend();
        let err =
            alloc(1, 16, BufferKind::Persistent).expect_err("metal ordinal 1 should be rejected");
        assert!(
            err.to_string().contains("ordinal 0"),
            "unexpected error: {err}"
        );
    }
}
