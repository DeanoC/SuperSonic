use std::ffi::c_void;
use std::ptr::NonNull;

use crate::backend::{Backend, BufferKind};
use crate::error::{GpuError, Result};
use crate::ops::{self, AllocatorKind};
use crate::scalar_type::ScalarType;

/// Owned GPU device memory with shape and dtype metadata.
///
/// Allocation routes through `ops::alloc`, which dispatches based on the
/// active `BufferPolicy`'s mapping for the supplied `BufferKind`:
/// `AllocStrategy::Default` uses `hipMalloc` / `cudaMalloc` / metal device
/// memory (always GPU-cacheable); `AllocStrategy::HostMapped` (HIP only)
/// uses `hipHostMalloc(MAPPED)` + `hipHostGetDevicePointer` so host and
/// device share the same physical pages — saves the H2D copy, but
/// bypasses GPU L2 on RDNA3.5 APUs. Each buffer remembers which
/// allocator produced it (and, for `UnifiedHost`, the original host
/// pointer needed by `hipHostFree`) so `Drop` issues the matching free.
///
/// The `_with_kind` constructors take an explicit `BufferKind`; the
/// short-named ones default to `BufferKind::Persistent` — the right
/// choice for weights, KV cache, activations, anything re-read across
/// kernel launches. Use `BufferKind::Scratch` only for one-shot staging
/// buffers with no GPU L2 reuse.
pub struct GpuBuffer {
    ptr: NonNull<c_void>,
    len_bytes: usize,
    dtype: ScalarType,
    shape: Vec<usize>,
    device_ordinal: usize,
    backend: Backend,
    allocator: AllocatorKind,
}

// GPU pointers are not thread-safe by default, but access is serialized by the
// decode forward pass (CLI is single-threaded; the HTTP server guards every
// session with a `tokio::sync::Mutex`). Mark Send so `GpuBuffer` can cross
// `spawn_blocking` boundaries, and Sync so buffers wrapped in `Arc<_>` — for
// example shared embedding / lm_head tensors — can be carried through an
// `Arc<Mutex<…>>` shared across axum handler tasks.
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

/// Owned host memory used as the certified-KV Tier-2 store.
///
/// CUDA/HIP allocate pinned host pages so selected FP16 blocks can be paged
/// back into device scratch without first staging through pageable memory.
pub struct HostBuffer {
    ptr: NonNull<c_void>,
    len_bytes: usize,
    dtype: ScalarType,
    shape: Vec<usize>,
    device_ordinal: usize,
    backend: Backend,
}

unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        ops::free_host_pinned(
            self.backend,
            self.device_ordinal,
            self.ptr.as_ptr(),
            self.len_bytes,
        );
    }
}

impl HostBuffer {
    pub fn zeros(ordinal: usize, dtype: ScalarType, shape: &[usize]) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let len_bytes = ops::byte_len(dtype, elems);
        let ptr = ops::alloc_host_pinned(ordinal, len_bytes)?;
        unsafe { std::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, len_bytes) };
        Ok(Self {
            ptr,
            len_bytes,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
            backend: crate::current_backend(),
        })
    }

    pub fn clone_host(&self) -> Result<Self> {
        let cloned = Self::zeros(self.device_ordinal, self.dtype, &self.shape)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr() as *const u8,
                cloned.ptr.as_ptr() as *mut u8,
                self.len_bytes,
            );
        }
        Ok(cloned)
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    pub fn device_ptr(&self) -> Result<*const c_void> {
        ops::host_pinned_device_ptr(self.backend, self.device_ordinal, self.ptr.as_ptr())
            .map(|ptr| ptr.as_ptr() as *const c_void)
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn device_mut_ptr(&mut self) -> Result<*mut c_void> {
        ops::host_pinned_device_ptr(self.backend, self.device_ordinal, self.ptr.as_ptr())
            .map(|ptr| ptr.as_ptr())
    }

    pub fn offset_ptr(&self, byte_offset: usize) -> *const c_void {
        debug_assert!(byte_offset <= self.len_bytes);
        unsafe { (self.ptr.as_ptr() as *const u8).add(byte_offset) as *const c_void }
    }

    pub fn offset_mut_ptr(&mut self, byte_offset: usize) -> *mut c_void {
        debug_assert!(byte_offset <= self.len_bytes);
        unsafe { (self.ptr.as_ptr() as *mut u8).add(byte_offset) as *mut c_void }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len_bytes) }
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len_bytes) }
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    pub fn dtype(&self) -> ScalarType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        ops::free(
            self.backend,
            self.device_ordinal,
            self.ptr.as_ptr(),
            self.allocator,
        );
    }
}

impl GpuBuffer {
    /// Allocate uninitialized device memory for the given shape and dtype.
    /// Defaults to `BufferKind::Persistent` — use `alloc_with_kind` to opt
    /// into `Scratch`.
    pub fn alloc(ordinal: usize, dtype: ScalarType, shape: &[usize]) -> Result<Self> {
        Self::alloc_with_kind(ordinal, dtype, shape, BufferKind::Persistent)
    }

    /// Allocate uninitialized device memory with an explicit kind hint.
    pub fn alloc_with_kind(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        kind: BufferKind,
    ) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let len_bytes = ops::byte_len(dtype, elems);
        let (ptr, allocator) = ops::alloc(ordinal, len_bytes, kind)?;
        Ok(Self {
            ptr,
            len_bytes,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
            backend: crate::current_backend(),
            allocator,
        })
    }

    /// Allocate zero-filled device memory. Defaults to `BufferKind::Persistent`.
    pub fn zeros(ordinal: usize, dtype: ScalarType, shape: &[usize]) -> Result<Self> {
        Self::zeros_with_kind(ordinal, dtype, shape, BufferKind::Persistent)
    }

    /// Allocate zero-filled device memory with an explicit kind hint.
    pub fn zeros_with_kind(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        kind: BufferKind,
    ) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let len_bytes = ops::byte_len(dtype, elems);
        let (ptr, allocator) = ops::alloc_zeros(ordinal, len_bytes, kind)?;
        Ok(Self {
            ptr,
            len_bytes,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
            backend: crate::current_backend(),
            allocator,
        })
    }

    /// Allocate device memory and copy host data into it. Defaults to
    /// `BufferKind::Persistent` since this constructor is typically used for
    /// weight uploads.
    pub fn from_host_bytes(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        data: &[u8],
    ) -> Result<Self> {
        Self::from_host_bytes_with_kind(ordinal, dtype, shape, data, BufferKind::Persistent)
    }

    /// `from_host_bytes` with an explicit kind hint.
    pub fn from_host_bytes_with_kind(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        data: &[u8],
        kind: BufferKind,
    ) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let expected = ops::byte_len(dtype, elems);
        if data.len() != expected {
            return Err(GpuError::InvalidArg(format!(
                "from_host_bytes: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        let (ptr, allocator) = ops::alloc(ordinal, expected, kind)?;
        ops::copy_h2d(
            ordinal,
            ptr.as_ptr(),
            data.as_ptr() as *const c_void,
            expected,
        )?;
        Ok(Self {
            ptr,
            len_bytes: expected,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
            backend: crate::current_backend(),
            allocator,
        })
    }

    /// Raw pointer for read-only FFI.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    /// Raw pointer for read-write FFI.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Pointer offset by `byte_offset` bytes.
    pub fn offset_ptr(&self, byte_offset: usize) -> *const c_void {
        unsafe { (self.ptr.as_ptr() as *const u8).add(byte_offset) as *const c_void }
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
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

    pub fn elem_count(&self) -> usize {
        ops::elem_count(&self.shape)
    }

    /// Copy entire buffer contents to a new host Vec<u8>.
    pub fn to_host_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; self.len_bytes];
        ops::copy_d2h(
            self.device_ordinal,
            buf.as_mut_ptr() as *mut c_void,
            self.ptr.as_ptr() as *const c_void,
            self.len_bytes,
        )?;
        Ok(buf)
    }

    /// Create a deep copy of this buffer on the same device.
    pub fn clone_device(&self) -> Result<Self> {
        let new_buf = Self::alloc(self.device_ordinal, self.dtype, &self.shape)?;
        ops::copy_d2d(
            self.device_ordinal,
            new_buf.ptr.as_ptr(),
            self.ptr.as_ptr() as *const c_void,
            self.len_bytes,
        )?;
        Ok(new_buf)
    }

    /// Grow the buffer along `seq_dim` from current capacity to `new_cap`.
    /// Allocates a new zero-filled buffer and copies old data with correct strides.
    /// Used for KV cache pre-allocation in chunks.
    pub fn grow_seq_dim(&self, seq_dim: usize, new_cap: usize) -> Result<Self> {
        if seq_dim >= self.shape.len() {
            return Err(GpuError::InvalidArg(format!(
                "grow_seq_dim: seq_dim {seq_dim} >= rank {}",
                self.shape.len()
            )));
        }
        let old_cap = self.shape[seq_dim];
        if new_cap <= old_cap {
            return Err(GpuError::InvalidArg(format!(
                "grow_seq_dim: new_cap {new_cap} <= old_cap {old_cap}"
            )));
        }
        let mut new_shape = self.shape.clone();
        new_shape[seq_dim] = new_cap;
        let new_buf = Self::zeros(self.device_ordinal, self.dtype, &new_shape)?;
        let elem_size = self.dtype.size_in_bytes();

        // Compute the number of "outer" slices (product of dims before seq_dim)
        // and the "inner" size (product of dims after seq_dim, in bytes).
        let outer: usize = self.shape[..seq_dim].iter().product();
        let inner_elems: usize = self.shape[seq_dim + 1..].iter().product();
        let inner_bytes = inner_elems * elem_size;

        // Each outer slice has old_cap * inner_bytes in the old buffer
        // and new_cap * inner_bytes in the new buffer.
        let old_slice_bytes = old_cap * inner_bytes;
        let new_slice_bytes = new_cap * inner_bytes;
        let src_base = self.ptr.as_ptr() as *const u8;
        let dst_base = new_buf.ptr.as_ptr() as *mut u8;

        for i in 0..outer {
            let src = unsafe { src_base.add(i * old_slice_bytes) } as *const c_void;
            let dst = unsafe { dst_base.add(i * new_slice_bytes) } as *mut c_void;
            ops::copy_d2d(self.device_ordinal, dst, src, old_slice_bytes)?;
        }
        Ok(new_buf)
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuBuffer({:?}, {:?}, {} bytes, device:{})",
            self.dtype, self.shape, self.len_bytes, self.device_ordinal
        )
    }
}
