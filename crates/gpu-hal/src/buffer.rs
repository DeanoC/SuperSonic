use std::ffi::c_void;
use std::ptr::NonNull;

use crate::error::{GpuError, Result};
use crate::ops;
use crate::scalar_type::ScalarType;

/// Owned GPU device memory with shape and dtype metadata.
/// Frees on drop via `hipFree`.
pub struct GpuBuffer {
    ptr: NonNull<c_void>,
    len_bytes: usize,
    dtype: ScalarType,
    shape: Vec<usize>,
    device_ordinal: usize,
}

// GPU pointers are not thread-safe by default, but access is serialized by the
// decode forward pass. Mark Send so GpuBuffer can be held across await points.
unsafe impl Send for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        ops::free(self.device_ordinal, self.ptr.as_ptr());
    }
}

impl GpuBuffer {
    /// Allocate uninitialized device memory for the given shape and dtype.
    pub fn alloc(ordinal: usize, dtype: ScalarType, shape: &[usize]) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let len_bytes = ops::byte_len(dtype, elems);
        let ptr = ops::alloc(ordinal, len_bytes)?;
        Ok(Self {
            ptr,
            len_bytes,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
        })
    }

    /// Allocate zero-filled device memory.
    pub fn zeros(ordinal: usize, dtype: ScalarType, shape: &[usize]) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let len_bytes = ops::byte_len(dtype, elems);
        let ptr = ops::alloc_zeros(ordinal, len_bytes)?;
        Ok(Self {
            ptr,
            len_bytes,
            dtype,
            shape: shape.to_vec(),
            device_ordinal: ordinal,
        })
    }

    /// Allocate device memory and copy host data into it.
    pub fn from_host_bytes(
        ordinal: usize,
        dtype: ScalarType,
        shape: &[usize],
        data: &[u8],
    ) -> Result<Self> {
        let elems = ops::elem_count(shape);
        let expected = ops::byte_len(dtype, elems);
        if data.len() != expected {
            return Err(GpuError::InvalidArg(format!(
                "from_host_bytes: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        let ptr = ops::alloc(ordinal, expected)?;
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
