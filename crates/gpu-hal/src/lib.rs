mod backend;
mod buffer;
#[cfg(supersonic_backend_cuda)]
mod cuda_sys;
mod error;
#[cfg(supersonic_backend_hip)]
mod hip_sys;
#[cfg(supersonic_backend_metal)]
mod metal_sys;
mod ops;
mod scalar_type;

pub use backend::{
    compiled_backends, current_backend, is_backend_compiled, set_backend, Backend, DeviceInfo,
};
pub use buffer::{GpuBuffer, HostBuffer};
pub use error::GpuError;
pub use ops::{
    alloc, alloc_zeros, copy_d2d, copy_d2h, copy_h2d, memset_zeros, query_device_info, set_device,
    sync, GpuEvent,
};
pub use scalar_type::ScalarType;
