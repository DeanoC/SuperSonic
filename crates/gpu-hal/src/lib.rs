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
    compiled_backends, current_backend, current_memory_architecture, is_backend_compiled,
    set_backend, set_memory_architecture, Backend, DeviceInfo, MemoryArchitecture,
};
pub use buffer::{GpuBuffer, HostBuffer};
pub use error::GpuError;
pub use ops::{
    copy_d2d, copy_d2h, copy_h2d, hal_profile_reset, hal_profile_set_enabled,
    hal_profile_snapshot, memset_zeros, query_device_info, set_device, sync, GpuEvent,
    HalProfileEntry, HalProfileSnapshot,
};
pub use scalar_type::ScalarType;
