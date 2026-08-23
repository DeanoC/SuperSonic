mod backend;
mod buffer;
mod error;
mod hip_sys;
mod ops;
mod scalar_type;
mod vmm;

pub use backend::{
    current_backend, current_buffer_policy, current_memory_architecture, current_strategy_for,
    set_buffer_policy, set_memory_architecture, AllocStrategy, Backend, BufferKind, BufferPolicy,
    DeviceInfo, MemoryArchitecture,
};
pub use buffer::{GpuBuffer, HostBuffer};
pub use error::{BackendApi, GpuError};
pub use ops::{
    copy_d2d, copy_d2h, copy_h2d, copy_h2d_async, copy_storage_to_device, hal_profile_enabled,
    hal_profile_reset, hal_profile_set_enabled, hal_profile_snapshot, memset_zeros,
    memset_zeros_async, query_device_info, set_device, storage_to_device_is_supported, sync,
    GpuEvent, GpuStream, HalProfileCapture, HalProfileEntry, HalProfileSnapshot, PinnedHostBuffer,
    RegisteredHostBuffer,
};
pub use scalar_type::ScalarType;
pub use vmm::{
    vmm_is_supported, VirtualAllocation, VirtualAllocationRole, VirtualAllocationStats,
    VirtualArena, VirtualArenaStats, VirtualBacking, VirtualBuffer, VirtualBufferStats,
};
