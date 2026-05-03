use std::ffi::{c_char, c_int, c_uint, c_void};

pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub(crate) const CUDA_MEM_ACCESS_FLAGS_PROT_READ_WRITE: c_uint = 3;
pub(crate) const CUDA_MEM_ALLOCATION_TYPE_PINNED: c_uint = 1;
pub(crate) const CUDA_MEM_HANDLE_TYPE_NONE: c_uint = 0;
pub(crate) const CUDA_MEM_LOCATION_TYPE_DEVICE: c_uint = 1;
pub(crate) const CUDA_MEM_ALLOCATION_GRANULARITY_RECOMMENDED: c_uint = 1;

pub(crate) type CuDevice = c_int;
pub(crate) type CuDevicePtr = u64;
pub(crate) type CuMemGenericAllocationHandle = u64;
pub(crate) type CuResult = c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CuMemLocation {
    pub type_: c_uint,
    pub id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CuMemAccessDesc {
    pub location: CuMemLocation,
    pub flags: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CuMemAllocationPropAllocFlags {
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CuMemAllocationProp {
    pub type_: c_uint,
    pub requested_handle_types: c_uint,
    pub location: CuMemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub alloc_flags: CuMemAllocationPropAllocFlags,
}

#[allow(non_snake_case)]
#[repr(C)]
pub(crate) struct CudaDeviceProp {
    pub name: [c_char; 256],
    pub uuid: [u8; 16],
    pub luid: [u8; 8],
    pub luidDeviceNodeMask: c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: c_int,
    pub sparseCudaArraySupported: c_int,
    pub hostRegisterReadOnlySupported: c_int,
    pub timelineSemaphoreInteropSupported: c_int,
    pub memoryPoolsSupported: c_int,
    pub gpuDirectRDMASupported: c_int,
    pub gpuDirectRDMAFlushWritesOptions: c_uint,
    pub gpuDirectRDMAWritesOrdering: c_int,
    pub memoryPoolSupportedHandleTypes: c_uint,
    pub deferredMappingCudaArraySupported: c_int,
    pub ipcEventSupported: c_int,
    pub clusterLaunch: c_int,
    pub unifiedFunctionPointers: c_int,
    pub reserved: [c_int; 63],
}

#[link(name = "cudart")]
unsafe extern "C" {
    pub(crate) fn cudaGetDevice(device: *mut c_int) -> c_int;
    pub(crate) fn cudaSetDevice(device: c_int) -> c_int;
    pub(crate) fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub(crate) fn cudaFree(ptr: *mut c_void) -> c_int;
    pub(crate) fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    pub(crate) fn cudaHostGetDevicePointer(
        device_ptr: *mut *mut c_void,
        host_ptr: *mut c_void,
        flags: c_uint,
    ) -> c_int;
    pub(crate) fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    pub(crate) fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
    ) -> c_int;
    pub(crate) fn cudaMemset(dst: *mut c_void, value: c_int, size: usize) -> c_int;
    pub(crate) fn cudaDeviceSynchronize() -> c_int;
    pub(crate) fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: c_int) -> c_int;
}

#[cfg(supersonic_backend_cuda)]
#[link(name = "cuda")]
unsafe extern "C" {
    pub(crate) fn cuInit(flags: c_uint) -> CuResult;
    pub(crate) fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    pub(crate) fn cuMemAddressReserve(
        ptr: *mut CuDevicePtr,
        size: usize,
        alignment: usize,
        addr: CuDevicePtr,
        flags: u64,
    ) -> CuResult;
    pub(crate) fn cuMemAddressFree(ptr: CuDevicePtr, size: usize) -> CuResult;
    pub(crate) fn cuMemCreate(
        handle: *mut CuMemGenericAllocationHandle,
        size: usize,
        prop: *const CuMemAllocationProp,
        flags: u64,
    ) -> CuResult;
    pub(crate) fn cuMemRelease(handle: CuMemGenericAllocationHandle) -> CuResult;
    pub(crate) fn cuMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const CuMemAllocationProp,
        option: c_uint,
    ) -> CuResult;
    pub(crate) fn cuMemMap(
        ptr: CuDevicePtr,
        size: usize,
        offset: usize,
        handle: CuMemGenericAllocationHandle,
        flags: u64,
    ) -> CuResult;
    pub(crate) fn cuMemUnmap(ptr: CuDevicePtr, size: usize) -> CuResult;
    pub(crate) fn cuMemSetAccess(
        ptr: CuDevicePtr,
        size: usize,
        desc: *const CuMemAccessDesc,
        count: usize,
    ) -> CuResult;
}
