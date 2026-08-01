#pragma once

#if defined(SUPERSONIC_QWEN36_CUDA_BRIDGE) || defined(__CUDACC__)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <math.h>
#include <stdint.h>
#include <string.h>

using hip_bfloat16 = __nv_bfloat16;

#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) \
    kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)

#define hipSuccess cudaSuccess
#define hipError_t cudaError_t
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDevice cudaGetDevice
#define hipSetDevice cudaSetDevice
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipGetLastError cudaGetLastError
#define hipDeviceSynchronize cudaDeviceSynchronize

__device__ __forceinline__ unsigned int supersonic_cuda_atomic_load_u32(
    unsigned int* ptr,
    int order) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> atom(*ptr);
    switch (order) {
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_SEQ_CST:
            return atom.load(cuda::memory_order_acquire);
        default:
            return atom.load(cuda::memory_order_relaxed);
    }
}

__device__ __forceinline__ void supersonic_cuda_atomic_store_u32(
    unsigned int* ptr,
    unsigned int value,
    int order) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> atom(*ptr);
    switch (order) {
        case __ATOMIC_RELEASE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_SEQ_CST:
            atom.store(value, cuda::memory_order_release);
            break;
        default:
            atom.store(value, cuda::memory_order_relaxed);
            break;
    }
}

#define __atomic_load_n(ptr, order) \
    supersonic_cuda_atomic_load_u32((unsigned int*)(ptr), (order))
#define __atomic_store_n(ptr, val, order) \
    supersonic_cuda_atomic_store_u32((unsigned int*)(ptr), (unsigned int)(val), (order))

#ifndef __HIP_PLATFORM_AMD__
#define __shfl(val, lane) __shfl_sync(0xffffffffu, val, lane)
#define __shfl_down(val, delta) __shfl_down_sync(0xffffffffu, val, delta)
#define __shfl_xor(val, lane_mask) __shfl_xor_sync(0xffffffffu, val, lane_mask)
#endif
#else
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#endif
