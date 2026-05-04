#pragma once

#if defined(SUPERSONIC_QWEN36_CUDA_BRIDGE) || defined(__CUDACC__)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipGetLastError cudaGetLastError
#define hipDeviceSynchronize cudaDeviceSynchronize

__device__ __forceinline__ unsigned int supersonic_cuda_atomic_load_u32(
    const unsigned int* ptr,
    int order) {
    unsigned int value = *(const volatile unsigned int*)ptr;
    if (order != __ATOMIC_RELAXED) {
        __threadfence();
    }
    return value;
}

__device__ __forceinline__ void supersonic_cuda_atomic_store_u32(
    unsigned int* ptr,
    unsigned int value,
    int order) {
    if (order != __ATOMIC_RELAXED) {
        __threadfence();
    }
    *(volatile unsigned int*)ptr = value;
}

#define __atomic_load_n(ptr, order) \
    supersonic_cuda_atomic_load_u32((const unsigned int*)(ptr), (order))
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
