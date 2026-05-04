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

#define __atomic_load_n(ptr, order) (*(volatile unsigned int*)(ptr))
#define __atomic_store_n(ptr, val, order) (*(volatile unsigned int*)(ptr) = (val))

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
