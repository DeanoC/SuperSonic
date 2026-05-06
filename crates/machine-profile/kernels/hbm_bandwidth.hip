#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void mp_hbm_read_kernel(const float4 *src, uint64_t n4, float *sink)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = src[i];
    if (v.x + v.y + v.z + v.w < -1.0e30f) sink[0] = v.x;
}

extern "C" __global__ void mp_hbm_write_kernel(float4 *dst, uint64_t n4, float seed)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v;
    v.x = seed; v.y = seed; v.z = seed; v.w = seed;
    dst[i] = v;
}

extern "C" __global__ void mp_hbm_copy_kernel(const float4 *src, float4 *dst, uint64_t n4)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    dst[i] = src[i];
}
