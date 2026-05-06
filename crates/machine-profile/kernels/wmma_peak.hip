#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__)

// Use ext_vector_type (proven pattern in this repo's gemma4.hip / qwen36_moe helpers.cuh).
// For f16: _Float16 x16 — matches the wmma f16 intrinsic signature.
// For bf16: short x16 — matches the wmma bf16 intrinsic signature (same bit layout).
typedef _Float16  mp_half16  __attribute__((ext_vector_type(16)));
typedef short     mp_short16 __attribute__((ext_vector_type(16)));
typedef float     mp_float8  __attribute__((ext_vector_type(8)));

extern "C" __global__ void mp_wmma_peak_f16_kernel(uint64_t iters, float *sink)
{
    mp_half16 a = {};
    mp_half16 b = {};
    mp_float8 c = {};
    for (uint64_t i = 0; i < iters; ++i) {
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    if (threadIdx.x == 0 && c[0] < -1e30f) sink[blockIdx.x] = c[0];
}

extern "C" __global__ void mp_wmma_peak_bf16_kernel(uint64_t iters, float *sink)
{
    mp_short16 a = {};
    mp_short16 b = {};
    mp_float8  c = {};
    for (uint64_t i = 0; i < iters; ++i) {
        c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    if (threadIdx.x == 0 && c[0] < -1e30f) sink[blockIdx.x] = c[0];
}

#else

extern "C" __global__ void mp_wmma_peak_f16_kernel(uint64_t iters, float *sink)
{
    if (threadIdx.x == 0) sink[blockIdx.x] = 0.0f;
}
extern "C" __global__ void mp_wmma_peak_bf16_kernel(uint64_t iters, float *sink)
{
    // TODO: bf16 wmma not supported on this arch
    if (threadIdx.x == 0) sink[blockIdx.x] = 0.0f;
}

#endif
