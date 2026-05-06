#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

// Vector-typed inputs/outputs for the wmma intrinsics. ext_vector_type is the
// proven pattern in this repo (see gemma4.hip, qwen36_moe helpers.cuh).
// Defined unconditionally — they're harmless on archs without wmma builtins.
typedef _Float16 mp_half16  __attribute__((ext_vector_type(16)));
typedef short    mp_short16 __attribute__((ext_vector_type(16)));
typedef float    mp_float8  __attribute__((ext_vector_type(8)));
typedef int8_t   mp_i8x16   __attribute__((ext_vector_type(16)));
typedef int32_t  mp_i32x8   __attribute__((ext_vector_type(8)));

// Detect wmma builtins per-dtype via clang's __has_builtin. Each --offload-arch=X
// slice gets its own preprocessor state, so the real-vs-fallback decision is
// taken per arch by the toolchain — no hardcoded arch list to maintain.

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
#define MP_HAS_WMMA_F16 1
#else
#define MP_HAS_WMMA_F16 0
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
#define MP_HAS_WMMA_BF16 1
#else
#define MP_HAS_WMMA_BF16 0
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
#define MP_HAS_WMMA_I8 1
#else
#define MP_HAS_WMMA_I8 0
#endif

#if MP_HAS_WMMA_F16
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
#else
extern "C" __global__ void mp_wmma_peak_f16_kernel(uint64_t iters, float *sink)
{
    if (threadIdx.x == 0) sink[blockIdx.x] = 0.0f;
}
#endif

#if MP_HAS_WMMA_BF16
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
extern "C" __global__ void mp_wmma_peak_bf16_kernel(uint64_t iters, float *sink)
{
    if (threadIdx.x == 0) sink[blockIdx.x] = 0.0f;
}
#endif

#if MP_HAS_WMMA_I8
extern "C" __global__ void mp_wmma_peak_i8_kernel(uint64_t iters, int *sink)
{
    mp_i8x16 a = {};
    mp_i8x16 b = {};
    mp_i32x8 c = {};
    for (uint64_t i = 0; i < iters; ++i) {
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    }
    if (threadIdx.x == 0 && c[0] < -2000000000) sink[blockIdx.x] = c[0];
}
#else
extern "C" __global__ void mp_wmma_peak_i8_kernel(uint64_t iters, int *sink)
{
    if (threadIdx.x == 0) sink[blockIdx.x] = 0;
}
#endif

// Probe kernel — writes the per-dtype builtin availability into flags_out[0..3].
// flags_out[0] = f16, [1] = bf16, [2] = i8. The host launcher runs this with a
// 1x1 grid; the device-arch slice the loader picks reveals the real per-dtype
// support without the host hardcoding any arch names.
extern "C" __global__ void mp_wmma_probe_kernel(int *flags_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    flags_out[0] = MP_HAS_WMMA_F16;
    flags_out[1] = MP_HAS_WMMA_BF16;
    flags_out[2] = MP_HAS_WMMA_I8;
}
