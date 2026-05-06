#include <hip/hip_runtime.h>
#include <stdint.h>
#include <chrono>

extern "C" __global__ void mp_lds_bandwidth_kernel(uint64_t iters, uint64_t *bytes_out);

extern "C" double mp_lds_bandwidth_run(int device, uint32_t cu_count, uint64_t iters)
{
    hipSetDevice(device);
    uint64_t *d_bytes = nullptr;
    hipMalloc(&d_bytes, sizeof(uint64_t) * cu_count);
    hipMemset(d_bytes, 0, sizeof(uint64_t) * cu_count);

    // Warmup
    hipLaunchKernelGGL(mp_lds_bandwidth_kernel, dim3(cu_count), dim3(1024), 0, 0,
                       iters / 10, d_bytes);
    hipDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(mp_lds_bandwidth_kernel, dim3(cu_count), dim3(1024), 0, 0,
                       iters, d_bytes);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    uint64_t *h_bytes = new uint64_t[cu_count];
    hipMemcpy(h_bytes, d_bytes, sizeof(uint64_t) * cu_count, hipMemcpyDeviceToHost);
    uint64_t total = 0;
    for (uint32_t i = 0; i < cu_count; ++i) total += h_bytes[i];
    delete[] h_bytes;
    hipFree(d_bytes);

    double secs = std::chrono::duration<double>(t1 - t0).count();
    return (double)total / secs / 1e9; // GB/s aggregate
}
