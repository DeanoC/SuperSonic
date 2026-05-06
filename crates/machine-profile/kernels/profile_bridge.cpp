#include <hip/hip_runtime.h>
#include <stdint.h>
#include <chrono>
#include <cstring>

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

extern "C" __global__ void mp_hbm_read_kernel(const float4 *src, uint64_t n4, float *sink);
extern "C" __global__ void mp_hbm_write_kernel(float4 *dst, uint64_t n4, float seed);
extern "C" __global__ void mp_hbm_copy_kernel(const float4 *src, float4 *dst, uint64_t n4);

extern "C" double mp_hbm_bandwidth_read(int device, uint64_t bytes)
{
    hipSetDevice(device);
    uint64_t n4 = bytes / 16;
    float4 *src = nullptr; float *sink = nullptr;
    hipMalloc(&src, n4 * sizeof(float4));
    hipMalloc(&sink, sizeof(float));
    hipMemset(src, 0, n4 * sizeof(float4));
    int threads = 256;
    uint64_t blocks = (n4 + threads - 1) / threads;

    for (int w = 0; w < 3; ++w) {
        hipLaunchKernelGGL(mp_hbm_read_kernel, dim3(blocks), dim3(threads), 0, 0, src, n4, sink);
    }
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 10;
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(mp_hbm_read_kernel, dim3(blocks), dim3(threads), 0, 0, src, n4, sink);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    hipFree(src); hipFree(sink);
    return (double)(bytes * reps) / secs / 1e9;
}

extern "C" double mp_hbm_bandwidth_write(int device, uint64_t bytes)
{
    hipSetDevice(device);
    uint64_t n4 = bytes / 16;
    float4 *dst = nullptr;
    hipMalloc(&dst, n4 * sizeof(float4));
    int threads = 256;
    uint64_t blocks = (n4 + threads - 1) / threads;

    for (int w = 0; w < 3; ++w)
        hipLaunchKernelGGL(mp_hbm_write_kernel, dim3(blocks), dim3(threads), 0, 0, dst, n4, 1.0f);
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 10;
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(mp_hbm_write_kernel, dim3(blocks), dim3(threads), 0, 0, dst, n4, 1.0f);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    hipFree(dst);
    return (double)(bytes * reps) / secs / 1e9;
}

extern "C" double mp_hbm_bandwidth_copy(int device, uint64_t bytes)
{
    hipSetDevice(device);
    uint64_t n4 = bytes / 16;
    float4 *src = nullptr, *dst = nullptr;
    hipMalloc(&src, n4 * sizeof(float4));
    hipMalloc(&dst, n4 * sizeof(float4));
    hipMemset(src, 0, n4 * sizeof(float4));
    int threads = 256;
    uint64_t blocks = (n4 + threads - 1) / threads;

    for (int w = 0; w < 3; ++w)
        hipLaunchKernelGGL(mp_hbm_copy_kernel, dim3(blocks), dim3(threads), 0, 0, src, dst, n4);
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 10;
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(mp_hbm_copy_kernel, dim3(blocks), dim3(threads), 0, 0, src, dst, n4);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    hipFree(src); hipFree(dst);
    return (double)(2 * bytes * reps) / secs / 1e9;
}

extern "C" __global__ void mp_wmma_peak_f16_kernel(uint64_t iters, float *sink);
extern "C" __global__ void mp_wmma_peak_bf16_kernel(uint64_t iters, float *sink);
extern "C" __global__ void mp_wmma_peak_i8_kernel(uint64_t iters, int *sink);

extern "C" double mp_wmma_peak_f16(int device, uint32_t cu_count, uint64_t iters)
{
    hipSetDevice(device);
    float *sink = nullptr;
    hipMalloc(&sink, sizeof(float) * cu_count);
    hipMemset(sink, 0, sizeof(float) * cu_count);
    int threads = 32; // wave32 on RDNA
    uint32_t blocks = cu_count * 2;

    // warmup
    hipLaunchKernelGGL(mp_wmma_peak_f16_kernel, dim3(blocks), dim3(threads), 0, 0,
                       iters / 4, sink);
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 5;
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(mp_wmma_peak_f16_kernel, dim3(blocks), dim3(threads), 0, 0,
                           iters, sink);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    hipFree(sink);
    double secs = std::chrono::duration<double>(t1 - t0).count();
    // 16x16x16 wmma → 16*16*16*2 = 8192 flops per wave per iter, wave size 32
    // → 256 flops per thread per iter
    double total_flops = 256.0
        * (double)iters
        * (double)threads
        * (double)blocks
        * (double)reps;
    return total_flops / secs / 1e12; // TFLOPS
}

extern "C" double mp_wmma_peak_bf16(int device, uint32_t cu_count, uint64_t iters)
{
    hipSetDevice(device);
    float *sink = nullptr;
    hipMalloc(&sink, sizeof(float) * cu_count);
    hipMemset(sink, 0, sizeof(float) * cu_count);
    int threads = 32; // wave32 on RDNA
    uint32_t blocks = cu_count * 2;

    // warmup
    hipLaunchKernelGGL(mp_wmma_peak_bf16_kernel, dim3(blocks), dim3(threads), 0, 0,
                       iters / 4, sink);
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 5;
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(mp_wmma_peak_bf16_kernel, dim3(blocks), dim3(threads), 0, 0,
                           iters, sink);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    hipFree(sink);
    double secs = std::chrono::duration<double>(t1 - t0).count();
    // 16x16x16 wmma → 256 flops per thread per iter
    double total_flops = 256.0
        * (double)iters
        * (double)threads
        * (double)blocks
        * (double)reps;
    return total_flops / secs / 1e12; // TFLOPS
}

extern "C" double mp_wmma_peak_i8(int device, uint32_t cu_count, uint64_t iters)
{
    hipSetDevice(device);
    int *sink = nullptr;
    hipMalloc(&sink, sizeof(int) * cu_count);
    hipMemset(sink, 0, sizeof(int) * cu_count);
    int threads = 32;
    uint32_t blocks = cu_count * 2;

    hipLaunchKernelGGL(mp_wmma_peak_i8_kernel, dim3(blocks), dim3(threads), 0, 0, iters / 4, sink);
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 5;
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(mp_wmma_peak_i8_kernel, dim3(blocks), dim3(threads), 0, 0, iters, sink);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    hipFree(sink);
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double total_ops = 256.0 * (double)iters * (double)threads * (double)blocks * (double)reps;
    return total_ops / secs / 1e12;
}

struct MpTransferSample { uint64_t bytes; double gb_s; };

extern "C" int mp_pcie_h2d(int device, MpTransferSample *out, int max_samples)
{
    hipSetDevice(device);
    const uint64_t sizes[] = { 4096ULL, 65536ULL, 1ULL<<20, 1ULL<<22, 1ULL<<24, 1ULL<<26, 1ULL<<28 };
    int n = sizeof(sizes) / sizeof(sizes[0]);
    if (n > max_samples) n = max_samples;

    void *d_buf = nullptr;
    hipMalloc(&d_buf, sizes[n - 1]);
    void *h_buf = nullptr;
    hipHostMalloc(&h_buf, sizes[n - 1]);
    memset(h_buf, 0, sizes[n - 1]);

    for (int i = 0; i < n; ++i) {
        // warmup
        hipMemcpy(d_buf, h_buf, sizes[i], hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        int reps = sizes[i] >= (1ULL<<24) ? 4 : 32;
        for (int r = 0; r < reps; ++r)
            hipMemcpy(d_buf, h_buf, sizes[i], hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        out[i].bytes = sizes[i];
        out[i].gb_s = (double)(sizes[i] * reps) / secs / 1e9;
    }
    hipHostFree(h_buf);
    hipFree(d_buf);
    return n;
}

extern "C" int mp_pcie_d2h(int device, MpTransferSample *out, int max_samples)
{
    hipSetDevice(device);
    const uint64_t sizes[] = { 4096ULL, 65536ULL, 1ULL<<20, 1ULL<<22, 1ULL<<24, 1ULL<<26, 1ULL<<28 };
    int n = sizeof(sizes) / sizeof(sizes[0]);
    if (n > max_samples) n = max_samples;

    void *d_buf = nullptr;
    hipMalloc(&d_buf, sizes[n - 1]);
    hipMemset(d_buf, 0, sizes[n - 1]);
    void *h_buf = nullptr;
    hipHostMalloc(&h_buf, sizes[n - 1]);

    for (int i = 0; i < n; ++i) {
        hipMemcpy(h_buf, d_buf, sizes[i], hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        int reps = sizes[i] >= (1ULL<<24) ? 4 : 32;
        for (int r = 0; r < reps; ++r)
            hipMemcpy(h_buf, d_buf, sizes[i], hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        out[i].bytes = sizes[i];
        out[i].gb_s = (double)(sizes[i] * reps) / secs / 1e9;
    }
    hipHostFree(h_buf);
    hipFree(d_buf);
    return n;
}

extern "C" double mp_pcie_duplex(int device, uint64_t bytes)
{
    hipSetDevice(device);
    void *d_buf = nullptr; void *h_buf = nullptr;
    hipMalloc(&d_buf, bytes);
    hipHostMalloc(&h_buf, bytes);
    hipStream_t s_h2d, s_d2h;
    hipStreamCreate(&s_h2d);
    hipStreamCreate(&s_d2h);
    hipMemcpyAsync(d_buf, h_buf, bytes, hipMemcpyHostToDevice, s_h2d);
    hipMemcpyAsync(h_buf, d_buf, bytes, hipMemcpyDeviceToHost, s_d2h);
    hipStreamSynchronize(s_h2d); hipStreamSynchronize(s_d2h);
    auto t0 = std::chrono::high_resolution_clock::now();
    int reps = 4;
    for (int r = 0; r < reps; ++r) {
        hipMemcpyAsync(d_buf, h_buf, bytes, hipMemcpyHostToDevice, s_h2d);
        hipMemcpyAsync(h_buf, d_buf, bytes, hipMemcpyDeviceToHost, s_d2h);
    }
    hipStreamSynchronize(s_h2d); hipStreamSynchronize(s_d2h);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    hipStreamDestroy(s_h2d); hipStreamDestroy(s_d2h);
    hipFree(d_buf); hipHostFree(h_buf);
    return (double)(2 * bytes * reps) / secs / 1e9;
}
