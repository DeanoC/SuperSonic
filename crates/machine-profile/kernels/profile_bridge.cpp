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
extern "C" __global__ void mp_wmma_probe_kernel(int *flags_out);

// Runtime probe: writes 1 into flags_out[k] for each dtype the device's
// arch slice has a real wmma intrinsic for, 0 otherwise. Returns 0 on
// success, non-zero on hip error. flags_out_3 must point to int[3]
// (f16, bf16, i8).
extern "C" int mp_wmma_probe(int device, int *flags_out_3)
{
    hipSetDevice(device);
    int *d_flags = nullptr;
    if (hipMalloc(&d_flags, sizeof(int) * 3) != hipSuccess) {
        flags_out_3[0] = 0; flags_out_3[1] = 0; flags_out_3[2] = 0;
        return -1;
    }
    int sentinels[3] = { -1, -1, -1 };
    hipMemcpy(d_flags, sentinels, sizeof(sentinels), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(mp_wmma_probe_kernel, dim3(1), dim3(1), 0, 0, d_flags);
    if (hipDeviceSynchronize() != hipSuccess) {
        hipFree(d_flags);
        flags_out_3[0] = 0; flags_out_3[1] = 0; flags_out_3[2] = 0;
        return -2;
    }
    hipMemcpy(flags_out_3, d_flags, sizeof(int) * 3, hipMemcpyDeviceToHost);
    hipFree(d_flags);
    // If kernel didn't run (no compatible arch slice), sentinels stay -1.
    // Treat as unsupported across the board.
    for (int i = 0; i < 3; ++i) {
        if (flags_out_3[i] < 0) flags_out_3[i] = 0;
    }
    return 0;
}

extern "C" double mp_wmma_peak_f16(int device, uint32_t cu_count, uint64_t iters)
{
    hipSetDevice(device);
    int threads = 32; // wave32 on RDNA
    uint32_t blocks = cu_count * 2;
    // Fallback (non-RDNA3) kernels write sink[blockIdx.x] unconditionally, so
    // size the sink for every launched block, not just per-CU.
    float *sink = nullptr;
    hipMalloc(&sink, sizeof(float) * blocks);
    hipMemset(sink, 0, sizeof(float) * blocks);

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
    int threads = 32; // wave32 on RDNA
    uint32_t blocks = cu_count * 2;
    // Sink sized for every launched block; fallback kernels write unconditionally.
    float *sink = nullptr;
    hipMalloc(&sink, sizeof(float) * blocks);
    hipMemset(sink, 0, sizeof(float) * blocks);

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
    int threads = 32;
    uint32_t blocks = cu_count * 2;
    // Sink sized for every launched block; fallback kernels write unconditionally.
    int *sink = nullptr;
    hipMalloc(&sink, sizeof(int) * blocks);
    hipMemset(sink, 0, sizeof(int) * blocks);

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

extern "C" int mp_query_device_info(int device,
                                    char *arch_name_out, uint32_t arch_name_len,
                                    uint64_t *total_vram_bytes_out,
                                    uint32_t *warp_size_out,
                                    uint32_t *clock_rate_khz_out,
                                    uint32_t *pci_device_id_out)
{
    hipSetDevice(device);
    hipDeviceProp_t props;
    int status = hipGetDeviceProperties(&props, device);
    if (status != 0) return status;
    strncpy(arch_name_out, props.gcnArchName, arch_name_len - 1);
    arch_name_out[arch_name_len - 1] = '\0';
    *total_vram_bytes_out = (uint64_t)props.totalGlobalMem;
    *warp_size_out = (uint32_t)props.warpSize;
    *clock_rate_khz_out = (uint32_t)props.clockRate;
    // hipDeviceAttributePciChipId returns the GPU manufacturer (hardware) device ID,
    // e.g. 0x744c for RX 7900 XTX (gfx1100/Navi 31). This is distinct from
    // props.pciDeviceID which is the PCIe slot number (0-31), not the product ID.
    int chip_id = 0;
    hipDeviceGetAttribute(&chip_id, hipDeviceAttributePciChipId, device);
    *pci_device_id_out = (uint32_t)chip_id;
    return 0;
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
