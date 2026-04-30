// gfx1150 GPU L2 bypass microbenchmark — diagnoses the regression from the
// closed PR #54 (`feat/unified-memory-allocator`).
//
// Background: routing all GpuBuffer allocations through `hipHostMalloc(MAPPED)
// + hipHostGetDevicePointer` on gfx1150 (RDNA3.5 APU, AMD Radeon 890M) made
// SuperSonic decode 2.2× slower on Qwen3.5-0.8B INT4 and 1.5× slower on
// Gemma4-E2B INT4 — the opposite of the predicted "zero-copy on unified
// memory" win.
//
// rocprofv3 PMC counters are not supported on gfx1150 (ROCm 7.0.3 reports
// "Agent HW architecture is not supported, no counter metrics found"), so
// L2 hit/miss is unobservable. This standalone bench triangulates the
// mechanism via timing alone:
//
//   1. streaming reads of 256 MiB-1 GiB buffers (no cache reuse possible)
//      — both allocators land at ~94 GiB/s, ruling out a raw-bandwidth
//      regression.
//   2. repeated reads of buffers that fit in L2 / MALL — `hipMalloc`
//      sustains 82-96 GiB/s, but `hipHostMalloc(MAPPED)` collapses to
//      32-56 GiB/s. That is the GPU L2 *not engaging* for host-mapped
//      memory.
//   3. scattered single-element reads across many small buffers —
//      `hipMalloc` ~3 µs, host-mapped ~9 µs, again consistent with cache
//      bypass dominating short access patterns.
//   4. tiny-grid kernel launches over a 1 MiB buffer — host-mapped 2.6×
//      slower per launch, again because no L2 hits land between launches.
//
// Conclusion: AMD's APU coherence model treats `hipHostMalloc(MAPPED)`
// memory as coarse-grained CPU-coherent, so the GPU bypasses L2 to keep
// its view consistent with possible CPU writes. There's no "GPU
// cacheable but host mapped" combination in `hipHostMalloc` /
// `hipExtMallocWithFlags` on this stack — the coherence guarantee is
// the API contract.
//
// The Apple-silicon "unified memory" mental model does not transfer:
// Apple's GPU caches host-allocated memory; AMD's RDNA3.5 APU GPU does
// not. RDNA4 laptops may behave differently again — re-run this bench
// before drawing platform conclusions.
//
// Build: hipcc --offload-arch=gfx1150 -O3 membench_l2_bypass.hip -o membench
// Run:   ./membench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef HIP_HOST_MALLOC_MAPPED
#define HIP_HOST_MALLOC_MAPPED 0x2u
#endif

#define HIPCHK(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { \
      fprintf(stderr, "HIP error %d at %s:%d: %s\n", _e, __FILE__, __LINE__, hipGetErrorString(_e)); \
      std::exit(1); \
    } \
  } while (0)

__global__ void streaming_read_kernel(const uint4* __restrict__ in,
                                      uint4* __restrict__ sink,
                                      size_t n_uint4) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = tid; i < n_uint4; i += stride) {
    uint4 v = in[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (tid < 1024) sink[tid] = acc;
}

__global__ void scatter_read_kernel(const uint4* const* __restrict__ ptrs,
                                    int n_buffers,
                                    uint4* __restrict__ sink) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_buffers) return;
  uint4 v = ptrs[tid][0];
  sink[tid % 1024] = v;
}

void* alloc_one(const std::string& mode, size_t bytes, void** host_out) {
  *host_out = nullptr;
  if (mode == "discrete") {
    void* p = nullptr;
    HIPCHK(hipMalloc(&p, bytes));
    return p;
  }
  void* host_ptr = nullptr;
  HIPCHK(hipHostMalloc(&host_ptr, bytes, HIP_HOST_MALLOC_MAPPED));
  void* dev_ptr = nullptr;
  HIPCHK(hipHostGetDevicePointer(&dev_ptr, host_ptr, 0));
  *host_out = host_ptr;
  return dev_ptr;
}

void free_one(const std::string& mode, void* dev_ptr, void* host_ptr) {
  if (mode == "discrete") HIPCHK(hipFree(dev_ptr));
  else                    HIPCHK(hipHostFree(host_ptr));
}

double bench_streaming(const std::string& mode, size_t bytes, int iters, int warmup) {
  void* host = nullptr;
  void* dev = alloc_one(mode, bytes, &host);
  void* sink = nullptr; HIPCHK(hipMalloc(&sink, 1024 * sizeof(uint4)));
  size_t n_uint4 = bytes / sizeof(uint4);

  if (mode == "unified") std::memset(host, 0xab, bytes);
  else { HIPCHK(hipMemset(dev, 0xab, bytes)); HIPCHK(hipDeviceSynchronize()); }

  hipDeviceProp_t props{}; HIPCHK(hipGetDeviceProperties(&props, 0));
  dim3 block(256), grid((unsigned)props.multiProcessorCount * 4);

  for (int i = 0; i < warmup; ++i) streaming_read_kernel<<<grid, block>>>((const uint4*)dev, (uint4*)sink, n_uint4);
  HIPCHK(hipDeviceSynchronize());

  hipEvent_t s, e; HIPCHK(hipEventCreate(&s)); HIPCHK(hipEventCreate(&e));
  HIPCHK(hipEventRecord(s));
  for (int i = 0; i < iters; ++i) streaming_read_kernel<<<grid, block>>>((const uint4*)dev, (uint4*)sink, n_uint4);
  HIPCHK(hipEventRecord(e));
  HIPCHK(hipEventSynchronize(e));
  float ms = 0; HIPCHK(hipEventElapsedTime(&ms, s, e));
  HIPCHK(hipEventDestroy(s)); HIPCHK(hipEventDestroy(e));

  double per = ms / iters;
  double rate = (double)bytes / (per / 1000.0) / (1024.0 * 1024.0 * 1024.0);
  fprintf(stdout, "  %-9s size=%6zuMiB iters=%d per=%.4fms rate=%.2f GiB/s\n",
          mode.c_str(), bytes / (1024 * 1024), iters, per, rate);

  HIPCHK(hipFree(sink));
  free_one(mode, dev, host);
  return rate;
}

void bench_scatter(const std::string& mode, int n_buffers, size_t per_buf_bytes, int iters) {
  std::vector<void*> hosts(n_buffers, nullptr), devs(n_buffers, nullptr);
  for (int i = 0; i < n_buffers; ++i) {
    devs[i] = alloc_one(mode, per_buf_bytes, &hosts[i]);
    if (mode == "unified") std::memset(hosts[i], 0xab + (i & 15), per_buf_bytes);
    else { HIPCHK(hipMemset(devs[i], 0xab + (i & 15), per_buf_bytes)); }
  }
  HIPCHK(hipDeviceSynchronize());

  void** ptrs_dev = nullptr;
  HIPCHK(hipMalloc(&ptrs_dev, n_buffers * sizeof(void*)));
  HIPCHK(hipMemcpy(ptrs_dev, devs.data(), n_buffers * sizeof(void*), hipMemcpyHostToDevice));
  void* sink = nullptr; HIPCHK(hipMalloc(&sink, 1024 * sizeof(uint4)));

  dim3 block(256), grid((n_buffers + 255) / 256);
  for (int i = 0; i < 5; ++i) scatter_read_kernel<<<grid, block>>>((const uint4* const*)ptrs_dev, n_buffers, (uint4*)sink);
  HIPCHK(hipDeviceSynchronize());

  hipEvent_t s, e; HIPCHK(hipEventCreate(&s)); HIPCHK(hipEventCreate(&e));
  HIPCHK(hipEventRecord(s));
  for (int i = 0; i < iters; ++i) scatter_read_kernel<<<grid, block>>>((const uint4* const*)ptrs_dev, n_buffers, (uint4*)sink);
  HIPCHK(hipEventRecord(e));
  HIPCHK(hipEventSynchronize(e));
  float ms = 0; HIPCHK(hipEventElapsedTime(&ms, s, e));
  HIPCHK(hipEventDestroy(s)); HIPCHK(hipEventDestroy(e));
  fprintf(stdout, "  %-9s n=%d per_buf=%zuKiB iters=%d per=%.4fms (%.3fus/buf)\n",
          mode.c_str(), n_buffers, per_buf_bytes / 1024, iters, ms / iters, ms * 1000.0 / iters / n_buffers);

  HIPCHK(hipFree(sink)); HIPCHK(hipFree(ptrs_dev));
  for (int i = 0; i < n_buffers; ++i) free_one(mode, devs[i], hosts[i]);
}

int main(int /*argc*/, char** /*argv*/) {
  printf("=== streaming reads (no cache reuse possible) — baseline ===\n");
  bench_streaming("discrete", 256ull * 1024 * 1024, 5, 2);
  bench_streaming("unified",  256ull * 1024 * 1024, 5, 2);
  bench_streaming("discrete", 1024ull * 1024 * 1024, 5, 2);
  bench_streaming("unified",  1024ull * 1024 * 1024, 5, 2);

  printf("\n=== cache-fitting repeated reads — exposes GPU L2 bypass ===\n");
  bench_streaming("discrete", 4ull * 1024 * 1024, 200, 3);
  bench_streaming("unified",  4ull * 1024 * 1024, 200, 3);
  bench_streaming("discrete", 256ull * 1024,    1000, 3);
  bench_streaming("unified",  256ull * 1024,    1000, 3);

  printf("\n=== scattered single-element reads across many buffers ===\n");
  bench_scatter("discrete", 256,  1ull * 1024 * 1024, 100);
  bench_scatter("unified",  256,  1ull * 1024 * 1024, 100);
  bench_scatter("discrete", 1024, 256ull * 1024,      100);
  bench_scatter("unified",  1024, 256ull * 1024,      100);
  return 0;
}
