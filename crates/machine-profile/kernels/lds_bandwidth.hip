#include <hip/hip_runtime.h>
#include <stdint.h>

// One block per CU; each thread issues many LDS loads/stores in a loop.
// Returns total bytes moved through LDS in `bytes_out`.
extern "C" __global__ void mp_lds_bandwidth_kernel(
    uint64_t iters,
    uint64_t *bytes_out)
{
    __shared__ float lds[1024];
    int tid = threadIdx.x;
    if (tid < 1024) lds[tid] = (float)tid;
    __syncthreads();

    float acc = 0.0f;
    for (uint64_t i = 0; i < iters; ++i) {
        // Each iteration touches 1024 floats = 4096 bytes per block.
        acc += lds[(tid + i) & 1023];
    }
    if (tid == 0 && acc < -1.0f) {
        bytes_out[blockIdx.x] = (uint64_t)acc; // sink
    } else if (tid == 0) {
        bytes_out[blockIdx.x] = iters * 4096ULL;
    }
}
