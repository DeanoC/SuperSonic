#include <metal_stdlib>
using namespace metal;

// Scaffold kernel compiled by the Metal backend build. Real Qwen3.8 kernels
// will live alongside the HIP `.hip` sources under `kernels/metal/`.
kernel void supersonic_scaffold_smoke(device bfloat* out [[buffer(0)]],
                                      uint tid [[thread_position_in_grid]]) {
    out[tid] = bfloat(1.0);
}
