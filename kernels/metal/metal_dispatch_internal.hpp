#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#import <Metal/Metal.h>

namespace supersonic::metal::dispatch {

bool init_library();
bool bind_buffer(id<MTLComputeCommandEncoder> encoder, NSUInteger index, const void* ptr);

using EncoderConfigure = void (*)(id<MTLComputeCommandEncoder> encoder, bool* configured, void* ctx);

bool run_pipeline(const char* pipeline_name, EncoderConfigure configure, void* ctx);

bool dispatch_1d(
    const char* pipeline_name,
    std::uint32_t total_elems,
    std::uint32_t threads,
    const void* buffers[],
    std::uint32_t buffer_count,
    const void* params,
    std::size_t params_size,
    std::uint32_t params_index);

bool dispatch_rows(
    const char* pipeline_name,
    std::uint32_t rows,
    std::uint32_t threads,
    const void* buffers[],
    std::uint32_t buffer_count,
    const void* params,
    std::size_t params_size,
    std::uint32_t params_index);

bool dispatch_grid(
    const char* pipeline_name,
    std::uint32_t grid_x,
    std::uint32_t grid_y,
    std::uint32_t grid_z,
    std::uint32_t threads,
    const void* buffers[],
    std::uint32_t buffer_count,
    const void* params,
    std::size_t params_size,
    std::uint32_t params_index);

}  // namespace supersonic::metal::dispatch

#endif
