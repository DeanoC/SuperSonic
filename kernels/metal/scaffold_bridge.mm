#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace {

id<MTLDevice> metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

std::string normalized_arch_name(id<MTLDevice> device) {
    NSString* name = device ? device.name : nil;
    if (name == nil) {
        return "apple-gpu";
    }
    NSString* lowered = [[name lowercaseString] stringByReplacingOccurrencesOfString:@" " withString:@"-"];
    return std::string([lowered UTF8String]);
}

int query_device_info(
    size_t ordinal,
    char* arch_name_out,
    size_t arch_name_len,
    uint64_t* total_vram_out
) {
    if (ordinal != 0 || arch_name_out == nullptr || arch_name_len == 0 || total_vram_out == nullptr) {
        return 1;
    }
    id<MTLDevice> device = metal_device();
    if (device == nil) {
        return 2;
    }
    const std::string arch_name = normalized_arch_name(device);
    const size_t max_copy_len = arch_name_len > 0 ? arch_name_len - 1 : 0;
    const size_t copy_len = std::min(max_copy_len, arch_name.size());
    memcpy(arch_name_out, arch_name.data(), copy_len);
    arch_name_out[copy_len] = '\0';

    uint64_t total_vram = 0;
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        total_vram = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
    }
    if (total_vram == 0) {
        total_vram = static_cast<uint64_t>(NSProcessInfo.processInfo.physicalMemory);
    }
    *total_vram_out = total_vram;
    return 0;
}

}  // namespace

extern "C" int supersonic_metal_scaffold_compile_smoke() {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return 1;
        }
        NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void supersonic_scaffold_smoke(device bfloat* out [[buffer(0)]],\n"
             "                                      uint tid [[thread_position_in_grid]]) {\n"
             "    out[tid] = bfloat(1.0);\n"
             "}\n";
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil || error != nil) {
            return 2;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"supersonic_scaffold_smoke"];
        if (function == nil) {
            return 3;
        }
        return 0;
    }
}

extern "C" int supersonic_query_gpu_info(
    int ordinal,
    unsigned char* arch_name_out,
    size_t arch_name_len,
    uint64_t* total_vram_out
) {
  return query_device_info(
      static_cast<size_t>(ordinal),
      reinterpret_cast<char*>(arch_name_out),
      arch_name_len,
      total_vram_out);
}

extern "C" int supersonic_hip_device_clock_khz(int ordinal, uint32_t* clock_khz_out) {
    if (ordinal != 0 || clock_khz_out == nullptr) {
        return 1;
    }
    *clock_khz_out = 0;
    return 0;
}
