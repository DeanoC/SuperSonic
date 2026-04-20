#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct BufferRecord {
    uintptr_t start;
    size_t len;
    __strong id<MTLBuffer> buffer;
};

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<BufferRecord>& registry() {
    static std::vector<BufferRecord> records;
    return records;
}

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

}  // namespace

extern "C" int supersonic_metal_alloc(size_t len_bytes, void** ptr_out) {
    @autoreleasepool {
        if (ptr_out == nullptr || len_bytes == 0) {
            return 1;
        }
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return 2;
        }
        id<MTLBuffer> buffer = [device newBufferWithLength:len_bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return 3;
        }
        void* ptr = [buffer contents];
        if (ptr == nullptr) {
            return 4;
        }
        {
            std::lock_guard<std::mutex> lock(registry_mutex());
            registry().push_back(BufferRecord{
                reinterpret_cast<uintptr_t>(ptr),
                len_bytes,
                buffer,
            });
        }
        *ptr_out = ptr;
        return 0;
    }
}

extern "C" int supersonic_metal_free(void* ptr) {
    @autoreleasepool {
        if (ptr == nullptr) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(registry_mutex());
        auto& records = registry();
        const uintptr_t target = reinterpret_cast<uintptr_t>(ptr);
        auto it = std::find_if(records.begin(), records.end(), [target](const BufferRecord& record) {
            return record.start == target;
        });
        if (it == records.end()) {
            return 1;
        }
        records.erase(it);
        return 0;
    }
}

extern "C" int supersonic_metal_lookup_buffer(
    const void* ptr,
    void** buffer_out,
    size_t* offset_out
) {
    if (ptr == nullptr || buffer_out == nullptr || offset_out == nullptr) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(registry_mutex());
    const uintptr_t target = reinterpret_cast<uintptr_t>(ptr);
    for (const auto& record : registry()) {
        const uintptr_t end = record.start + record.len;
        if (target >= record.start && target < end) {
            *buffer_out = (__bridge void*)record.buffer;
            *offset_out = static_cast<size_t>(target - record.start);
            return 0;
        }
    }
    return 2;
}

extern "C" int supersonic_metal_query_device_info(
    size_t ordinal,
    char* arch_name_out,
    size_t arch_name_len,
    uint64_t* total_vram_out,
    uint32_t* warp_size_out,
    uint32_t* clock_rate_khz_out
) {
    @autoreleasepool {
        if (ordinal != 0 || arch_name_out == nullptr || arch_name_len == 0 || total_vram_out == nullptr ||
            warp_size_out == nullptr || clock_rate_khz_out == nullptr) {
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
        *warp_size_out = 32;
        *clock_rate_khz_out = 0;
        return 0;
    }
}
