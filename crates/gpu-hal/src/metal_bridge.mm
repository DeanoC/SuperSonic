#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>

#include <algorithm>
#include <cstring>
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

id<MTLCommandQueue> metal_queue() {
    static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
    return queue;
}

std::mutex& dispatch_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct ActiveCommandBuffer {
    id<MTLCommandBuffer> command_buffer = nil;
    bool has_work = false;
    int encoder_count = 0;
};

ActiveCommandBuffer& active_command_buffer() {
    static ActiveCommandBuffer state;
    return state;
}

constexpr int kMaxEncodersPerCommandBuffer = 8192;

std::vector<id<MTLCommandBuffer>>& pending_command_buffers() {
    static std::vector<id<MTLCommandBuffer>> buffers;
    return buffers;
}

id<MTLCommandBuffer> ensure_command_buffer() {
    ActiveCommandBuffer& state = active_command_buffer();
    if (state.command_buffer == nil) {
        state.command_buffer = [metal_queue() commandBuffer];
        state.has_work = false;
        state.encoder_count = 0;
    }
    return state.command_buffer;
}

void commit_active_command_buffer_locked() {
    ActiveCommandBuffer& state = active_command_buffer();
    if (state.command_buffer != nil && state.has_work) {
        [state.command_buffer commit];
        pending_command_buffers().push_back(state.command_buffer);
        state.command_buffer = nil;
        state.has_work = false;
        state.encoder_count = 0;
    }
}

void wait_pending_command_buffers_locked() {
    for (id<MTLCommandBuffer> command_buffer : pending_command_buffers()) {
        [command_buffer waitUntilCompleted];
    }
    pending_command_buffers().clear();
}

void note_command_buffer_work_locked() {
    ActiveCommandBuffer& state = active_command_buffer();
    state.has_work = true;
    state.encoder_count += 1;
    if (state.encoder_count >= kMaxEncodersPerCommandBuffer) {
        commit_active_command_buffer_locked();
    }
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

extern "C" const void* supersonic_metal_dummy_buffer() {
    @autoreleasepool {
        static void* ptr = nullptr;
        if (ptr != nullptr) {
            return ptr;
        }
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return nullptr;
        }
        id<MTLBuffer> buffer = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return nullptr;
        }
        void* contents = [buffer contents];
        if (contents == nullptr) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(registry_mutex());
            registry().push_back(BufferRecord{
                reinterpret_cast<uintptr_t>(contents),
                16,
                buffer,
            });
        }
        ptr = contents;
        return ptr;
    }
}

extern "C" void supersonic_metal_dispatch_wait() {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(dispatch_mutex());
        commit_active_command_buffer_locked();
        wait_pending_command_buffers_locked();
    }
}

extern "C" int supersonic_metal_copy_d2d(void* dst, const void* src, size_t len) {
    if (dst == nullptr || src == nullptr || len == 0) {
        return 1;
    }
    void* dst_buffer = nullptr;
    void* src_buffer = nullptr;
    size_t dst_offset = 0;
    size_t src_offset = 0;
    if (supersonic_metal_lookup_buffer(dst, &dst_buffer, &dst_offset) != 0 ||
        supersonic_metal_lookup_buffer(src, &src_buffer, &src_offset) != 0) {
        std::memcpy(dst, src, len);
        return 0;
    }
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(dispatch_mutex());
        id<MTLCommandBuffer> command_buffer = ensure_command_buffer();
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        if (blit == nil) {
            return 2;
        }
        [blit copyFromBuffer:(__bridge id<MTLBuffer>)src_buffer
                  sourceOffset:src_offset
                    toBuffer:(__bridge id<MTLBuffer>)dst_buffer
           destinationOffset:dst_offset
                          size:len];
        [blit endEncoding];
        note_command_buffer_work_locked();
        return 0;
    }
}

extern "C" int supersonic_metal_submit_compute(
    void* pipeline,
    void (*configure)(id<MTLComputeCommandEncoder> encoder, bool* configured, void* ctx),
    void* ctx) {
    if (pipeline == nullptr || configure == nullptr) {
        return 1;
    }
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(dispatch_mutex());
        id<MTLComputePipelineState> pipeline_state = (__bridge id<MTLComputePipelineState>)pipeline;
        id<MTLCommandBuffer> command_buffer = ensure_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 2;
        }
        [encoder setComputePipelineState:pipeline_state];
        bool configured = true;
        configure(encoder, &configured, ctx);
        if (!configured) {
            [encoder endEncoding];
            return 3;
        }
        [encoder endEncoding];
        note_command_buffer_work_locked();
        return 0;
    }
}

extern "C" int supersonic_metal_compile_shader_smoke() {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return 1;
        }
        NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void supersonic_smoke(device bfloat* out [[buffer(0)]],\n"
             "                             uint tid [[thread_position_in_grid]]) {\n"
             "    out[tid] = bfloat(1.0);\n"
             "}\n";
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil || error != nil) {
            return 2;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"supersonic_smoke"];
        if (function == nil) {
            return 3;
        }
        return 0;
    }
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
