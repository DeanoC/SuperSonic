#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

extern "C" int supersonic_metal_lookup_buffer(
    const void* ptr,
    void** buffer_out,
    size_t* offset_out);

extern "C" const void* supersonic_metal_dummy_buffer();

id<MTLDevice> metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> metal_queue() {
    static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
    return queue;
}

std::mutex& library_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct LibraryHolder {
    id<MTLLibrary> value = nil;
};

LibraryHolder& prefill_library_holder() {
    static LibraryHolder holder;
    return holder;
}

std::unordered_map<std::string, id<MTLComputePipelineState>>& pipeline_cache() {
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    return cache;
}

bool bind_host_buffer(
    id<MTLComputeCommandEncoder> encoder,
    NSUInteger index,
    const void* ptr) {
    void* buffer = nullptr;
    size_t offset = 0;
    if (ptr == nullptr) {
        return false;
    }
    if (supersonic_metal_lookup_buffer(ptr, &buffer, &offset) != 0) {
        return false;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffer offset:offset atIndex:index];
    return true;
}

bool load_prefill_library() {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(library_mutex());
        if (prefill_library_holder().value != nil) {
            return true;
        }
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return false;
        }
        NSError* error = nil;
#ifdef SUPERSONIC_METAL_METALLIB_DIR
        NSString* path =
            [@(SUPERSONIC_METAL_METALLIB_DIR) stringByAppendingPathComponent:@"prefill.metallib"];
        NSURL* url = [NSURL fileURLWithPath:path];
        prefill_library_holder().value = [device newLibraryWithURL:url error:&error];
#endif
        return prefill_library_holder().value != nil;
    }
}

id<MTLComputePipelineState> pipeline_for(const char* name) {
    if (!load_prefill_library()) {
        return nil;
    }
    std::lock_guard<std::mutex> lock(library_mutex());
    const std::string key(name);
    auto found = pipeline_cache().find(key);
    if (found != pipeline_cache().end()) {
        return found->second;
    }
    id<MTLFunction> function =
        [prefill_library_holder().value newFunctionWithName:@(name)];
    if (function == nil) {
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [metal_device() newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        return nil;
    }
    pipeline_cache()[key] = pipeline;
    return pipeline;
}

template <typename Configure>
bool run_command_encoder(const char* pipeline_name, Configure configure) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = pipeline_for(pipeline_name);
        if (pipeline == nil) {
            return false;
        }
        id<MTLCommandBuffer> command_buffer = [metal_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return false;
        }
        [encoder setComputePipelineState:pipeline];
        bool configured = true;
        configure(encoder, &configured);
        if (!configured) {
            [encoder endEncoding];
            return false;
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        return command_buffer.error == nil;
    }
}

}  // namespace

id<MTLBuffer> full_attention_row_counter() {
    static id<MTLBuffer> counter = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        counter = [metal_device() newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
        if (counter != nil) {
            *static_cast<uint32_t*>(counter.contents) = 0u;
        }
    });
    return counter;
}

namespace supersonic::metal {

bool init_prefill_library() {
    return load_prefill_library();
}

bool embedding_lookup_u32(
    int dtype,
    int token_count,
    int vocab_size,
    int hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out) {
    struct Params {
        int token_count;
        int vocab_size;
        int hidden_size;
        int dtype;
    } params{token_count, vocab_size, hidden_size, dtype};

    const int total_elems = token_count * hidden_size;
    const NSUInteger threads = 256;
    const NSUInteger grid =
        static_cast<NSUInteger>((total_elems + int(threads) - 1) / int(threads));

    return run_command_encoder(
        "supersonic_metal_embedding_lookup_u32",
        [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
            if (!bind_host_buffer(encoder, 0, embeddings) ||
                !bind_host_buffer(encoder, 1, indexes) || !bind_host_buffer(encoder, 2, out)) {
                *configured = false;
                return;
            }
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(grid * threads, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        });
}

bool rms_norm(
    int dtype,
    int n_rows,
    int n_cols,
    float eps,
    int add_unit_offset,
    const void* xs,
    const void* weight,
    void* out) {
    struct Params {
        int n_rows;
        int n_cols;
        float eps;
        int add_unit_offset;
        int dtype;
    } params{n_rows, n_cols, eps, add_unit_offset, dtype};

    return run_command_encoder("supersonic_metal_rms_norm", [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
        if (!bind_host_buffer(encoder, 0, xs) || !bind_host_buffer(encoder, 1, weight) ||
            !bind_host_buffer(encoder, 2, out)) {
            *configured = false;
            return;
        }
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
}

bool matmul_rhs_transposed_tiled(
    int dtype,
    std::uint32_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs,
    void* out) {
    struct Params {
        std::uint32_t batch_elems;
        int m;
        int n;
        int k;
        int dtype;
    } params{batch_elems, m, n, k, dtype};

    const int grid_x = (n + 15) / 16;
    const int grid_y = (m + 15) / 16;

    return run_command_encoder(
        "supersonic_metal_matmul_rhs_transposed_tiled",
        [&](id<MTLComputeCommandEncoder> encoder, bool* configured) {
            if (!bind_host_buffer(encoder, 0, lhs) || !bind_host_buffer(encoder, 1, rhs) ||
                !bind_host_buffer(encoder, 2, out)) {
                *configured = false;
                return;
            }
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch_elems)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
}

#include "metal_dispatch_ops.inc"

}  // namespace supersonic::metal
