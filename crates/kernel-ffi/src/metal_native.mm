#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>
#include <stdint.h>

extern "C" int supersonic_metal_lookup_buffer(
    const void* ptr,
    void** buffer_out,
    size_t* offset_out
);

namespace {

struct MatmulParams {
    uint32_t batch_elems;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct FullAttentionParams {
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t head_dim;
    uint32_t seqlen_offset;
    float scale;
};

struct RmsNormParams {
    uint32_t n_rows;
    uint32_t n_cols;
    float eps;
    uint32_t add_unit_offset;
};

struct LinearConvParams {
    uint32_t conv_dim;
    uint32_t total_len;
    uint32_t seq_len;
    uint32_t kernel_size;
};

id<MTLDevice> metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> metal_queue() {
    static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
    return queue;
}

void configure_precise_math(MTLCompileOptions* options) {
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
}

id<MTLComputePipelineState> matmul_pipeline_bf16(NSError** error_out) {
    static std::mutex mutex;
    static bool attempted = false;
    static __strong id<MTLComputePipelineState> pipeline = nil;
    static __strong NSError* build_error = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:1
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"MATMUL(
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint batch_elems;
    uint m;
    uint n;
    uint k;
};

kernel void supersonic_matmul_rhs_transposed_bf16(
    device const bfloat* lhs [[buffer(0)]],
    device const bfloat* rhs [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (batch >= params.batch_elems || row >= params.m || col >= params.n) {
        return;
    }
    float acc = 0.0f;
    uint lhs_base = (batch * params.m + row) * params.k;
    uint rhs_base = col * params.k;
    for (uint kk = 0; kk < params.k; ++kk) {
        acc += float(lhs[lhs_base + kk]) * float(rhs[rhs_base + kk]);
    }
    out[(batch * params.m + row) * params.n + col] = bfloat(acc);
}
)MATMUL";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:2
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile matmul library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:3
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load matmul function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:4
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create matmul pipeline"
                                                                             }];
                        }
                    }
                }
            }
        }
    }

    if (pipeline == nil && error_out != nullptr) {
        *error_out = build_error;
    }
    return pipeline;
}

id<MTLComputePipelineState> full_attention_pipeline_bf16_f32(NSError** error_out) {
    static std::mutex mutex;
    static bool attempted = false;
    static __strong id<MTLComputePipelineState> pipeline = nil;
    static __strong NSError* build_error = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:11
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"FATTN(
#include <metal_stdlib>
using namespace metal;

struct FullAttentionParams {
    uint q_heads;
    uint kv_heads;
    uint q_len;
    uint kv_len;
    uint head_dim;
    uint seqlen_offset;
    float scale;
};

kernel void supersonic_full_attention_prefill_bf16_f32(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* key [[buffer(1)]],
    device const bfloat* value [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant FullAttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint q_pos = gid.y;
    uint q_head = gid.z;
    if (q_head >= params.q_heads || q_pos >= params.q_len || d >= params.head_dim) {
        return;
    }

    uint num_kv_groups = params.q_heads / params.kv_heads;
    uint kv_head = q_head / num_kv_groups;
    uint max_attend = min(params.seqlen_offset + q_pos + 1, params.kv_len);
    uint query_base = (q_head * params.q_len + q_pos) * params.head_dim;

    float max_score = -INFINITY;
    for (uint kv_pos = 0; kv_pos < max_attend; ++kv_pos) {
        uint key_base = (kv_head * params.kv_len + kv_pos) * params.head_dim;
        float dot = 0.0f;
        for (uint kk = 0; kk < params.head_dim; ++kk) {
            dot += float(query[query_base + kk]) * float(key[key_base + kk]);
        }
        float score = dot * params.scale;
        max_score = max(max_score, score);
    }

    float denom = 0.0f;
    float numer = 0.0f;
    for (uint kv_pos = 0; kv_pos < max_attend; ++kv_pos) {
        uint key_base = (kv_head * params.kv_len + kv_pos) * params.head_dim;
        float dot = 0.0f;
        for (uint kk = 0; kk < params.head_dim; ++kk) {
            dot += float(query[query_base + kk]) * float(key[key_base + kk]);
        }
        float weight = exp((dot * params.scale) - max_score);
        denom += weight;
        uint value_base = (kv_head * params.kv_len + kv_pos) * params.head_dim;
        numer += weight * float(value[value_base + d]);
    }

    out[(q_head * params.q_len + q_pos) * params.head_dim + d] = numer / denom;
}
)FATTN";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:12
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile full-attention library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_full_attention_prefill_bf16_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:13
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load full-attention function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:14
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create full-attention pipeline"
                                                                             }];
                        }
                    }
                }
            }
        }
    }

    if (pipeline == nil && error_out != nullptr) {
        *error_out = build_error;
    }
    return pipeline;
}

id<MTLComputePipelineState> rms_norm_pipeline_bf16(NSError** error_out) {
    static std::mutex mutex;
    static bool attempted = false;
    static __strong id<MTLComputePipelineState> pipeline = nil;
    static __strong NSError* build_error = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:31
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"RMS(
#include <metal_stdlib>
using namespace metal;

struct RmsNormParams {
    uint n_rows;
    uint n_cols;
    float eps;
    uint add_unit_offset;
};

kernel void supersonic_rms_norm_rows_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= params.n_rows || col >= params.n_cols) {
        return;
    }

    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint kk = 0; kk < params.n_cols; ++kk) {
        float value = float(input[row_base + kk]);
        mean_sq += value * value;
    }
    float inv_rms = 1.0f / sqrt((mean_sq / float(params.n_cols)) + params.eps);
    float scale = float(weight[col]) + (params.add_unit_offset != 0 ? 1.0f : 0.0f);
    out[row_base + col] = bfloat(float(input[row_base + col]) * inv_rms * scale);
}
)RMS";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:32
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile RMSNorm library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_rms_norm_rows_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:33
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load RMSNorm function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:34
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create RMSNorm pipeline"
                                                                             }];
                        }
                    }
                }
            }
        }
    }

    if (pipeline == nil && error_out != nullptr) {
        *error_out = build_error;
    }
    return pipeline;
}

id<MTLComputePipelineState> linear_prefill_conv_pack_pipeline_bf16(NSError** error_out) {
    static std::mutex mutex;
    static bool attempted = false;
    static __strong id<MTLComputePipelineState> pipeline = nil;
    static __strong NSError* build_error = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:51
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"LCONV(
#include <metal_stdlib>
using namespace metal;

struct LinearConvParams {
    uint conv_dim;
    uint total_len;
    uint seq_len;
    uint kernel_size;
};

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void supersonic_linear_prefill_conv_pack_bf16(
    device const bfloat* mixed [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant LinearConvParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint ch = gid.x;
    uint t = gid.y;
    if (ch >= params.conv_dim || t >= params.seq_len) {
        return;
    }

    uint mixed_base = ch * params.total_len + t;
    uint weight_base = ch * params.kernel_size;
    float acc = 0.0f;
    for (uint kk = 0; kk < params.kernel_size; ++kk) {
        acc += float(mixed[mixed_base + kk]) * float(weights[weight_base + kk]);
    }
    out[t * params.conv_dim + ch] = bfloat(silu(acc));
}
)LCONV";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:52
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile linear conv library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_linear_prefill_conv_pack_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:53
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load linear conv function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:54
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create linear conv pipeline"
                                                                             }];
                        }
                    }
                }
            }
        }
    }

    if (pipeline == nil && error_out != nullptr) {
        *error_out = build_error;
    }
    return pipeline;
}

int lookup_buffer(
    const void* ptr,
    id<MTLBuffer>* buffer_out,
    size_t* offset_out
) {
    void* raw_buffer = nullptr;
    size_t offset = 0;
    int status = supersonic_metal_lookup_buffer(ptr, &raw_buffer, &offset);
    if (status != 0) {
        return status;
    }
    if (buffer_out != nullptr) {
        *buffer_out = (__bridge id<MTLBuffer>)raw_buffer;
    }
    if (offset_out != nullptr) {
        *offset_out = offset;
    }
    return 0;
}

}  // namespace

extern "C" int supersonic_metal_matmul_rhs_transposed_bf16(
    size_t batch_elems,
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_ptr,
    const void* rhs_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (batch_elems == 0 || m == 0 || n == 0 || k == 0 || lhs_ptr == nullptr || rhs_ptr == nullptr ||
            out_ptr == nullptr) {
            return 1;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 2;
        }

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 3;
        }
        if (lookup_buffer(rhs_ptr, &rhs, &rhs_offset) != 0) {
            return 4;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 5;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 6;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 7;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 8;
        }

        MatmulParams params = {
            static_cast<uint32_t>(batch_elems),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
        [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
        [encoder setBuffer:out offset:out_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width = std::min<NSUInteger>(8, std::max<NSUInteger>(1, n));
        NSUInteger tg_height =
            std::min<NSUInteger>(8, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tg_width));
        if (tg_height == 0) {
            tg_height = 1;
        }
        MTLSize threads_per_group = MTLSizeMake(tg_width, tg_height, 1);
        MTLSize threads_per_grid = MTLSizeMake(n, m, batch_elems);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 9;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_full_attention_prefill_bf16_f32(
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    float scale,
    size_t seqlen_offset,
    const void* query_ptr,
    const void* key_ptr,
    const void* value_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 || head_dim == 0 ||
            query_ptr == nullptr || key_ptr == nullptr || value_ptr == nullptr || out_ptr == nullptr) {
            return 21;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = full_attention_pipeline_bf16_f32(&pipeline_error);
        if (pipeline == nil) {
            return 22;
        }

        id<MTLBuffer> query = nil;
        id<MTLBuffer> key = nil;
        id<MTLBuffer> value = nil;
        id<MTLBuffer> out = nil;
        size_t query_offset = 0;
        size_t key_offset = 0;
        size_t value_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(query_ptr, &query, &query_offset) != 0) {
            return 23;
        }
        if (lookup_buffer(key_ptr, &key, &key_offset) != 0) {
            return 24;
        }
        if (lookup_buffer(value_ptr, &value, &value_offset) != 0) {
            return 25;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 26;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 27;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 28;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 29;
        }

        FullAttentionParams params = {
            static_cast<uint32_t>(q_heads),
            static_cast<uint32_t>(kv_heads),
            static_cast<uint32_t>(q_len),
            static_cast<uint32_t>(kv_len),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(seqlen_offset),
            scale,
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:query offset:query_offset atIndex:0];
        [encoder setBuffer:key offset:key_offset atIndex:1];
        [encoder setBuffer:value offset:value_offset atIndex:2];
        [encoder setBuffer:out offset:out_offset atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

        NSUInteger tg_width = std::min<NSUInteger>(16, std::max<NSUInteger>(1, head_dim));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(head_dim, q_len, q_heads);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 30;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_rms_norm_rows_bf16(
    size_t n_rows,
    size_t n_cols,
    float eps,
    bool add_unit_offset,
    const void* input_ptr,
    const void* weight_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (n_rows == 0 || n_cols == 0 || input_ptr == nullptr || weight_ptr == nullptr || out_ptr == nullptr) {
            return 41;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = rms_norm_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 42;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> out = nil;
        size_t input_offset = 0;
        size_t weight_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 43;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 44;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 45;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 46;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 47;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 48;
        }

        RmsNormParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            add_unit_offset ? 1u : 0u,
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:input_offset atIndex:0];
        [encoder setBuffer:weight offset:weight_offset atIndex:1];
        [encoder setBuffer:out offset:out_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width = std::min<NSUInteger>(32, std::max<NSUInteger>(1, n_cols));
        NSUInteger tg_height =
            std::min<NSUInteger>(8, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tg_width));
        if (tg_height == 0) {
            tg_height = 1;
        }
        MTLSize threads_per_group = MTLSizeMake(tg_width, tg_height, 1);
        MTLSize threads_per_grid = MTLSizeMake(n_cols, n_rows, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 49;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_linear_prefill_conv_pack_bf16(
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_ptr,
    const void* weights_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (conv_dim == 0 || total_len == 0 || seq_len == 0 || kernel_size == 0 || mixed_ptr == nullptr ||
            weights_ptr == nullptr || out_ptr == nullptr) {
            return 61;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = linear_prefill_conv_pack_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 62;
        }

        id<MTLBuffer> mixed = nil;
        id<MTLBuffer> weights = nil;
        id<MTLBuffer> out = nil;
        size_t mixed_offset = 0;
        size_t weights_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(mixed_ptr, &mixed, &mixed_offset) != 0) {
            return 63;
        }
        if (lookup_buffer(weights_ptr, &weights, &weights_offset) != 0) {
            return 64;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 65;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 66;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 67;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 68;
        }

        LinearConvParams params = {
            static_cast<uint32_t>(conv_dim),
            static_cast<uint32_t>(total_len),
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(kernel_size),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:mixed offset:mixed_offset atIndex:0];
        [encoder setBuffer:weights offset:weights_offset atIndex:1];
        [encoder setBuffer:out offset:out_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width = std::min<NSUInteger>(32, std::max<NSUInteger>(1, conv_dim));
        NSUInteger tg_height =
            std::min<NSUInteger>(8, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tg_width));
        if (tg_height == 0) {
            tg_height = 1;
        }
        MTLSize threads_per_group = MTLSizeMake(tg_width, tg_height, 1);
        MTLSize threads_per_grid = MTLSizeMake(conv_dim, seq_len, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 69;
        }
        return 0;
    }
}
