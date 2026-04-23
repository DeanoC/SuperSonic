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

struct RmsNormGatedParams {
    uint32_t n_rows;
    uint32_t n_cols;
    float eps;
};

struct LinearConvParams {
    uint32_t conv_dim;
    uint32_t total_len;
    uint32_t seq_len;
    uint32_t kernel_size;
};

struct ElementwiseParams {
    uint32_t total_elems;
};

struct MulScalarParams {
    uint32_t total_elems;
    float scalar;
};

struct TransposeShdHsdParams {
    uint32_t s;
    uint32_t h;
    uint32_t d;
    uint32_t total_elems;
};

struct SplitQkvParams {
    uint32_t s;
    uint32_t key_dim;
    uint32_t val_dim;
    uint32_t src_stride;
    uint32_t total_elems;
};

struct SplitQgateParams {
    uint32_t s;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t src_stride;
    uint32_t total_elems;
};

struct RepeatInterleaveHeadsParams {
    uint32_t s;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t repeats;
    uint32_t dst_heads;
    uint32_t total_elems;
};

struct ComputeBetaGParams {
    uint32_t seq_len;
    uint32_t nv;
    uint32_t total_elems;
};

struct DeltaRecurrentPrefillParams {
    uint32_t seq_len;
    uint32_t k_head_dim;
    uint32_t v_head_dim;
    uint32_t out_rows;
    uint32_t total_threads;
};

struct L2NormParams {
    uint32_t n_rows;
    uint32_t n_cols;
    float eps;
    uint32_t total_elems;
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

id<MTLComputePipelineState> rms_norm_gated_pipeline_bf16(NSError** error_out) {
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
                                                   code:224
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"RMSG(
#include <metal_stdlib>
using namespace metal;

struct RmsNormGatedParams {
    uint n_rows;
    uint n_cols;
    float eps;
};

kernel void supersonic_rms_norm_gated_bf16(
    device const bfloat* hidden [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant RmsNormGatedParams& params [[buffer(4)]],
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
        float value = float(hidden[row_base + kk]);
        mean_sq += value * value;
    }
    float inv_rms = 1.0f / sqrt((mean_sq / float(params.n_cols)) + params.eps);
    float gate_value = float(gate[row_base + col]);
    float sig = 1.0f / (1.0f + exp(-gate_value));
    float silu_gate = gate_value * sig;
    float value = float(hidden[row_base + col]) * inv_rms * float(weight[col]) * silu_gate;
    out[row_base + col] = bfloat(value);
}
)RMSG";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:225
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile gated RMSNorm library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_rms_norm_gated_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:226
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load gated RMSNorm function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:227
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create gated RMSNorm pipeline"
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

id<MTLComputePipelineState> element_add_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_element_add_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:71
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"EADD(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
    uint total_elems;
};

kernel void supersonic_element_add_bf16(
    device const bfloat* lhs [[buffer(0)]],
    device const bfloat* rhs [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = bfloat(float(lhs[gid]) + float(rhs[gid]));
}

kernel void supersonic_element_add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = lhs[gid] + rhs[gid];
}
)EADD";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:72
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile element-add library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:73
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load element-add function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:74
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create element-add pipeline"
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

id<MTLComputePipelineState> sigmoid_mul_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_sigmoid_mul_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:191
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"SIGM(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
    uint total_elems;
};

kernel void supersonic_sigmoid_mul_bf16(
    device const bfloat* data [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    float gv = float(gate[gid]);
    float sig = 1.0f / (1.0f + exp(-gv));
    out[gid] = bfloat(float(data[gid]) * sig);
}

kernel void supersonic_sigmoid_mul_f32(
    device const float* data [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    float sig = 1.0f / (1.0f + exp(-gate[gid]));
    out[gid] = data[gid] * sig;
}
)SIGM";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:192
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile sigmoid-mul library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:193
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load sigmoid-mul function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:194
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create sigmoid-mul pipeline"
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

id<MTLComputePipelineState> swiglu_mul_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_swiglu_mul_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:211
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"SWIGLU(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
    uint total_elems;
};

kernel void supersonic_swiglu_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    float gv = float(gate[gid]);
    float sig = 1.0f / (1.0f + exp(-gv));
    out[gid] = bfloat(gv * sig * float(up[gid]));
}

kernel void supersonic_swiglu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    float gv = gate[gid];
    float sig = 1.0f / (1.0f + exp(-gv));
    out[gid] = gv * sig * up[gid];
}
)SWIGLU";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:212
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile swiglu-mul library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:213
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load swiglu-mul function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:214
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create swiglu-mul pipeline"
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

id<MTLComputePipelineState> cast_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableSet* attempted = nil;
    static __strong NSMutableDictionary* pipelines = nil;
    static __strong NSMutableDictionary* build_errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (attempted == nil) {
        attempted = [[NSMutableSet alloc] init];
        pipelines = [[NSMutableDictionary alloc] init];
        build_errors = [[NSMutableDictionary alloc] init];
    }

    id<MTLComputePipelineState> cached = [pipelines objectForKey:function_name];
    if (cached != nil) {
        return cached;
    }

    if (![attempted containsObject:function_name]) {
        [attempted addObject:function_name];
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                NSError* error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                     code:81
                                                 userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
                [build_errors setObject:error forKey:function_name];
            } else {
                static const char* kSource = R"CAST(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
    uint total_elems;
};

kernel void supersonic_cast_bf16_to_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = input[gid];
}

kernel void supersonic_cast_f32_to_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = input[gid];
}

kernel void supersonic_cast_u32_to_u32(
    device const uint* input [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = input[gid];
}

kernel void supersonic_cast_bf16_to_f32(
    device const bfloat* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = float(input[gid]);
}

kernel void supersonic_cast_f32_to_bf16(
    device const float* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = bfloat(input[gid]);
}
)CAST";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    NSError* error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                          code:82
                                                                      userInfo:@{
                                                                          NSLocalizedDescriptionKey :
                                                                              @"Failed to compile cast library"
                                                                      }];
                    [build_errors setObject:error forKey:function_name];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        NSError* error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                             code:83
                                                         userInfo:@{
                                                             NSLocalizedDescriptionKey :
                                                                 @"Failed to load cast function"
                                                         }];
                        [build_errors setObject:error forKey:function_name];
                    } else {
                        NSError* pipeline_error = nil;
                        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                                      error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            NSError* error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                   code:84
                                                                               userInfo:@{
                                                                                   NSLocalizedDescriptionKey :
                                                                                       @"Failed to create cast pipeline"
                                                                               }];
                            [build_errors setObject:error forKey:function_name];
                        } else {
                            [pipelines setObject:pipeline forKey:function_name];
                            [build_errors removeObjectForKey:function_name];
                            return pipeline;
                        }
                    }
                }
            }
        }
    }

    if (error_out != nullptr) {
        *error_out = [build_errors objectForKey:function_name];
    }
    return nil;
}

id<MTLComputePipelineState> mul_scalar_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_mul_scalar_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:91
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"MSCL(
#include <metal_stdlib>
using namespace metal;

struct MulScalarParams {
    uint total_elems;
    float scalar;
};

kernel void supersonic_mul_scalar_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant MulScalarParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = bfloat(float(input[gid]) * params.scalar);
}

kernel void supersonic_mul_scalar_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant MulScalarParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    out[gid] = input[gid] * params.scalar;
}
)MSCL";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:92
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile mul-scalar library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:93
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load mul-scalar function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:94
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create mul-scalar pipeline"
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

id<MTLComputePipelineState> transpose_shd_hsd_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_transpose_shd_hsd_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:101
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"TSHD(
#include <metal_stdlib>
using namespace metal;

struct TransposeShdHsdParams {
    uint s;
    uint h;
    uint d;
    uint total_elems;
};

kernel void supersonic_transpose_shd_hsd_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant TransposeShdHsdParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint elem = gid % params.d;
    uint head = (gid / params.d) % params.h;
    uint seq = gid / (params.d * params.h);
    uint dst_idx = (head * params.s + seq) * params.d + elem;
    dst[dst_idx] = src[gid];
}

kernel void supersonic_transpose_shd_hsd_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant TransposeShdHsdParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint elem = gid % params.d;
    uint head = (gid / params.d) % params.h;
    uint seq = gid / (params.d * params.h);
    uint dst_idx = (head * params.s + seq) * params.d + elem;
    dst[dst_idx] = src[gid];
}
)TSHD";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:102
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile transpose-shd-hsd library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:103
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load transpose-shd-hsd function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:104
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create transpose-shd-hsd pipeline"
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

id<MTLComputePipelineState> split_qkv_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_split_qkv_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:111
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"SQKV(
#include <metal_stdlib>
using namespace metal;

struct SplitQkvParams {
    uint s;
    uint key_dim;
    uint val_dim;
    uint src_stride;
    uint total_elems;
};

kernel void supersonic_split_qkv_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* q [[buffer(1)]],
    device bfloat* k [[buffer(2)]],
    device bfloat* v [[buffer(3)]],
    constant SplitQkvParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint row = gid / params.src_stride;
    uint col = gid - row * params.src_stride;
    if (col < params.key_dim) {
        q[row * params.key_dim + col] = src[gid];
    } else if (col < params.key_dim * 2) {
        k[row * params.key_dim + col - params.key_dim] = src[gid];
    } else {
        v[row * params.val_dim + col - params.key_dim * 2] = src[gid];
    }
}

kernel void supersonic_split_qkv_f32(
    device const float* src [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* k [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant SplitQkvParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint row = gid / params.src_stride;
    uint col = gid - row * params.src_stride;
    if (col < params.key_dim) {
        q[row * params.key_dim + col] = src[gid];
    } else if (col < params.key_dim * 2) {
        k[row * params.key_dim + col - params.key_dim] = src[gid];
    } else {
        v[row * params.val_dim + col - params.key_dim * 2] = src[gid];
    }
}
)SQKV";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:112
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile split-qkv library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:113
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load split-qkv function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:114
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create split-qkv pipeline"
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

id<MTLComputePipelineState> split_qgate_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_split_qgate_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:123
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"SQGT(
#include <metal_stdlib>
using namespace metal;

struct SplitQgateParams {
    uint s;
    uint num_heads;
    uint head_dim;
    uint src_stride;
    uint total_elems;
};

kernel void supersonic_split_qgate_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* query [[buffer(1)]],
    device bfloat* gate [[buffer(2)]],
    constant SplitQgateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint elem = gid % params.head_dim;
    uint head = (gid / params.head_dim) % params.num_heads;
    uint row = gid / (params.head_dim * params.num_heads);
    uint src_idx = row * params.src_stride + head * params.head_dim * 2 + elem;
    query[gid] = src[src_idx];
    gate[gid] = src[src_idx + params.head_dim];
}

kernel void supersonic_split_qgate_f32(
    device const float* src [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* gate [[buffer(2)]],
    constant SplitQgateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint elem = gid % params.head_dim;
    uint head = (gid / params.head_dim) % params.num_heads;
    uint row = gid / (params.head_dim * params.num_heads);
    uint src_idx = row * params.src_stride + head * params.head_dim * 2 + elem;
    query[gid] = src[src_idx];
    gate[gid] = src[src_idx + params.head_dim];
}
)SQGT";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:124
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile split-qgate library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:125
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load split-qgate function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:126
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create split-qgate pipeline"
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

id<MTLComputePipelineState> repeat_interleave_heads_pipeline(
    NSString* function_name,
    NSError** error_out
) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_repeat_interleave_heads_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:134
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"RPTI(
#include <metal_stdlib>
using namespace metal;

struct RepeatInterleaveHeadsParams {
    uint s;
    uint n_heads;
    uint head_dim;
    uint repeats;
    uint dst_heads;
    uint total_elems;
};

kernel void supersonic_repeat_interleave_heads_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant RepeatInterleaveHeadsParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint elem = gid % params.head_dim;
    uint dst_head = (gid / params.head_dim) % params.dst_heads;
    uint row = gid / (params.dst_heads * params.head_dim);
    uint src_head = dst_head / params.repeats;
    uint src_idx = ((row * params.n_heads) + src_head) * params.head_dim + elem;
    dst[gid] = src[src_idx];
}

kernel void supersonic_repeat_interleave_heads_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant RepeatInterleaveHeadsParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint elem = gid % params.head_dim;
    uint dst_head = (gid / params.head_dim) % params.dst_heads;
    uint row = gid / (params.dst_heads * params.head_dim);
    uint src_head = dst_head / params.repeats;
    uint src_idx = ((row * params.n_heads) + src_head) * params.head_dim + elem;
    dst[gid] = src[src_idx];
}
)RPTI";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:135
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile repeat-interleave-heads library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:136
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load repeat-interleave-heads function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:137
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create repeat-interleave-heads pipeline"
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

id<MTLComputePipelineState> compute_beta_g_pipeline(NSError** error_out) {
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
                                                   code:144
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"CBG(
#include <metal_stdlib>
using namespace metal;

struct ComputeBetaGParams {
    uint seq_len;
    uint nv;
    uint total_elems;
};

kernel void supersonic_compute_beta_g_f32(
    device const float* b [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* dt_bias [[buffer(2)]],
    device const float* a_log_exp [[buffer(3)]],
    device float* beta [[buffer(4)]],
    device float* g [[buffer(5)]],
    constant ComputeBetaGParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint t = gid / params.nv;
    uint h = gid - t * params.nv;
    uint dst_idx = h * params.seq_len + t;
    float bv = b[gid];
    float av = a[gid] + dt_bias[h];
    beta[dst_idx] = 1.0f / (1.0f + exp(-bv));
    float sp = (av > 20.0f) ? av : log(1.0f + exp(av));
    g[dst_idx] = -sp * a_log_exp[h];
}
)CBG";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:145
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile compute-beta-g library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:@"supersonic_compute_beta_g_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:146
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load compute-beta-g function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:147
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create compute-beta-g pipeline"
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

id<MTLComputePipelineState> delta_recurrent_prefill_pipeline(NSError** error_out) {
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
                                                   code:157
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"DRP(
#include <metal_stdlib>
using namespace metal;

struct DeltaRecurrentPrefillParams {
    uint seq_len;
    uint k_head_dim;
    uint v_head_dim;
    uint out_rows;
    uint total_threads;
};

kernel void supersonic_delta_recurrent_prefill_f32(
    device const float* initial_state [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device const float* value [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device const float* g [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant DeltaRecurrentPrefillParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }
    uint head = gid / params.v_head_dim;
    uint vv = gid - head * params.v_head_dim;

    uint state_head_base = head * params.k_head_dim * params.v_head_dim;
    uint qk_head_base = head * params.seq_len * params.k_head_dim;
    uint v_head_base = head * params.seq_len * params.v_head_dim;
    uint bg_head_base = head * params.seq_len;
    uint out_head_base = head * params.out_rows * params.v_head_dim;

    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        uint src_idx = state_head_base + kk * params.v_head_dim + vv;
        uint dst_idx = out_head_base + (params.seq_len + kk) * params.v_head_dim + vv;
        out[dst_idx] = initial_state[src_idx];
    }

    for (uint t = 0; t < params.seq_len; ++t) {
        float decay = exp(g[bg_head_base + t]);
        for (uint kk = 0; kk < params.k_head_dim; ++kk) {
            uint state_idx = out_head_base + (params.seq_len + kk) * params.v_head_dim + vv;
            out[state_idx] *= decay;
        }

        uint qk_t_base = qk_head_base + t * params.k_head_dim;
        uint v_t_base = v_head_base + t * params.v_head_dim;
        float kv_mem = 0.0f;
        for (uint kk = 0; kk < params.k_head_dim; ++kk) {
            uint state_idx = out_head_base + (params.seq_len + kk) * params.v_head_dim + vv;
            kv_mem = fma(out[state_idx], key[qk_t_base + kk], kv_mem);
        }

        float delta = (value[v_t_base + vv] - kv_mem) * beta[bg_head_base + t];
        for (uint kk = 0; kk < params.k_head_dim; ++kk) {
            uint state_idx = out_head_base + (params.seq_len + kk) * params.v_head_dim + vv;
            out[state_idx] = fma(key[qk_t_base + kk], delta, out[state_idx]);
        }

        float acc = 0.0f;
        for (uint kk = 0; kk < params.k_head_dim; ++kk) {
            uint state_idx = out_head_base + (params.seq_len + kk) * params.v_head_dim + vv;
            acc = fma(out[state_idx], query[qk_t_base + kk], acc);
        }
        out[out_head_base + t * params.v_head_dim + vv] = acc;
    }
}
)DRP";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:158
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile delta-recurrent-prefill library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_delta_recurrent_prefill_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:159
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load delta-recurrent-prefill function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:160
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create delta-recurrent-prefill pipeline"
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

id<MTLComputePipelineState> l2norm_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static bool attempted_bf16 = false;
    static bool attempted_f32 = false;
    static __strong id<MTLComputePipelineState> pipeline_bf16 = nil;
    static __strong id<MTLComputePipelineState> pipeline_f32 = nil;
    static __strong NSError* build_error_bf16 = nil;
    static __strong NSError* build_error_f32 = nil;

    const bool want_bf16 = [function_name isEqualToString:@"supersonic_l2norm_bf16"];
    bool& attempted = want_bf16 ? attempted_bf16 : attempted_f32;
    __strong id<MTLComputePipelineState>& pipeline = want_bf16 ? pipeline_bf16 : pipeline_f32;
    __strong NSError*& build_error = want_bf16 ? build_error_bf16 : build_error_f32;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:178
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"L2N(
#include <metal_stdlib>
using namespace metal;

struct L2NormParams {
    uint n_rows;
    uint n_cols;
    float eps;
    uint total_elems;
};

kernel void supersonic_l2norm_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant L2NormParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint row = gid / params.n_cols;
    uint col = gid - row * params.n_cols;
    uint base = row * params.n_cols;
    float norm_sq = 0.0f;
    for (uint c = 0; c < params.n_cols; ++c) {
        float v = input[base + c];
        norm_sq = fma(v, v, norm_sq);
    }
    float inv_norm = rsqrt(norm_sq + params.eps);
    out[base + col] = input[base + col] * inv_norm;
}

kernel void supersonic_l2norm_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant L2NormParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint row = gid / params.n_cols;
    uint col = gid - row * params.n_cols;
    uint base = row * params.n_cols;
    float norm_sq = 0.0f;
    for (uint c = 0; c < params.n_cols; ++c) {
        float v = float(input[base + c]);
        norm_sq = fma(v, v, norm_sq);
    }
    float inv_norm = rsqrt(norm_sq + params.eps);
    out[base + col] = bfloat(float(input[base + col]) * inv_norm);
}
)L2N";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:179
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile l2norm library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:180
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load l2norm function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:181
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create l2norm pipeline"
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

static int supersonic_metal_element_add_impl(
    size_t total_elems,
    const void* lhs_ptr,
    const void* rhs_ptr,
    void* out_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || lhs_ptr == nullptr || rhs_ptr == nullptr || out_ptr == nullptr) {
            return 71;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = element_add_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 72;
        }

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 73;
        }
        if (lookup_buffer(rhs_ptr, &rhs, &rhs_offset) != 0) {
            return 74;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 75;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 76;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 77;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 78;
        }

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
        [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
        [encoder setBuffer:out offset:out_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 79;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_element_add_bf16(
    size_t total_elems,
    const void* lhs_ptr,
    const void* rhs_ptr,
    void* out_ptr
) {
    return supersonic_metal_element_add_impl(
        total_elems,
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        @"supersonic_element_add_bf16"
    );
}

extern "C" int supersonic_metal_element_add_f32(
    size_t total_elems,
    const void* lhs_ptr,
    const void* rhs_ptr,
    void* out_ptr
) {
    return supersonic_metal_element_add_impl(
        total_elems,
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        @"supersonic_element_add_f32"
    );
}

static int supersonic_metal_sigmoid_mul_impl(
    size_t total_elems,
    const void* data_ptr,
    const void* gate_ptr,
    void* out_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || data_ptr == nullptr || gate_ptr == nullptr || out_ptr == nullptr) {
            return 195;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = sigmoid_mul_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 196;
        }

        id<MTLBuffer> data = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> out = nil;
        size_t data_offset = 0;
        size_t gate_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(data_ptr, &data, &data_offset) != 0) {
            return 197;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 198;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 199;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 200;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 201;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 202;
        }

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:data offset:data_offset atIndex:0];
        [encoder setBuffer:gate offset:gate_offset atIndex:1];
        [encoder setBuffer:out offset:out_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width =
            std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 203;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_sigmoid_mul_bf16(
    size_t total_elems,
    const void* data_ptr,
    const void* gate_ptr,
    void* out_ptr
) {
    return supersonic_metal_sigmoid_mul_impl(
        total_elems,
        data_ptr,
        gate_ptr,
        out_ptr,
        @"supersonic_sigmoid_mul_bf16"
    );
}

extern "C" int supersonic_metal_sigmoid_mul_f32(
    size_t total_elems,
    const void* data_ptr,
    const void* gate_ptr,
    void* out_ptr
) {
    return supersonic_metal_sigmoid_mul_impl(
        total_elems,
        data_ptr,
        gate_ptr,
        out_ptr,
        @"supersonic_sigmoid_mul_f32"
    );
}

static int supersonic_metal_swiglu_mul_impl(
    size_t total_elems,
    const void* gate_ptr,
    const void* up_ptr,
    void* out_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || gate_ptr == nullptr || up_ptr == nullptr || out_ptr == nullptr) {
            return 215;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = swiglu_mul_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 216;
        }

        id<MTLBuffer> gate = nil;
        id<MTLBuffer> up = nil;
        id<MTLBuffer> out = nil;
        size_t gate_offset = 0;
        size_t up_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 217;
        }
        if (lookup_buffer(up_ptr, &up, &up_offset) != 0) {
            return 218;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 219;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 220;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 221;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 222;
        }

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:gate offset:gate_offset atIndex:0];
        [encoder setBuffer:up offset:up_offset atIndex:1];
        [encoder setBuffer:out offset:out_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width =
            std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 223;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_swiglu_mul_bf16(
    size_t total_elems,
    const void* gate_ptr,
    const void* up_ptr,
    void* out_ptr
) {
    return supersonic_metal_swiglu_mul_impl(
        total_elems,
        gate_ptr,
        up_ptr,
        out_ptr,
        @"supersonic_swiglu_mul_bf16"
    );
}

extern "C" int supersonic_metal_swiglu_mul_f32(
    size_t total_elems,
    const void* gate_ptr,
    const void* up_ptr,
    void* out_ptr
) {
    return supersonic_metal_swiglu_mul_impl(
        total_elems,
        gate_ptr,
        up_ptr,
        out_ptr,
        @"supersonic_swiglu_mul_f32"
    );
}

static int supersonic_metal_cast_impl(
    size_t total_elems,
    const void* input_ptr,
    void* out_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || input_ptr == nullptr || out_ptr == nullptr) {
            return 81;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = cast_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 82;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> out = nil;
        size_t input_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 83;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 84;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 85;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 86;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 87;
        }

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:input_offset atIndex:0];
        [encoder setBuffer:out offset:out_offset atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 88;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_cast_bf16_to_bf16(
    size_t total_elems,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_cast_impl(total_elems, input_ptr, out_ptr, @"supersonic_cast_bf16_to_bf16");
}

extern "C" int supersonic_metal_cast_f32_to_f32(
    size_t total_elems,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_cast_impl(total_elems, input_ptr, out_ptr, @"supersonic_cast_f32_to_f32");
}

extern "C" int supersonic_metal_cast_u32_to_u32(
    size_t total_elems,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_cast_impl(total_elems, input_ptr, out_ptr, @"supersonic_cast_u32_to_u32");
}

extern "C" int supersonic_metal_cast_bf16_to_f32(
    size_t total_elems,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_cast_impl(total_elems, input_ptr, out_ptr, @"supersonic_cast_bf16_to_f32");
}

extern "C" int supersonic_metal_cast_f32_to_bf16(
    size_t total_elems,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_cast_impl(total_elems, input_ptr, out_ptr, @"supersonic_cast_f32_to_bf16");
}

static int supersonic_metal_mul_scalar_impl(
    size_t total_elems,
    float scalar,
    const void* input_ptr,
    void* out_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || input_ptr == nullptr || out_ptr == nullptr) {
            return 91;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = mul_scalar_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 92;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> out = nil;
        size_t input_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 93;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 94;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 95;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 96;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 97;
        }

        MulScalarParams params = {static_cast<uint32_t>(total_elems), scalar};

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:input_offset atIndex:0];
        [encoder setBuffer:out offset:out_offset atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 98;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_mul_scalar_bf16(
    size_t total_elems,
    float scalar,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_mul_scalar_impl(
        total_elems,
        scalar,
        input_ptr,
        out_ptr,
        @"supersonic_mul_scalar_bf16"
    );
}

extern "C" int supersonic_metal_mul_scalar_f32(
    size_t total_elems,
    float scalar,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_mul_scalar_impl(
        total_elems,
        scalar,
        input_ptr,
        out_ptr,
        @"supersonic_mul_scalar_f32"
    );
}

static int supersonic_metal_transpose_shd_hsd_impl(
    size_t s,
    size_t h,
    size_t d,
    const void* src_ptr,
    void* dst_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (s == 0 || h == 0 || d == 0) {
            return 0;
        }
        if (s > UINT32_MAX || h > UINT32_MAX || d > UINT32_MAX || src_ptr == nullptr || dst_ptr == nullptr) {
            return 101;
        }
        size_t total_elems = s * h * d;
        if (total_elems > UINT32_MAX || total_elems / d / h != s) {
            return 102;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = transpose_shd_hsd_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 103;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 104;
        }
        if (lookup_buffer(dst_ptr, &dst, &dst_offset) != 0) {
            return 105;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 106;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 107;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 108;
        }

        TransposeShdHsdParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(h),
            static_cast<uint32_t>(d),
            static_cast<uint32_t>(total_elems),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src offset:src_offset atIndex:0];
        [encoder setBuffer:dst offset:dst_offset atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 109;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_transpose_shd_hsd_bf16(
    size_t s,
    size_t h,
    size_t d,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_transpose_shd_hsd_impl(
        s,
        h,
        d,
        src_ptr,
        dst_ptr,
        @"supersonic_transpose_shd_hsd_bf16"
    );
}

extern "C" int supersonic_metal_transpose_shd_hsd_f32(
    size_t s,
    size_t h,
    size_t d,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_transpose_shd_hsd_impl(
        s,
        h,
        d,
        src_ptr,
        dst_ptr,
        @"supersonic_transpose_shd_hsd_f32"
    );
}

static int supersonic_metal_split_qkv_impl(
    size_t s,
    size_t key_dim,
    size_t val_dim,
    const void* src_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (s == 0 || (key_dim == 0 && val_dim == 0)) {
            return 0;
        }
        if (s > UINT32_MAX || key_dim > UINT32_MAX || val_dim > UINT32_MAX || src_ptr == nullptr ||
            q_ptr == nullptr || k_ptr == nullptr || v_ptr == nullptr) {
            return 111;
        }
        if (key_dim > (SIZE_MAX - val_dim) / 2) {
            return 112;
        }
        size_t src_stride = key_dim * 2 + val_dim;
        if (src_stride > UINT32_MAX || src_stride < key_dim || src_stride < val_dim) {
            return 112;
        }
        size_t total_elems = s * src_stride;
        if (total_elems > UINT32_MAX || (src_stride != 0 && total_elems / src_stride != s)) {
            return 113;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = split_qkv_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 114;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> q = nil;
        id<MTLBuffer> k = nil;
        id<MTLBuffer> v = nil;
        size_t src_offset = 0;
        size_t q_offset = 0;
        size_t k_offset = 0;
        size_t v_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 115;
        }
        if (lookup_buffer(q_ptr, &q, &q_offset) != 0) {
            return 116;
        }
        if (lookup_buffer(k_ptr, &k, &k_offset) != 0) {
            return 117;
        }
        if (lookup_buffer(v_ptr, &v, &v_offset) != 0) {
            return 118;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 119;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 120;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 121;
        }

        SplitQkvParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(key_dim),
            static_cast<uint32_t>(val_dim),
            static_cast<uint32_t>(src_stride),
            static_cast<uint32_t>(total_elems),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src offset:src_offset atIndex:0];
        [encoder setBuffer:q offset:q_offset atIndex:1];
        [encoder setBuffer:k offset:k_offset atIndex:2];
        [encoder setBuffer:v offset:v_offset atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 122;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_split_qkv_bf16(
    size_t s,
    size_t key_dim,
    size_t val_dim,
    const void* src_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr
) {
    return supersonic_metal_split_qkv_impl(
        s,
        key_dim,
        val_dim,
        src_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        @"supersonic_split_qkv_bf16"
    );
}

extern "C" int supersonic_metal_split_qkv_f32(
    size_t s,
    size_t key_dim,
    size_t val_dim,
    const void* src_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr
) {
    return supersonic_metal_split_qkv_impl(
        s,
        key_dim,
        val_dim,
        src_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        @"supersonic_split_qkv_f32"
    );
}

static int supersonic_metal_split_qgate_impl(
    size_t s,
    size_t num_heads,
    size_t head_dim,
    const void* src_ptr,
    void* query_ptr,
    void* gate_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (s == 0 || num_heads == 0 || head_dim == 0) {
            return 0;
        }
        if (s > UINT32_MAX || num_heads > UINT32_MAX || head_dim > UINT32_MAX ||
            src_ptr == nullptr || query_ptr == nullptr || gate_ptr == nullptr) {
            return 123;
        }
        if (num_heads != 0 && head_dim > SIZE_MAX / num_heads) {
            return 124;
        }
        size_t dst_stride = num_heads * head_dim;
        if (head_dim > SIZE_MAX / 2) {
            return 124;
        }
        size_t per_head_src = head_dim * 2;
        if (num_heads != 0 && per_head_src > SIZE_MAX / num_heads) {
            return 124;
        }
        size_t src_stride = num_heads * per_head_src;
        if (s != 0 && dst_stride > SIZE_MAX / s) {
            return 125;
        }
        size_t total_elems = s * dst_stride;
        if (total_elems > UINT32_MAX || src_stride > UINT32_MAX) {
            return 125;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = split_qgate_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 126;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> query = nil;
        id<MTLBuffer> gate = nil;
        size_t src_offset = 0;
        size_t query_offset = 0;
        size_t gate_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 127;
        }
        if (lookup_buffer(query_ptr, &query, &query_offset) != 0) {
            return 128;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 129;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 130;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 131;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 132;
        }

        SplitQgateParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(src_stride),
            static_cast<uint32_t>(total_elems),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src offset:src_offset atIndex:0];
        [encoder setBuffer:query offset:query_offset atIndex:1];
        [encoder setBuffer:gate offset:gate_offset atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 133;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_split_qgate_bf16(
    size_t s,
    size_t num_heads,
    size_t head_dim,
    const void* src_ptr,
    void* query_ptr,
    void* gate_ptr
) {
    return supersonic_metal_split_qgate_impl(
        s,
        num_heads,
        head_dim,
        src_ptr,
        query_ptr,
        gate_ptr,
        @"supersonic_split_qgate_bf16"
    );
}

extern "C" int supersonic_metal_split_qgate_f32(
    size_t s,
    size_t num_heads,
    size_t head_dim,
    const void* src_ptr,
    void* query_ptr,
    void* gate_ptr
) {
    return supersonic_metal_split_qgate_impl(
        s,
        num_heads,
        head_dim,
        src_ptr,
        query_ptr,
        gate_ptr,
        @"supersonic_split_qgate_f32"
    );
}

static int supersonic_metal_repeat_interleave_heads_impl(
    size_t s,
    size_t n_heads,
    size_t head_dim,
    size_t repeats,
    const void* src_ptr,
    void* dst_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (s == 0 || n_heads == 0 || head_dim == 0 || repeats == 0) {
            return 0;
        }
        if (s > UINT32_MAX || n_heads > UINT32_MAX || head_dim > UINT32_MAX || repeats > UINT32_MAX ||
            src_ptr == nullptr || dst_ptr == nullptr) {
            return 134;
        }
        if (n_heads != 0 && repeats > SIZE_MAX / n_heads) {
            return 135;
        }
        size_t dst_heads = n_heads * repeats;
        if (dst_heads > UINT32_MAX || (s != 0 && dst_heads > SIZE_MAX / s)) {
            return 135;
        }
        if (head_dim != 0 && dst_heads > SIZE_MAX / head_dim) {
            return 136;
        }
        size_t total_elems = s * dst_heads * head_dim;
        if (total_elems > UINT32_MAX) {
            return 136;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            repeat_interleave_heads_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 137;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 138;
        }
        if (lookup_buffer(dst_ptr, &dst, &dst_offset) != 0) {
            return 139;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 140;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 141;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 142;
        }

        RepeatInterleaveHeadsParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(n_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(repeats),
            static_cast<uint32_t>(dst_heads),
            static_cast<uint32_t>(total_elems),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src offset:src_offset atIndex:0];
        [encoder setBuffer:dst offset:dst_offset atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 143;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_repeat_interleave_heads_bf16(
    size_t s,
    size_t n_heads,
    size_t head_dim,
    size_t repeats,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_repeat_interleave_heads_impl(
        s,
        n_heads,
        head_dim,
        repeats,
        src_ptr,
        dst_ptr,
        @"supersonic_repeat_interleave_heads_bf16"
    );
}

extern "C" int supersonic_metal_repeat_interleave_heads_f32(
    size_t s,
    size_t n_heads,
    size_t head_dim,
    size_t repeats,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_repeat_interleave_heads_impl(
        s,
        n_heads,
        head_dim,
        repeats,
        src_ptr,
        dst_ptr,
        @"supersonic_repeat_interleave_heads_f32"
    );
}

extern "C" int supersonic_metal_compute_beta_g_f32(
    size_t seq_len,
    size_t nv,
    const void* b_ptr,
    const void* a_ptr,
    const void* dt_bias_ptr,
    const void* a_log_exp_ptr,
    void* beta_ptr,
    void* g_ptr
) {
    @autoreleasepool {
        if (seq_len == 0 || nv == 0) {
            return 0;
        }
        if (seq_len > UINT32_MAX || nv > UINT32_MAX || b_ptr == nullptr || a_ptr == nullptr ||
            dt_bias_ptr == nullptr || a_log_exp_ptr == nullptr || beta_ptr == nullptr || g_ptr == nullptr) {
            return 144;
        }
        size_t total_elems = seq_len * nv;
        if (total_elems > UINT32_MAX || (nv != 0 && total_elems / nv != seq_len)) {
            return 145;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = compute_beta_g_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 146;
        }

        id<MTLBuffer> b = nil;
        id<MTLBuffer> a = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log_exp = nil;
        id<MTLBuffer> beta = nil;
        id<MTLBuffer> g = nil;
        size_t b_offset = 0;
        size_t a_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_exp_offset = 0;
        size_t beta_offset = 0;
        size_t g_offset = 0;
        if (lookup_buffer(b_ptr, &b, &b_offset) != 0) {
            return 147;
        }
        if (lookup_buffer(a_ptr, &a, &a_offset) != 0) {
            return 148;
        }
        if (lookup_buffer(dt_bias_ptr, &dt_bias, &dt_bias_offset) != 0) {
            return 149;
        }
        if (lookup_buffer(a_log_exp_ptr, &a_log_exp, &a_log_exp_offset) != 0) {
            return 150;
        }
        if (lookup_buffer(beta_ptr, &beta, &beta_offset) != 0) {
            return 151;
        }
        if (lookup_buffer(g_ptr, &g, &g_offset) != 0) {
            return 152;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 153;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 154;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 155;
        }

        ComputeBetaGParams params = {
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(nv),
            static_cast<uint32_t>(total_elems),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:b offset:b_offset atIndex:0];
        [encoder setBuffer:a offset:a_offset atIndex:1];
        [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:2];
        [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:3];
        [encoder setBuffer:beta offset:beta_offset atIndex:4];
        [encoder setBuffer:g offset:g_offset atIndex:5];
        [encoder setBytes:&params length:sizeof(params) atIndex:6];

        NSUInteger tg_width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 156;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_delta_recurrent_prefill_f32(
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state_ptr,
    const void* query_ptr,
    const void* key_ptr,
    const void* value_ptr,
    const void* beta_ptr,
    const void* g_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (batch_heads == 0 || seq_len == 0 || k_head_dim == 0 || v_head_dim == 0) {
            return 161;
        }
        if (batch_heads > UINT32_MAX || seq_len > UINT32_MAX || k_head_dim > UINT32_MAX ||
            v_head_dim > UINT32_MAX || initial_state_ptr == nullptr || query_ptr == nullptr ||
            key_ptr == nullptr || value_ptr == nullptr || beta_ptr == nullptr || g_ptr == nullptr ||
            out_ptr == nullptr) {
            return 162;
        }

        if (v_head_dim != 0 && batch_heads > SIZE_MAX / v_head_dim) {
            return 163;
        }
        size_t total_threads = batch_heads * v_head_dim;
        if (total_threads > UINT32_MAX) {
            return 164;
        }
        if (k_head_dim > SIZE_MAX - seq_len) {
            return 165;
        }
        size_t out_rows = seq_len + k_head_dim;
        if (out_rows > UINT32_MAX) {
            return 165;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = delta_recurrent_prefill_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 166;
        }

        id<MTLBuffer> initial_state = nil;
        id<MTLBuffer> query = nil;
        id<MTLBuffer> key = nil;
        id<MTLBuffer> value = nil;
        id<MTLBuffer> beta = nil;
        id<MTLBuffer> g = nil;
        id<MTLBuffer> out = nil;
        size_t initial_state_offset = 0;
        size_t query_offset = 0;
        size_t key_offset = 0;
        size_t value_offset = 0;
        size_t beta_offset = 0;
        size_t g_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(initial_state_ptr, &initial_state, &initial_state_offset) != 0) {
            return 167;
        }
        if (lookup_buffer(query_ptr, &query, &query_offset) != 0) {
            return 168;
        }
        if (lookup_buffer(key_ptr, &key, &key_offset) != 0) {
            return 169;
        }
        if (lookup_buffer(value_ptr, &value, &value_offset) != 0) {
            return 170;
        }
        if (lookup_buffer(beta_ptr, &beta, &beta_offset) != 0) {
            return 171;
        }
        if (lookup_buffer(g_ptr, &g, &g_offset) != 0) {
            return 172;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 173;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 174;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 175;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 176;
        }

        DeltaRecurrentPrefillParams params = {
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(k_head_dim),
            static_cast<uint32_t>(v_head_dim),
            static_cast<uint32_t>(out_rows),
            static_cast<uint32_t>(total_threads),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:initial_state offset:initial_state_offset atIndex:0];
        [encoder setBuffer:query offset:query_offset atIndex:1];
        [encoder setBuffer:key offset:key_offset atIndex:2];
        [encoder setBuffer:value offset:value_offset atIndex:3];
        [encoder setBuffer:beta offset:beta_offset atIndex:4];
        [encoder setBuffer:g offset:g_offset atIndex:5];
        [encoder setBuffer:out offset:out_offset atIndex:6];
        [encoder setBytes:&params length:sizeof(params) atIndex:7];

        NSUInteger tg_width =
            std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_threads, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 177;
        }
        return 0;
    }
}

static int supersonic_metal_l2norm_impl(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* input_ptr,
    void* out_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (n_rows == 0 || n_cols == 0) {
            return 0;
        }
        if (n_rows > UINT32_MAX || n_cols > UINT32_MAX || input_ptr == nullptr || out_ptr == nullptr) {
            return 182;
        }
        if (n_cols != 0 && n_rows > SIZE_MAX / n_cols) {
            return 183;
        }
        size_t total_elems = n_rows * n_cols;
        if (total_elems > UINT32_MAX) {
            return 183;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = l2norm_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 184;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> out = nil;
        size_t input_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 185;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 186;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 187;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 188;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 189;
        }

        L2NormParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            static_cast<uint32_t>(total_elems),
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:input_offset atIndex:0];
        [encoder setBuffer:out offset:out_offset atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];

        NSUInteger tg_width =
            std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
        MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
        MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
        [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 190;
        }
        return 0;
    }
}

extern "C" int supersonic_metal_l2norm_f32(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_l2norm_impl(n_rows, n_cols, eps, input_ptr, out_ptr, @"supersonic_l2norm_f32");
}

extern "C" int supersonic_metal_l2norm_bf16(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* input_ptr,
    void* out_ptr
) {
    return supersonic_metal_l2norm_impl(n_rows, n_cols, eps, input_ptr, out_ptr, @"supersonic_l2norm_bf16");
}

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

extern "C" int supersonic_metal_rms_norm_gated_bf16(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden_ptr,
    const void* gate_ptr,
    const void* weight_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (n_rows == 0 || n_cols == 0 || n_rows > UINT32_MAX || n_cols > UINT32_MAX ||
            hidden_ptr == nullptr || gate_ptr == nullptr || weight_ptr == nullptr || out_ptr == nullptr) {
            return 228;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = rms_norm_gated_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 229;
        }

        id<MTLBuffer> hidden = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> out = nil;
        size_t hidden_offset = 0;
        size_t gate_offset = 0;
        size_t weight_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(hidden_ptr, &hidden, &hidden_offset) != 0) {
            return 230;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 231;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 232;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 233;
        }

        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 234;
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 235;
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return 236;
        }

        RmsNormGatedParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
        };

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:hidden offset:hidden_offset atIndex:0];
        [encoder setBuffer:gate offset:gate_offset atIndex:1];
        [encoder setBuffer:weight offset:weight_offset atIndex:2];
        [encoder setBuffer:out offset:out_offset atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

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
            return 237;
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
