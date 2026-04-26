#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdint.h>
#include <string>

extern "C" int supersonic_metal_lookup_buffer(
    const void* ptr,
    void** buffer_out,
    size_t* offset_out
);
extern "C" void supersonic_metal_profile_record(
    const char* op,
    const char* path,
    double elapsed_ms
);

namespace {

using MetalClock = std::chrono::steady_clock;

inline void record_runtime_profile(const char* op, MetalClock::time_point start) {
    supersonic_metal_profile_record(
        op,
        "runtime",
        std::chrono::duration<double, std::milli>(MetalClock::now() - start).count()
    );
}

inline void record_profile_elapsed(const char* op, const char* path, double elapsed_ms) {
    if (std::isfinite(elapsed_ms) && elapsed_ms >= 0.0) {
        supersonic_metal_profile_record(op, path, elapsed_ms);
    }
}

inline void record_command_buffer_gpu_profile(id<MTLCommandBuffer> command_buffer, const std::string& label) {
    if (command_buffer == nil) {
        return;
    }
    double gpu_elapsed_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    record_profile_elapsed("command_buffer_gpu", "runtime", gpu_elapsed_ms);
    if (!label.empty()) {
        std::string labeled_op = "command_buffer_gpu:" + label;
        record_profile_elapsed(labeled_op.c_str(), "runtime", gpu_elapsed_ms);
    }
}

struct MatmulParams {
    uint32_t batch_elems;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct QwenLinearProjectionParams {
    uint32_t hidden_dim;
    uint32_t qkv_dim;
    uint32_t val_dim;
    uint32_t num_value_heads;
    uint32_t total_cols;
};

struct QwenMlpParams {
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
};

struct QwenFullProjectionParams {
    uint32_t hidden_dim;
    uint32_t q_proj_dim;
    uint32_t kv_dim;
    uint32_t total_cols;
};

struct FullAttentionParams {
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t kv_stride;
    uint32_t head_dim;
    uint32_t seqlen_offset;
    float scale;
};

struct FullAttentionDecodeParams {
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t kv_len;
    uint32_t kv_stride;
    uint32_t head_dim;
    float scale;
};

struct EmbeddingLookupParams {
    uint32_t token_count;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t total_elems;
};

struct RmsNormParams {
    uint32_t n_rows;
    uint32_t n_cols;
    float eps;
    uint32_t add_unit_offset;
    uint32_t block_size;
};

struct RopeParams {
    uint32_t seq_len;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t half_rot;
    uint32_t pos_offset;
    uint32_t total_pairs;
};

struct TransposePadConvParams {
    uint32_t s;
    uint32_t c;
    uint32_t pad;
    uint32_t stride;
    uint32_t total_dst;
};

struct ExtractConvStateParams {
    uint32_t s;
    uint32_t c;
    uint32_t kern_minus_1;
    uint32_t copy;
    uint32_t start;
    uint32_t dst_start;
    uint32_t total_dst;
};

struct RmsNormGatedParams {
    uint32_t n_rows;
    uint32_t n_cols;
    float eps;
    uint32_t block_size;
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

struct LinearDecodeApplyParams {
    uint32_t num_v_heads;
    uint32_t num_k_heads;
    uint32_t head_repeat;
    uint32_t k_head_dim;
    uint32_t v_head_dim;
    uint32_t value_dim;
    uint32_t state_dim;
    uint32_t total_threads;
};

struct QwenLinearPrepParams {
    uint32_t key_dim;
    uint32_t val_dim;
    uint32_t num_key_heads;
    uint32_t key_head_dim;
    uint32_t total_threads;
    float eps;
    float q_scale;
};

struct QwenLinearPrepDecodeApplyParams {
    uint32_t num_v_heads;
    uint32_t num_k_heads;
    uint32_t head_repeat;
    uint32_t k_head_dim;
    uint32_t v_head_dim;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t state_dim;
    uint32_t total_threads;
    float eps;
    float q_scale;
};

struct ConvStateUpdateParams {
    uint32_t channels;
    uint32_t state_len;
    uint32_t total_threads;
};

struct LinearConvValueDecayParams {
    uint32_t conv_dim;
    uint32_t state_len;
    uint32_t kernel_size;
    uint32_t num_heads;
    uint32_t out_width;
    uint32_t total_threads;
};

struct L2NormParams {
    uint32_t n_rows;
    uint32_t n_cols;
    float eps;
    uint32_t total_elems;
    uint32_t block_size;
};

struct LmHeadArgmaxParams {
    uint32_t in_dim;
    uint32_t vocab_size;
    uint32_t block_size;
    uint32_t partial_count;
};

id<MTLDevice> metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> metal_queue() {
    static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
    return queue;
}

struct MetalBatchState {
    __strong id<MTLCommandBuffer> command_buffer = nil;
    __strong id<MTLComputeCommandEncoder> encoder = nil;
    bool has_work = false;
    std::string label;
};

thread_local int metal_batch_depth = 0;
thread_local MetalBatchState* metal_batch_state = nullptr;

int metal_batch_ensure_command_buffer() {
    if (metal_batch_state == nullptr) {
        metal_batch_state = new MetalBatchState();
    }
    if (metal_batch_state->command_buffer != nil) {
        return 0;
    }
    id<MTLCommandQueue> queue = metal_queue();
    if (queue == nil) {
        return 901;
    }
    auto command_buffer_start = MetalClock::now();
    metal_batch_state->command_buffer = [queue commandBuffer];
    record_runtime_profile("command_buffer_create", command_buffer_start);
    if (metal_batch_state->command_buffer == nil) {
        return 902;
    }
    metal_batch_state->has_work = false;
    return 0;
}

int metal_batch_ensure_compute_encoder() {
    int status = metal_batch_ensure_command_buffer();
    if (status != 0) {
        return status;
    }
    if (metal_batch_state->encoder != nil) {
        return 0;
    }
    auto encoder_start = MetalClock::now();
    metal_batch_state->encoder = [metal_batch_state->command_buffer computeCommandEncoder];
    record_runtime_profile("compute_encoder_create", encoder_start);
    if (metal_batch_state->encoder == nil) {
        metal_batch_state->command_buffer = nil;
        return 903;
    }
    return 0;
}

int metal_batch_close_encoder(bool restart) {
    if (metal_batch_state == nullptr || metal_batch_state->command_buffer == nil) {
        return 0;
    }

    id<MTLCommandBuffer> command_buffer = metal_batch_state->command_buffer;
    id<MTLComputeCommandEncoder> encoder = metal_batch_state->encoder;
    bool has_work = metal_batch_state->has_work;
    std::string label = metal_batch_state->label;
    metal_batch_state->encoder = nil;
    metal_batch_state->command_buffer = nil;
    metal_batch_state->has_work = false;
    metal_batch_state->label.clear();

    if (encoder != nil) {
        auto end_encoding_start = MetalClock::now();
        [encoder endEncoding];
        record_runtime_profile("encoder_end", end_encoding_start);
    }
    if (has_work) {
        auto commit_start = MetalClock::now();
        [command_buffer commit];
        record_runtime_profile("command_buffer_commit", commit_start);
        auto wait_start = MetalClock::now();
        [command_buffer waitUntilCompleted];
        record_runtime_profile("command_buffer_wait", wait_start);
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 904;
        }
        record_command_buffer_gpu_profile(command_buffer, label);
    }

    (void)restart;
    return 0;
}

int metal_batch_set_label(const char* label) {
    if (metal_batch_depth <= 0) {
        return 0;
    }
    if (metal_batch_state == nullptr) {
        metal_batch_state = new MetalBatchState();
    }
    metal_batch_state->label = label == nullptr ? "" : label;
    return 0;
}

int metal_batch_begin() {
    if (metal_batch_depth++ == 0) {
        if (metal_batch_state == nullptr) {
            metal_batch_state = new MetalBatchState();
        }
    }
    return 0;
}

int metal_batch_flush() {
    if (metal_batch_depth <= 0) {
        return 0;
    }
    return metal_batch_close_encoder(true);
}

int metal_batch_end() {
    if (metal_batch_depth <= 0) {
        return 905;
    }
    metal_batch_depth--;
    if (metal_batch_depth == 0) {
        int status = metal_batch_close_encoder(false);
        delete metal_batch_state;
        metal_batch_state = nullptr;
        return status;
    }
    return 0;
}

template <typename EncodeFn>
int encode_or_submit(EncodeFn encode, int queue_error, int command_buffer_error, int encoder_error, int completion_error) {
    if (metal_batch_depth > 0 && metal_batch_state != nullptr) {
        int status = metal_batch_ensure_compute_encoder();
        if (status != 0) {
            return status;
        }
        encode(metal_batch_state->encoder);
        metal_batch_state->has_work = true;
        [metal_batch_state->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 0;
    }

    id<MTLCommandQueue> queue = metal_queue();
    if (queue == nil) {
        return queue_error;
    }
    auto command_buffer_start = MetalClock::now();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    record_runtime_profile("command_buffer_create", command_buffer_start);
    if (command_buffer == nil) {
        return command_buffer_error;
    }
    auto encoder_start = MetalClock::now();
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    record_runtime_profile("compute_encoder_create", encoder_start);
    if (encoder == nil) {
        return encoder_error;
    }

    encode(encoder);
    auto end_encoding_start = MetalClock::now();
    [encoder endEncoding];
    record_runtime_profile("encoder_end", end_encoding_start);
    auto commit_start = MetalClock::now();
    [command_buffer commit];
    record_runtime_profile("command_buffer_commit", commit_start);
    auto wait_start = MetalClock::now();
    [command_buffer waitUntilCompleted];
    record_runtime_profile("command_buffer_wait", wait_start);

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return completion_error;
    }
    record_command_buffer_gpu_profile(command_buffer, "");
    return 0;
}

int encode_blit_copy_or_submit(
    id<MTLBuffer> src,
    NSUInteger src_offset,
    id<MTLBuffer> dst,
    NSUInteger dst_offset,
    NSUInteger bytes,
    int queue_error,
    int command_buffer_error,
    int encoder_error,
    int completion_error
) {
    if (metal_batch_depth > 0 && metal_batch_state != nullptr) {
        if (metal_batch_state->command_buffer == nil) {
            int status = metal_batch_ensure_command_buffer();
            if (status != 0) {
                return status;
            }
        }
        if (metal_batch_state->encoder != nil) {
            auto end_encoding_start = MetalClock::now();
            [metal_batch_state->encoder endEncoding];
            record_runtime_profile("encoder_end", end_encoding_start);
            metal_batch_state->encoder = nil;
        }

        auto blit_start = MetalClock::now();
        id<MTLBlitCommandEncoder> blit = [metal_batch_state->command_buffer blitCommandEncoder];
        record_runtime_profile("blit_encoder_create", blit_start);
        if (blit == nil) {
            return encoder_error;
        }
        [blit copyFromBuffer:src
                sourceOffset:src_offset
                    toBuffer:dst
           destinationOffset:dst_offset
                        size:bytes];
        auto blit_end_start = MetalClock::now();
        [blit endEncoding];
        record_runtime_profile("encoder_end", blit_end_start);
        metal_batch_state->has_work = true;
        return 0;
    }

    id<MTLCommandQueue> queue = metal_queue();
    if (queue == nil) {
        return queue_error;
    }
    auto command_buffer_start = MetalClock::now();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    record_runtime_profile("command_buffer_create", command_buffer_start);
    if (command_buffer == nil) {
        return command_buffer_error;
    }
    auto blit_start = MetalClock::now();
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    record_runtime_profile("blit_encoder_create", blit_start);
    if (blit == nil) {
        return encoder_error;
    }
    [blit copyFromBuffer:src
            sourceOffset:src_offset
                toBuffer:dst
       destinationOffset:dst_offset
                    size:bytes];
    auto blit_end_start = MetalClock::now();
    [blit endEncoding];
    record_runtime_profile("encoder_end", blit_end_start);
    auto commit_start = MetalClock::now();
    [command_buffer commit];
    record_runtime_profile("command_buffer_commit", commit_start);
    auto wait_start = MetalClock::now();
    [command_buffer waitUntilCompleted];
    record_runtime_profile("command_buffer_wait", wait_start);
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return completion_error;
    }
    record_command_buffer_gpu_profile(command_buffer, "");
    return 0;
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

// SIMD-group cooperative GEMV for the M=1 decode case: out[1, n] = lhs[1, k] · rhs[n, k]^T.
// One SIMD-group per output column (32 threads cooperate on the K reduction);
// `simd_sum` collapses partials, lane 0 writes the result. This replaces the
// per-cell sequential-K kernel for M=1 — the path lm_head, q/k/v/o projections,
// and MLP gate/up/down all hit during decode.
id<MTLComputePipelineState> matmul_pipeline_bf16_gemv_m1(NSError** error_out) {
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
                                                   code:501
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"GEMV(
#include <metal_stdlib>
using namespace metal;

struct MatmulGemvParams {
    uint n;
    uint k;
};

kernel void supersonic_matmul_rhs_transposed_bf16_gemv_m1(
    device const bfloat* lhs [[buffer(0)]],
    device const bfloat* rhs [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant MatmulGemvParams& params [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (col >= params.n) {
        return;
    }
    uint k = params.k;
    uint rhs_base = col * k;
    float partial = 0.0f;
    for (uint kk = lane; kk < k; kk += 32u) {
        partial += float(lhs[kk]) * float(rhs[rhs_base + kk]);
    }
    float sum = simd_sum(partial);
    if (lane == 0) {
        out[col] = bfloat(sum);
    }
}
)GEMV";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:502
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile gemv library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_bf16_gemv_m1"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:503
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load gemv function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:504
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create gemv pipeline"
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

// Tiled SIMD-group GEMV. Each threadgroup handles 32 output columns. All 1024
// threads cooperate to load `lhs` (K elements) into threadgroup memory once,
// then 32 SIMD-groups each compute one output column from that shared cache.
// The reuse factor on `lhs` device reads is 32x. Biggest win on lm_head where
// N is huge (248k) and the same K=1024 lhs vector is otherwise re-read by
// every output cell. Caller must ensure K*4 bytes fit within the device's
// threadgroup memory limit (~32KB on Apple Silicon → K <= 8192).
id<MTLComputePipelineState> matmul_pipeline_bf16_gemv_m1_tiled(NSError** error_out) {
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
                                                   code:601
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"GEMVTILED(
#include <metal_stdlib>
using namespace metal;

struct MatmulGemvParams {
    uint n;
    uint k;
};

kernel void supersonic_matmul_rhs_transposed_bf16_gemv_m1_tiled(
    device const bfloat* lhs [[buffer(0)]],
    device const bfloat* rhs [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant MatmulGemvParams& params [[buffer(3)]],
    threadgroup float* shared_lhs [[threadgroup(0)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint k = params.k;
    uint n = params.n;
    // Cooperative load: 1024 threads × ceil(K/1024) elements each.
    for (uint i = thread_id; i < k; i += 1024u) {
        shared_lhs[i] = float(lhs[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 32 SIMD-groups × 32 cols per threadgroup.
    uint col = tg_id * 32u + simd_id;
    if (col >= n) {
        return;
    }
    uint rhs_base = col * k;
    float partial = 0.0f;
    for (uint kk = simd_lane; kk < k; kk += 32u) {
        partial += shared_lhs[kk] * float(rhs[rhs_base + kk]);
    }
    float sum = simd_sum(partial);
    if (simd_lane == 0) {
        out[col] = bfloat(sum);
    }
}
)GEMVTILED";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:602
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile gemv tiled library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_bf16_gemv_m1_tiled"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:603
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load gemv tiled function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:604
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create gemv tiled pipeline"
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

id<MTLComputePipelineState> matmul_pipeline_f32(NSError** error_out) {
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
                                                   code:301
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"MATMULF32(
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint batch_elems;
    uint m;
    uint n;
    uint k;
};

kernel void supersonic_matmul_rhs_transposed_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
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
        acc = fma(lhs[lhs_base + kk], rhs[rhs_base + kk], acc);
    }
    out[(batch * params.m + row) * params.n + col] = acc;
}
)MATMULF32";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:302
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile F32 matmul library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:303
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load F32 matmul function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:304
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create F32 matmul pipeline"
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

id<MTLComputePipelineState> matmul_int4_pipeline_bf16(NSError** error_out) {
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
                                                   code:401
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                // GPTQ INT4 dequant matmul, bit-exact with the HIP/CUDA path:
                //   rhs       [batch, n, k/2]      packed u8 (low nibble = even k index)
                //   scale     [n/group, k/group]   bf16
                //   zero      [n/group, k/group]   bf16
                //   out       [batch, m, n]        bf16 = lhs · dequant(rhs)^T
                // Dequant: w = bf16_round_rne(nibble * scale - zero * scale).
                static const char* kSource = R"INT4MATMUL(
#include <metal_stdlib>
using namespace metal;

struct MatmulInt4Params {
    uint batch_elems;
    uint m;
    uint n;
    uint k;
    uint group_size;
};

inline float bf16_round_rne_finite(float x) {
    uint bits = as_type<uint>(x);
    uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    return as_type<float>(bits);
}

kernel void supersonic_matmul_rhs_transposed_int4_bf16(
    device const bfloat* lhs [[buffer(0)]],
    device const uchar* rhs_int4 [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device const bfloat* zero [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant MatmulInt4Params& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint batch = gid.z;
    if (batch >= params.batch_elems || row >= params.m || col >= params.n) {
        return;
    }
    uint k = params.k;
    uint k_packed = k / 2u;
    uint group_size = params.group_size;
    uint scale_cols = (k + group_size - 1u) / group_size;
    uint scale_row = col / group_size;

    uint lhs_base = (batch * params.m + row) * k;
    uint rhs_base = (batch * params.n + col) * k_packed;

    float acc = 0.0f;
    for (uint kk = 0; kk < k; ++kk) {
        uint byte_idx = kk >> 1u;
        uint packed_byte = uint(rhs_int4[rhs_base + byte_idx]);
        uint nibble = (kk & 1u) != 0u ? ((packed_byte >> 4u) & 0xFu) : (packed_byte & 0xFu);
        uint sc_idx = scale_row * scale_cols + (kk / group_size);
        float s = float(scale[sc_idx]);
        float z = float(zero[sc_idx]);
        float w = bf16_round_rne_finite(float(nibble) * s - z * s);
        acc += float(lhs[lhs_base + kk]) * w;
    }
    out[(batch * params.m + row) * params.n + col] = bfloat(acc);
}
)INT4MATMUL";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:402
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile int4 matmul library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_int4_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:403
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load int4 matmul function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:404
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create int4 matmul pipeline"
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

id<MTLComputePipelineState> matmul_residual_pipeline_bf16(NSError** error_out) {
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
                                                   code:341
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"MATMULRESIDUAL(
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint batch_elems;
    uint m;
    uint n;
    uint k;
};

kernel void supersonic_matmul_rhs_transposed_residual_bf16(
    device const bfloat* lhs [[buffer(0)]],
    device const bfloat* rhs [[buffer(1)]],
    device const bfloat* residual [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant MatmulParams& params [[buffer(4)]],
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
    uint out_idx = (batch * params.m + row) * params.n + col;
    // Match the unfused path's rounding: materialize matmul as BF16 before the residual add.
    out[out_idx] = bfloat(float(bfloat(acc)) + float(residual[out_idx]));
}
)MATMULRESIDUAL";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:342
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile matmul residual library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_residual_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:343
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load matmul residual function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:344
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create matmul residual pipeline"
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

id<MTLComputePipelineState> qwen_mlp_gate_up_pipeline_bf16(NSError** error_out) {
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
                                                   code:371
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QWENMLPGATEUP(
#include <metal_stdlib>
using namespace metal;

struct QwenMlpParams {
    uint hidden_dim;
    uint intermediate_dim;
};

kernel void supersonic_qwen_mlp_gate_up_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* gate_weight [[buffer(1)]],
    device const bfloat* up_weight [[buffer(2)]],
    device bfloat* gate_out [[buffer(3)]],
    device bfloat* up_out [[buffer(4)]],
    constant QwenMlpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.intermediate_dim) {
        return;
    }
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint weight_base = gid * params.hidden_dim;
    for (uint kk = 0; kk < params.hidden_dim; ++kk) {
        float x = float(input[kk]);
        gate_acc += x * float(gate_weight[weight_base + kk]);
        up_acc += x * float(up_weight[weight_base + kk]);
    }
    gate_out[gid] = bfloat(gate_acc);
    up_out[gid] = bfloat(up_acc);
}
)QWENMLPGATEUP";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:372
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen MLP gate/up library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_mlp_gate_up_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:373
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen MLP gate/up function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:374
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create Qwen MLP gate/up pipeline"
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

id<MTLComputePipelineState> qwen_mlp_gate_up_swiglu_pipeline_bf16(NSError** error_out) {
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
                                                   code:421
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QMLPSWIGLU(
#include <metal_stdlib>
using namespace metal;

struct QwenMlpParams {
    uint hidden_dim;
    uint intermediate_dim;
};

kernel void supersonic_qwen_mlp_gate_up_swiglu_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* gate_weight [[buffer(1)]],
    device const bfloat* up_weight [[buffer(2)]],
    device bfloat* mlp_out [[buffer(3)]],
    constant QwenMlpParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.intermediate_dim) {
        return;
    }
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint weight_base = gid * params.hidden_dim;
    for (uint kk = 0; kk < params.hidden_dim; ++kk) {
        float x = float(input[kk]);
        gate_acc += x * float(gate_weight[weight_base + kk]);
        up_acc += x * float(up_weight[weight_base + kk]);
    }

    // Match the existing two-kernel path: gate/up projections are rounded to
    // BF16 before SiLU, then the SiLU product is rounded to BF16.
    bfloat gate_bf = bfloat(gate_acc);
    bfloat up_bf = bfloat(up_acc);
    float gate_v = float(gate_bf);
    float sig = 1.0f / (1.0f + exp(-gate_v));
    mlp_out[gid] = bfloat(gate_v * sig * float(up_bf));
}
)QMLPSWIGLU";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:422
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen MLP gate/up/swiglu library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_mlp_gate_up_swiglu_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:423
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen MLP gate/up/swiglu function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:424
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create Qwen MLP gate/up/swiglu pipeline"
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

id<MTLComputePipelineState> qwen_mlp_down_residual_pipeline_bf16(NSError** error_out) {
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
                                                   code:381
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QWENMLPDOWN(
#include <metal_stdlib>
using namespace metal;

struct QwenMlpParams {
    uint hidden_dim;
    uint intermediate_dim;
};

kernel void supersonic_qwen_mlp_down_residual_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device const bfloat* down_weight [[buffer(2)]],
    device const bfloat* residual [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant QwenMlpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.hidden_dim) {
        return;
    }
    float acc = 0.0f;
    uint weight_base = gid * params.intermediate_dim;
    for (uint ii = 0; ii < params.intermediate_dim; ++ii) {
        float gate_v = float(gate[ii]);
        float sig = 1.0f / (1.0f + exp(-gate_v));
        bfloat mlp_v = bfloat(gate_v * sig * float(up[ii]));
        acc += float(mlp_v) * float(down_weight[weight_base + ii]);
    }
    out[gid] = bfloat(float(bfloat(acc)) + float(residual[gid]));
}
)QWENMLPDOWN";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:382
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen MLP down library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_mlp_down_residual_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:383
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen MLP down function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:384
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create Qwen MLP down pipeline"
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

id<MTLComputePipelineState> qwen_linear_out_residual_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static __strong NSMutableDictionary<NSString*, NSError*>* errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (pipelines == nil) {
        pipelines = [[NSMutableDictionary alloc] init];
        errors = [[NSMutableDictionary alloc] init];
    }
    id<MTLComputePipelineState> cached = pipelines[function_name];
    if (cached != nil) {
        return cached;
    }
    NSError* cached_error = errors[function_name];
    if (cached_error != nil) {
        if (error_out != nullptr) {
            *error_out = cached_error;
        }
        return nil;
    }

    NSError* build_error = nil;
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                               code:385
                                           userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
        } else {
            static const char* kSource = R"QWENLINEAROUT(
#include <metal_stdlib>
using namespace metal;

struct QwenLinearOutParams {
    uint hidden_dim;
    uint num_rows;
    uint row_dim;
    float eps;
};

kernel void supersonic_qwen_linear_out_residual_f32_bf16(
    device const float* attn [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* norm_weight [[buffer(2)]],
    device const bfloat* out_proj [[buffer(3)]],
    device const bfloat* residual [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant QwenLinearOutParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.hidden_dim) {
        return;
    }
    float acc = 0.0f;
    for (uint row = 0; row < params.num_rows; ++row) {
        uint row_base = row * params.row_dim;
        float mean_sq = 0.0f;
        for (uint col = 0; col < params.row_dim; ++col) {
            float value = float(bfloat(attn[row_base + col]));
            mean_sq = fma(value, value, mean_sq);
        }
        float inv_rms = rsqrt((mean_sq / float(params.row_dim)) + params.eps);
        for (uint col = 0; col < params.row_dim; ++col) {
            uint idx = row_base + col;
            float gate_v = float(gate[idx]);
            float sig = 1.0f / (1.0f + exp(-gate_v));
            float hidden_v = float(bfloat(attn[idx]));
            bfloat gated = bfloat(hidden_v * inv_rms * float(norm_weight[col]) * (gate_v * sig));
            acc += float(gated) * float(out_proj[gid * (params.num_rows * params.row_dim) + idx]);
        }
    }
    out[gid] = bfloat(float(bfloat(acc)) + float(residual[gid]));
}

kernel void supersonic_qwen_linear_out_residual_bf16_bf16(
    device const bfloat* attn [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* norm_weight [[buffer(2)]],
    device const bfloat* out_proj [[buffer(3)]],
    device const bfloat* residual [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant QwenLinearOutParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.hidden_dim) {
        return;
    }
    float acc = 0.0f;
    for (uint row = 0; row < params.num_rows; ++row) {
        uint row_base = row * params.row_dim;
        float mean_sq = 0.0f;
        for (uint col = 0; col < params.row_dim; ++col) {
            float value = float(attn[row_base + col]);
            mean_sq = fma(value, value, mean_sq);
        }
        float inv_rms = rsqrt((mean_sq / float(params.row_dim)) + params.eps);
        for (uint col = 0; col < params.row_dim; ++col) {
            uint idx = row_base + col;
            float gate_v = float(gate[idx]);
            float sig = 1.0f / (1.0f + exp(-gate_v));
            bfloat gated =
                bfloat(float(attn[idx]) * inv_rms * float(norm_weight[col]) * (gate_v * sig));
            acc += float(gated) * float(out_proj[gid * (params.num_rows * params.row_dim) + idx]);
        }
    }
    out[gid] = bfloat(float(bfloat(acc)) + float(residual[gid]));
}
)QWENLINEAROUT";

            NSString* source = [NSString stringWithUTF8String:kSource];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            configure_precise_math(options);
            NSError* library_error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&library_error];
            if (library == nil || library_error != nil) {
                build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                   code:386
                                                               userInfo:@{
                                                                   NSLocalizedDescriptionKey :
                                                                       @"Failed to compile Qwen linear out library"
                                                               }];
            } else {
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                       code:387
                                                   userInfo:@{
                                                       NSLocalizedDescriptionKey :
                                                           @"Failed to load Qwen linear out function"
                                                   }];
                } else {
                    NSError* pipeline_error = nil;
                    pipeline = [device newComputePipelineStateWithFunction:function
                                                                     error:&pipeline_error];
                    if (pipeline == nil || pipeline_error != nil) {
                        build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                             code:388
                                                                         userInfo:@{
                                                                             NSLocalizedDescriptionKey :
                                                                                 @"Failed to create Qwen linear out pipeline"
                                                                         }];
                    }
                }
            }
        }
    }

    if (pipeline != nil) {
        pipelines[function_name] = pipeline;
        return pipeline;
    }
    if (build_error != nil) {
        errors[function_name] = build_error;
    }
    if (error_out != nullptr) {
        *error_out = build_error;
    }
    return nil;
}

id<MTLComputePipelineState> qwen_full_projection_pipeline_bf16(NSError** error_out) {
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
                                                   code:401
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QWENFULLPROJ(
#include <metal_stdlib>
using namespace metal;

struct QwenFullProjectionParams {
    uint hidden_dim;
    uint q_proj_dim;
    uint kv_dim;
    uint total_cols;
};

kernel void supersonic_qwen_full_projections_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* q_weight [[buffer(1)]],
    device const bfloat* k_weight [[buffer(2)]],
    device const bfloat* v_weight [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    device bfloat* v_out [[buffer(6)]],
    constant QwenFullProjectionParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_cols) {
        return;
    }

    device const bfloat* weight = q_weight;
    device bfloat* out = q_out;
    uint local_col = gid;
    if (gid >= params.q_proj_dim) {
        uint kv_col = gid - params.q_proj_dim;
        if (kv_col < params.kv_dim) {
            weight = k_weight;
            out = k_out;
            local_col = kv_col;
        } else {
            weight = v_weight;
            out = v_out;
            local_col = kv_col - params.kv_dim;
        }
    }

    float acc = 0.0f;
    uint weight_base = local_col * params.hidden_dim;
    for (uint kk = 0; kk < params.hidden_dim; ++kk) {
        acc += float(input[kk]) * float(weight[weight_base + kk]);
    }
    out[local_col] = bfloat(acc);
}
)QWENFULLPROJ";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:402
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen full projection library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_full_projections_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:403
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen full projection function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:404
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create Qwen full projection pipeline"
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

id<MTLComputePipelineState> qwen_linear_projection_pipeline(NSError** error_out) {
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
                                                   code:320
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QWENLINEAR(
#include <metal_stdlib>
using namespace metal;

struct QwenLinearProjectionParams {
    uint hidden_dim;
    uint qkv_dim;
    uint val_dim;
    uint num_value_heads;
    uint total_cols;
};

kernel void supersonic_qwen_linear_projections_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* qkv_weight [[buffer(1)]],
    device const bfloat* z_weight [[buffer(2)]],
    device const bfloat* a_weight [[buffer(3)]],
    device const bfloat* b_weight [[buffer(4)]],
    device bfloat* qkv_out [[buffer(5)]],
    device bfloat* z_out [[buffer(6)]],
    device bfloat* a_out [[buffer(7)]],
    device bfloat* b_out [[buffer(8)]],
    constant QwenLinearProjectionParams& params [[buffer(9)]],
    uint col [[thread_position_in_grid]]
) {
    if (col >= params.total_cols) {
        return;
    }

    device const bfloat* weight = qkv_weight;
    device bfloat* out = qkv_out;
    uint local_col = col;
    if (local_col >= params.qkv_dim) {
        local_col -= params.qkv_dim;
        if (local_col < params.val_dim) {
            weight = z_weight;
            out = z_out;
        } else {
            local_col -= params.val_dim;
            if (local_col < params.num_value_heads) {
                weight = a_weight;
                out = a_out;
            } else {
                local_col -= params.num_value_heads;
                weight = b_weight;
                out = b_out;
            }
        }
    }

    float acc = 0.0f;
    uint weight_base = local_col * params.hidden_dim;
    for (uint kk = 0; kk < params.hidden_dim; ++kk) {
        acc += float(input[kk]) * float(weight[weight_base + kk]);
    }
    out[local_col] = bfloat(acc);
}
)QWENLINEAR";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:321
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen linear projection library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_linear_projections_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:322
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen linear projection function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:323
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create Qwen linear projection pipeline"
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

id<MTLComputePipelineState> lm_head_argmax_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static __strong NSMutableDictionary<NSString*, NSError*>* errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (pipelines == nil) {
        pipelines = [[NSMutableDictionary alloc] init];
        errors = [[NSMutableDictionary alloc] init];
    }

    id<MTLComputePipelineState> cached = pipelines[function_name];
    if (cached != nil) {
        return cached;
    }
    NSError* cached_error = errors[function_name];
    if (cached_error != nil) {
        if (error_out != nullptr) {
            *error_out = cached_error;
        }
        return nil;
    }

    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            NSError* err = [NSError errorWithDomain:@"SuperSonicMetal"
                                               code:260
                                           userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            errors[function_name] = err;
            if (error_out != nullptr) {
                *error_out = err;
            }
            return nil;
        }

        static const char* kSource = R"LMARGMAX(
#include <metal_stdlib>
using namespace metal;

struct LmHeadArgmaxParams {
    uint in_dim;
    uint vocab_size;
    uint block_size;
    uint partial_count;
};

inline bool supersonic_argmax_better(float value, uint index, float best_value, uint best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

kernel void supersonic_lm_head_argmax_stage1_bf16(
    device const bfloat* hidden [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* partial_values [[buffer(2)]],
    device uint* partial_indices [[buffer(3)]],
    constant LmHeadArgmaxParams& params [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float values[512];
    threadgroup uint indices[512];

    float best_value = -INFINITY;
    uint best_index = 0;
    uint row = group_id * params.block_size + tid;
    if (tid < params.block_size && row < params.vocab_size) {
        float acc = 0.0f;
        uint weight_base = row * params.in_dim;
        for (uint kk = 0; kk < params.in_dim; ++kk) {
            acc += float(hidden[kk]) * float(weight[weight_base + kk]);
        }
        best_value = acc;
        best_index = row;
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = values[tid + stride];
            uint other_index = indices[tid + stride];
            if (supersonic_argmax_better(other_value, other_index, values[tid], indices[tid])) {
                values[tid] = other_value;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_values[group_id] = values[0];
        partial_indices[group_id] = indices[0];
    }
}

kernel void supersonic_argmax_stage1_bf16(
    device const bfloat* logits [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device uint* partial_indices [[buffer(2)]],
    constant LmHeadArgmaxParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float values[512];
    threadgroup uint indices[512];

    float best_value = -INFINITY;
    uint best_index = 0;
    uint row = group_id * params.block_size + tid;
    if (tid < params.block_size && row < params.vocab_size) {
        best_value = float(logits[row]);
        best_index = row;
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = values[tid + stride];
            uint other_index = indices[tid + stride];
            if (supersonic_argmax_better(other_value, other_index, values[tid], indices[tid])) {
                values[tid] = other_value;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_values[group_id] = values[0];
        partial_indices[group_id] = indices[0];
    }
}

kernel void supersonic_lm_head_argmax_stage2(
    device const float* partial_values [[buffer(0)]],
    device const uint* partial_indices [[buffer(1)]],
    device uint* out_index [[buffer(2)]],
    constant LmHeadArgmaxParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float values[512];
    threadgroup uint indices[512];

    float best_value = -INFINITY;
    uint best_index = 0;
    for (uint idx = tid; idx < params.partial_count; idx += params.block_size) {
        float value = partial_values[idx];
        uint index = partial_indices[idx];
        if (supersonic_argmax_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = values[tid + stride];
            uint other_index = indices[tid + stride];
            if (supersonic_argmax_better(other_value, other_index, values[tid], indices[tid])) {
                values[tid] = other_value;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_index[0] = indices[0];
    }
}
)LMARGMAX";

        NSString* source = [NSString stringWithUTF8String:kSource];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        configure_precise_math(options);
        NSError* library_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:options
                                                        error:&library_error];
        if (library == nil || library_error != nil) {
            NSError* err = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                 code:261
                                                             userInfo:@{
                                                                 NSLocalizedDescriptionKey :
                                                                     @"Failed to compile lm-head argmax library"
                                                             }];
            errors[function_name] = err;
            if (error_out != nullptr) {
                *error_out = err;
            }
            return nil;
        }

        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            NSError* err = [NSError errorWithDomain:@"SuperSonicMetal"
                                               code:262
                                           userInfo:@{
                                               NSLocalizedDescriptionKey :
                                                   @"Failed to load lm-head argmax function"
                                           }];
            errors[function_name] = err;
            if (error_out != nullptr) {
                *error_out = err;
            }
            return nil;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                     error:&pipeline_error];
        if (pipeline == nil || pipeline_error != nil) {
            NSError* err = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                 code:263
                                                             userInfo:@{
                                                                 NSLocalizedDescriptionKey :
                                                                     @"Failed to create lm-head argmax pipeline"
                                                             }];
            errors[function_name] = err;
            if (error_out != nullptr) {
                *error_out = err;
            }
            return nil;
        }

        pipelines[function_name] = pipeline;
        return pipeline;
    }
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
    uint kv_stride;
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
        uint key_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
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
        uint key_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
        float dot = 0.0f;
        for (uint kk = 0; kk < params.head_dim; ++kk) {
            dot += float(query[query_base + kk]) * float(key[key_base + kk]);
        }
        float weight = exp((dot * params.scale) - max_score);
        denom += weight;
        uint value_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
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

id<MTLComputePipelineState> full_attention_decode_pipeline_bf16_f32(NSError** error_out) {
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
                                                   code:501
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"FATTNDECODE(
#include <metal_stdlib>
using namespace metal;

struct FullAttentionDecodeParams {
    uint q_heads;
    uint kv_heads;
    uint kv_len;
    uint kv_stride;
    uint head_dim;
    float scale;
};

inline void reduce_sum_256(threadgroup float* scratch, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) { scratch[tid] += scratch[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { scratch[tid] += scratch[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { scratch[tid] += scratch[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { scratch[tid] += scratch[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) { scratch[tid] += scratch[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) { scratch[tid] += scratch[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { scratch[tid] += scratch[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 1) { scratch[tid] += scratch[tid + 1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void supersonic_full_attention_decode_bf16_f32(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* key [[buffer(1)]],
    device const bfloat* value [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant FullAttentionDecodeParams& params [[buffer(4)]],
    uint q_head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[256];
    threadgroup float max_score;
    threadgroup float denom;
    threadgroup float weight;

    if (q_head >= params.q_heads) {
        return;
    }

    uint num_kv_groups = params.q_heads / params.kv_heads;
    uint kv_head = q_head / num_kv_groups;
    uint query_base = q_head * params.head_dim;

    if (tid == 0) {
        max_score = -INFINITY;
        denom = 0.0f;
        weight = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_pos = 0; kv_pos < params.kv_len; ++kv_pos) {
        uint key_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
        scratch[tid] = tid < params.head_dim
            ? float(query[query_base + tid]) * float(key[key_base + tid])
            : 0.0f;
        reduce_sum_256(scratch, tid);
        if (tid == 0) {
            max_score = max(max_score, scratch[0] * params.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint kv_pos = 0; kv_pos < params.kv_len; ++kv_pos) {
        uint key_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
        scratch[tid] = tid < params.head_dim
            ? float(query[query_base + tid]) * float(key[key_base + tid])
            : 0.0f;
        reduce_sum_256(scratch, tid);
        if (tid == 0) {
            denom += exp((scratch[0] * params.scale) - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float numer = 0.0f;
    for (uint kv_pos = 0; kv_pos < params.kv_len; ++kv_pos) {
        uint key_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
        scratch[tid] = tid < params.head_dim
            ? float(query[query_base + tid]) * float(key[key_base + tid])
            : 0.0f;
        reduce_sum_256(scratch, tid);
        if (tid == 0) {
            weight = exp((scratch[0] * params.scale) - max_score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < params.head_dim) {
            uint value_base = (kv_head * params.kv_stride + kv_pos) * params.head_dim;
            numer += weight * float(value[value_base + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < params.head_dim) {
        out[q_head * params.head_dim + tid] = numer / denom;
    }
}
)FATTNDECODE";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:502
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile full-attention decode library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_full_attention_decode_bf16_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:503
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load full-attention decode function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:504
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create full-attention decode pipeline"
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

id<MTLComputePipelineState> embedding_lookup_pipeline_bf16(NSError** error_out) {
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
                                                   code:238
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"EMB(
#include <metal_stdlib>
using namespace metal;

struct EmbeddingLookupParams {
    uint token_count;
    uint vocab_size;
    uint hidden_size;
    uint total_elems;
};

kernel void supersonic_embedding_lookup_bf16(
    device const bfloat* embeddings [[buffer(0)]],
    device const uint* indexes [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant EmbeddingLookupParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint token_idx = gid / params.hidden_size;
    uint col = gid - token_idx * params.hidden_size;
    uint vocab_idx = indexes[token_idx];
    if (vocab_idx >= params.vocab_size) {
        out[gid] = bfloat(0.0f);
        return;
    }
    out[gid] = embeddings[vocab_idx * params.hidden_size + col];
}
)EMB";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:239
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile embedding lookup library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_embedding_lookup_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:240
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load embedding lookup function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:241
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create embedding lookup pipeline"
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
    uint block_size;
};

kernel void supersonic_rms_norm_rows_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = float(input[row_base + col]);
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float scale = float(weight[col]) + (params.add_unit_offset != 0 ? 1.0f : 0.0f);
        out[row_base + col] = bfloat(float(input[row_base + col]) * inv_rms * scale);
    }
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

id<MTLComputePipelineState> rms_norm_rope_pipeline_bf16(NSError** error_out) {
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
                                                   code:134
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"RMSROPE(
#include <metal_stdlib>
using namespace metal;

struct RmsNormRopeParams {
    uint n_rows;
    uint n_cols;
    uint rotary_dim;
    uint half_rot;
    uint pos_offset;
    float eps;
    uint block_size;
};

kernel void supersonic_rms_norm_rope_rows_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* cos_table [[buffer(2)]],
    device const bfloat* sin_table [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant RmsNormRopeParams& params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = float(input[row_base + col]);
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    uint table_base = params.pos_offset * params.half_rot;
    for (uint col = tid; col < params.half_rot; col += params.block_size) {
        float x0 = float(input[row_base + col]) * inv_rms * (float(weight[col]) + 1.0f);
        float x1 = float(input[row_base + col + params.half_rot]) * inv_rms *
                   (float(weight[col + params.half_rot]) + 1.0f);
        float c = float(cos_table[table_base + col]);
        float s = float(sin_table[table_base + col]);
        out[row_base + col] = bfloat(x0 * c - x1 * s);
        out[row_base + col + params.half_rot] = bfloat(x1 * c + x0 * s);
    }
    for (uint col = params.rotary_dim + tid; col < params.n_cols; col += params.block_size) {
        out[row_base + col] =
            bfloat(float(input[row_base + col]) * inv_rms * (float(weight[col]) + 1.0f));
    }
}
)RMSROPE";

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
                                                                           @"Failed to compile RMSNorm RoPE library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_rms_norm_rope_rows_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:136
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load RMSNorm RoPE function"
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
                                                                                     @"Failed to create RMSNorm RoPE pipeline"
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

id<MTLComputePipelineState> rms_norm_pipeline_f32(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static __strong NSMutableDictionary<NSString*, NSError*>* errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (pipelines == nil) {
        pipelines = [[NSMutableDictionary alloc] init];
        errors = [[NSMutableDictionary alloc] init];
    }

    id<MTLComputePipelineState> cached = pipelines[function_name];
    if (cached != nil) {
        return cached;
    }
    NSError* cached_error = errors[function_name];
    if (cached_error != nil) {
        if (error_out != nullptr) {
            *error_out = cached_error;
        }
        return nil;
    }

    NSError* build_error = nil;
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                              code:340
                                          userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
        } else {
            static const char* kSource = R"RMSF32(
#include <metal_stdlib>
using namespace metal;

struct RmsNormParams {
    uint n_rows;
    uint n_cols;
    float eps;
    uint add_unit_offset;
    uint block_size;
};

kernel void supersonic_rms_norm_rows_f32_weight_bf16(
    device const float* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = input[row_base + col];
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float scale = float(weight[col]) + (params.add_unit_offset != 0 ? 1.0f : 0.0f);
        out[row_base + col] = input[row_base + col] * inv_rms * scale;
    }
}

kernel void supersonic_rms_norm_rows_f32_weight_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = input[row_base + col];
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float scale = weight[col] + (params.add_unit_offset != 0 ? 1.0f : 0.0f);
        out[row_base + col] = input[row_base + col] * inv_rms * scale;
    }
}
)RMSF32";

            NSString* source = [NSString stringWithUTF8String:kSource];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            configure_precise_math(options);
            NSError* library_error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&library_error];
            if (library == nil || library_error != nil) {
                build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                   code:341
                                                               userInfo:@{
                                                                   NSLocalizedDescriptionKey :
                                                                       @"Failed to compile F32 RMSNorm library"
                                                               }];
            } else {
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                       code:342
                                                   userInfo:@{
                                                       NSLocalizedDescriptionKey :
                                                           @"Failed to load F32 RMSNorm function"
                                                   }];
                } else {
                    NSError* pipeline_error = nil;
                    pipeline = [device newComputePipelineStateWithFunction:function
                                                                     error:&pipeline_error];
                    if (pipeline == nil || pipeline_error != nil) {
                        build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                             code:343
                                                                         userInfo:@{
                                                                             NSLocalizedDescriptionKey :
                                                                                 @"Failed to create F32 RMSNorm pipeline"
                                                                         }];
                    }
                }
            }
        }
    }

    if (pipeline != nil) {
        pipelines[function_name] = pipeline;
        return pipeline;
    }
    if (build_error != nil) {
        errors[function_name] = build_error;
    }
    if (error_out != nullptr) {
        *error_out = build_error;
    }
    return nil;
}

id<MTLComputePipelineState> rope_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static __strong NSMutableDictionary<NSString*, NSError*>* errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (pipelines == nil) {
        pipelines = [[NSMutableDictionary alloc] init];
        errors = [[NSMutableDictionary alloc] init];
    }

    id<MTLComputePipelineState> cached = pipelines[function_name];
    if (cached != nil) {
        return cached;
    }
    NSError* cached_error = errors[function_name];
    if (cached_error != nil) {
        if (error_out != nullptr) {
            *error_out = cached_error;
        }
        return nil;
    }

    NSError* build_error = nil;
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                              code:350
                                          userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
        } else {
            static const char* kSource = R"ROPE(
#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint seq_len;
    uint num_heads;
    uint head_dim;
    uint rotary_dim;
    uint half_rot;
    uint pos_offset;
    uint total_pairs;
};

kernel void supersonic_apply_rope_prefill_bf16(
    device bfloat* data [[buffer(0)]],
    device const bfloat* cos_table [[buffer(1)]],
    device const bfloat* sin_table [[buffer(2)]],
    constant RopeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_pairs) {
        return;
    }
    uint i = gid % params.half_rot;
    uint tmp = gid / params.half_rot;
    uint head = tmp % params.num_heads;
    uint pos = tmp / params.num_heads;
    uint base = (pos * params.num_heads + head) * params.head_dim;
    uint table_base = (params.pos_offset + pos) * params.half_rot;
    float c = float(cos_table[table_base + i]);
    float s = float(sin_table[table_base + i]);
    float x0 = float(data[base + i]);
    float x1 = float(data[base + i + params.half_rot]);
    data[base + i] = bfloat(x0 * c - x1 * s);
    data[base + i + params.half_rot] = bfloat(x1 * c + x0 * s);
}

kernel void supersonic_apply_rope_prefill_f32(
    device float* data [[buffer(0)]],
    device const bfloat* cos_table [[buffer(1)]],
    device const bfloat* sin_table [[buffer(2)]],
    constant RopeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_pairs) {
        return;
    }
    uint i = gid % params.half_rot;
    uint tmp = gid / params.half_rot;
    uint head = tmp % params.num_heads;
    uint pos = tmp / params.num_heads;
    uint base = (pos * params.num_heads + head) * params.head_dim;
    uint table_base = (params.pos_offset + pos) * params.half_rot;
    float c = float(cos_table[table_base + i]);
    float s = float(sin_table[table_base + i]);
    float x0 = data[base + i];
    float x1 = data[base + i + params.half_rot];
    data[base + i] = x0 * c - x1 * s;
    data[base + i + params.half_rot] = x1 * c + x0 * s;
}
)ROPE";

            NSString* source = [NSString stringWithUTF8String:kSource];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            configure_precise_math(options);
            NSError* library_error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&library_error];
            if (library == nil || library_error != nil) {
                build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                   code:351
                                                               userInfo:@{
                                                                   NSLocalizedDescriptionKey :
                                                                       @"Failed to compile RoPE library"
                                                               }];
            } else {
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                       code:352
                                                   userInfo:@{
                                                       NSLocalizedDescriptionKey :
                                                           @"Failed to load RoPE function"
                                                   }];
                } else {
                    NSError* pipeline_error = nil;
                    pipeline = [device newComputePipelineStateWithFunction:function
                                                                     error:&pipeline_error];
                    if (pipeline == nil || pipeline_error != nil) {
                        build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                             code:353
                                                                         userInfo:@{
                                                                             NSLocalizedDescriptionKey :
                                                                                 @"Failed to create RoPE pipeline"
                                                                         }];
                    }
                }
            }
        }
    }

    if (pipeline != nil) {
        pipelines[function_name] = pipeline;
        return pipeline;
    }
    if (build_error != nil) {
        errors[function_name] = build_error;
    }
    if (error_out != nullptr) {
        *error_out = build_error;
    }
    return nil;
}

id<MTLComputePipelineState> conv_layout_pipeline(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static __strong NSMutableDictionary<NSString*, NSError*>* errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (pipelines == nil) {
        pipelines = [[NSMutableDictionary alloc] init];
        errors = [[NSMutableDictionary alloc] init];
    }

    id<MTLComputePipelineState> cached = pipelines[function_name];
    if (cached != nil) {
        return cached;
    }
    NSError* cached_error = errors[function_name];
    if (cached_error != nil) {
        if (error_out != nullptr) {
            *error_out = cached_error;
        }
        return nil;
    }

    NSError* build_error = nil;
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                              code:360
                                          userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
        } else {
            static const char* kSource = R"CONVLAYOUT(
#include <metal_stdlib>
using namespace metal;

struct TransposePadConvParams {
    uint s;
    uint c;
    uint pad;
    uint stride;
    uint total_dst;
};

struct ExtractConvStateParams {
    uint s;
    uint c;
    uint kern_minus_1;
    uint copy;
    uint start;
    uint dst_start;
    uint total_dst;
};

kernel void supersonic_transpose_pad_conv_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant TransposePadConvParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_dst) {
        return;
    }
    uint ch = gid / params.stride;
    uint pos = gid - ch * params.stride;
    if (pos < params.pad) {
        dst[gid] = bfloat(0.0f);
        return;
    }
    uint row = pos - params.pad;
    dst[gid] = src[row * params.c + ch];
}

kernel void supersonic_transpose_pad_conv_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant TransposePadConvParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_dst) {
        return;
    }
    uint ch = gid / params.stride;
    uint pos = gid - ch * params.stride;
    if (pos < params.pad) {
        dst[gid] = 0.0f;
        return;
    }
    uint row = pos - params.pad;
    dst[gid] = src[row * params.c + ch];
}

kernel void supersonic_extract_conv_state_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst [[buffer(1)]],
    constant ExtractConvStateParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_dst) {
        return;
    }
    uint ch = gid / params.kern_minus_1;
    uint i = gid - ch * params.kern_minus_1;
    if (i < params.dst_start) {
        dst[gid] = bfloat(0.0f);
        return;
    }
    uint src_row = params.start + (i - params.dst_start);
    dst[gid] = src[src_row * params.c + ch];
}

kernel void supersonic_extract_conv_state_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant ExtractConvStateParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_dst) {
        return;
    }
    uint ch = gid / params.kern_minus_1;
    uint i = gid - ch * params.kern_minus_1;
    if (i < params.dst_start) {
        dst[gid] = 0.0f;
        return;
    }
    uint src_row = params.start + (i - params.dst_start);
    dst[gid] = src[src_row * params.c + ch];
}
)CONVLAYOUT";

            NSString* source = [NSString stringWithUTF8String:kSource];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            configure_precise_math(options);
            NSError* library_error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&library_error];
            if (library == nil || library_error != nil) {
                build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                   code:361
                                                               userInfo:@{
                                                                   NSLocalizedDescriptionKey :
                                                                       @"Failed to compile conv layout library"
                                                               }];
            } else {
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                       code:362
                                                   userInfo:@{
                                                       NSLocalizedDescriptionKey :
                                                           @"Failed to load conv layout function"
                                                   }];
                } else {
                    NSError* pipeline_error = nil;
                    pipeline = [device newComputePipelineStateWithFunction:function
                                                                     error:&pipeline_error];
                    if (pipeline == nil || pipeline_error != nil) {
                        build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                             code:363
                                                                         userInfo:@{
                                                                             NSLocalizedDescriptionKey :
                                                                                 @"Failed to create conv layout pipeline"
                                                                         }];
                    }
                }
            }
        }
    }

    if (pipeline != nil) {
        pipelines[function_name] = pipeline;
        return pipeline;
    }
    if (build_error != nil) {
        errors[function_name] = build_error;
    }
    if (error_out != nullptr) {
        *error_out = build_error;
    }
    return nil;
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
    uint block_size;
};

kernel void supersonic_rms_norm_gated_bf16(
    device const bfloat* hidden [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant RmsNormGatedParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = float(hidden[row_base + col]);
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float gate_value = float(gate[row_base + col]);
        float sig = 1.0f / (1.0f + exp(-gate_value));
        float silu_gate = gate_value * sig;
        float value = float(hidden[row_base + col]) * inv_rms * float(weight[col]) * silu_gate;
        out[row_base + col] = bfloat(value);
    }
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

id<MTLComputePipelineState> rms_norm_gated_pipeline_f32(NSString* function_name, NSError** error_out) {
    static std::mutex mutex;
    static __strong NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static __strong NSMutableDictionary<NSString*, NSError*>* errors = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (pipelines == nil) {
        pipelines = [[NSMutableDictionary alloc] init];
        errors = [[NSMutableDictionary alloc] init];
    }

    id<MTLComputePipelineState> cached = pipelines[function_name];
    if (cached != nil) {
        return cached;
    }
    NSError* cached_error = errors[function_name];
    if (cached_error != nil) {
        if (error_out != nullptr) {
            *error_out = cached_error;
        }
        return nil;
    }

    NSError* build_error = nil;
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                              code:320
                                          userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
        } else {
            static const char* kSource = R"RMSGF32(
#include <metal_stdlib>
using namespace metal;

struct RmsNormGatedParams {
    uint n_rows;
    uint n_cols;
    float eps;
    uint block_size;
};

kernel void supersonic_rms_norm_gated_f32_weight_bf16(
    device const float* hidden [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant RmsNormGatedParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = hidden[row_base + col];
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float gate_value = gate[row_base + col];
        float silu_gate = gate_value / (1.0f + exp(-gate_value));
        out[row_base + col] = hidden[row_base + col] * inv_rms * float(weight[col]) * silu_gate;
    }
}

kernel void supersonic_rms_norm_gated_f32_weight_f32(
    device const float* hidden [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant RmsNormGatedParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }

    threadgroup float scratch[256];
    uint row_base = row * params.n_cols;
    float mean_sq = 0.0f;
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float value = hidden[row_base + col];
        mean_sq = fma(value, value, mean_sq);
    }
    scratch[tid] = mean_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((scratch[0] / float(params.n_cols)) + params.eps);
    for (uint col = tid; col < params.n_cols; col += params.block_size) {
        float gate_value = gate[row_base + col];
        float silu_gate = gate_value / (1.0f + exp(-gate_value));
        out[row_base + col] = hidden[row_base + col] * inv_rms * weight[col] * silu_gate;
    }
}
)RMSGF32";

            NSString* source = [NSString stringWithUTF8String:kSource];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            configure_precise_math(options);
            NSError* library_error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&library_error];
            if (library == nil || library_error != nil) {
                build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                   code:321
                                                               userInfo:@{
                                                                   NSLocalizedDescriptionKey :
                                                                       @"Failed to compile F32 gated RMSNorm library"
                                                               }];
            } else {
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                       code:322
                                                   userInfo:@{
                                                       NSLocalizedDescriptionKey :
                                                           @"Failed to load F32 gated RMSNorm function"
                                                   }];
                } else {
                    NSError* pipeline_error = nil;
                    pipeline = [device newComputePipelineStateWithFunction:function
                                                                     error:&pipeline_error];
                    if (pipeline == nil || pipeline_error != nil) {
                        build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                             code:323
                                                                         userInfo:@{
                                                                             NSLocalizedDescriptionKey :
                                                                                 @"Failed to create F32 gated RMSNorm pipeline"
                                                                         }];
                    }
                }
            }
        }
    }

    if (pipeline != nil) {
        pipelines[function_name] = pipeline;
        return pipeline;
    }
    if (build_error != nil) {
        errors[function_name] = build_error;
    }
    if (error_out != nullptr) {
        *error_out = build_error;
    }
    return nil;
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

id<MTLComputePipelineState> full_attention_gate_pipeline(NSError** error_out) {
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
                                                   code:204
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"FGATE(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
    uint total_elems;
};

kernel void supersonic_full_attention_gate_bf16(
    device const float* attn [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    bfloat attn_bf = bfloat(attn[gid]);
    float gv = float(gate[gid]);
    float sig = 1.0f / (1.0f + exp(-gv));
    out[gid] = bfloat(float(attn_bf) * sig);
}
)FGATE";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:205
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile full-attention gate library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_full_attention_gate_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:206
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load full-attention gate function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:207
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create full-attention gate pipeline"
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

id<MTLComputePipelineState> linear_decode_apply_parts_pipeline(NSError** error_out) {
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
                                                   code:181
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"LDA(
#include <metal_stdlib>
using namespace metal;

struct LinearDecodeApplyParams {
    uint num_v_heads;
    uint num_k_heads;
    uint head_repeat;
    uint k_head_dim;
    uint v_head_dim;
    uint value_dim;
    uint state_dim;
    uint total_threads;
};

static inline float softplus_stable(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

kernel void supersonic_linear_decode_apply_parts_f32(
    device const float* q_scaled [[buffer(0)]],
    device const float* k_normed [[buffer(1)]],
    device const float* v_linear [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const bfloat* dt_bias [[buffer(5)]],
    device const bfloat* a_log_exp [[buffer(6)]],
    device const float* initial_state [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant LinearDecodeApplyParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }
    uint v_head = gid / params.v_head_dim;
    uint vv = gid - v_head * params.v_head_dim;
    uint k_head = v_head / params.head_repeat;

    uint q_base = k_head * params.k_head_dim;
    uint k_base = k_head * params.k_head_dim;
    uint v_base = v_head * params.v_head_dim;
    float beta = 1.0f / (1.0f + exp(-float(b[v_head])));
    float g_exp =
        exp(-softplus_stable(float(a[v_head]) + float(dt_bias[v_head])) * float(a_log_exp[v_head]));

    uint state_head_base = (v_head * params.k_head_dim) * params.v_head_dim + vv;
    uint state_out_base = params.value_dim + (v_head * params.k_head_dim) * params.v_head_dim + vv;
    float kv_mem = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        float state = initial_state[state_head_base + kk * params.v_head_dim] * g_exp;
        kv_mem = fma(state, k_normed[k_base + kk], kv_mem);
        out[state_out_base + kk * params.v_head_dim] = state;
    }

    float delta = (v_linear[v_base + vv] - kv_mem) * beta;
    float out_value = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        uint state_idx = state_out_base + kk * params.v_head_dim;
        float state = fma(k_normed[k_base + kk], delta, out[state_idx]);
        out[state_idx] = state;
        out_value = fma(state, q_scaled[q_base + kk], out_value);
    }
    out[v_base + vv] = out_value;
}
)LDA";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:182
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile linear-decode-apply-parts library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_linear_decode_apply_parts_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:183
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load linear-decode-apply-parts function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                code:184
                                                                            userInfo:@{
                                                                                NSLocalizedDescriptionKey :
                                                                                    @"Failed to create linear-decode-apply-parts pipeline"
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

id<MTLComputePipelineState> qwen_linear_prep_pipeline(NSError** error_out) {
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
                                                   code:186
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"LDPREP(
#include <metal_stdlib>
using namespace metal;

struct QwenLinearPrepParams {
    uint key_dim;
    uint val_dim;
    uint num_key_heads;
    uint key_head_dim;
    uint total_threads;
    float eps;
    float q_scale;
};

kernel void supersonic_qwen_linear_prep_bf16_f32(
    device const bfloat* conv_pack [[buffer(0)]],
    device bfloat* q_bf16 [[buffer(1)]],
    device bfloat* k_bf16 [[buffer(2)]],
    device bfloat* v_bf16 [[buffer(3)]],
    device float* q_f32 [[buffer(4)]],
    device float* k_f32 [[buffer(5)]],
    device float* v_f32 [[buffer(6)]],
    device float* q_normed [[buffer(7)]],
    device float* q_scaled [[buffer(8)]],
    device float* k_normed [[buffer(9)]],
    constant QwenLinearPrepParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }

    if (gid < params.key_dim) {
        uint head = gid / params.key_head_dim;
        uint head_offset = head * params.key_head_dim;
        float q_val = float(conv_pack[gid]);
        float k_val = float(conv_pack[params.key_dim + gid]);
        q_bf16[gid] = bfloat(q_val);
        k_bf16[gid] = bfloat(k_val);
        q_f32[gid] = q_val;
        k_f32[gid] = k_val;

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint i = 0; i < params.key_head_dim; ++i) {
            float qh = float(conv_pack[head_offset + i]);
            float kh = float(conv_pack[params.key_dim + head_offset + i]);
            q_sum = fma(qh, qh, q_sum);
            k_sum = fma(kh, kh, k_sum);
        }
        float q_norm = q_val * rsqrt(q_sum + params.eps);
        float k_norm = k_val * rsqrt(k_sum + params.eps);
        q_normed[gid] = q_norm;
        q_scaled[gid] = q_norm * params.q_scale;
        k_normed[gid] = k_norm;
    }

    if (gid < params.val_dim) {
        float v_val = float(conv_pack[params.key_dim * 2 + gid]);
        v_bf16[gid] = bfloat(v_val);
        v_f32[gid] = v_val;
    }
}
)LDPREP";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:187
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen linear prep library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_linear_prep_bf16_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:188
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen linear prep function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                code:189
                                                                            userInfo:@{
                                                                                NSLocalizedDescriptionKey :
                                                                                    @"Failed to create Qwen linear prep pipeline"
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

id<MTLComputePipelineState> qwen_linear_prep_decode_apply_pipeline(NSError** error_out) {
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
                                                   code:580
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"LDPREPAPPLY(
#include <metal_stdlib>
using namespace metal;

struct QwenLinearPrepDecodeApplyParams {
    uint num_v_heads;
    uint num_k_heads;
    uint head_repeat;
    uint k_head_dim;
    uint v_head_dim;
    uint key_dim;
    uint value_dim;
    uint state_dim;
    uint total_threads;
    float eps;
    float q_scale;
};

static inline float softplus_stable(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

kernel void supersonic_qwen_linear_prep_decode_apply_bf16_f32(
    device const bfloat* conv_pack [[buffer(0)]],
    device const bfloat* a [[buffer(1)]],
    device const bfloat* b [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device const bfloat* a_log_exp [[buffer(4)]],
    device const float* initial_state [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant QwenLinearPrepDecodeApplyParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }

    uint v_head = gid / params.v_head_dim;
    uint vv = gid - v_head * params.v_head_dim;
    uint k_head = v_head / params.head_repeat;
    uint qk_base = k_head * params.k_head_dim;
    uint v_base = v_head * params.v_head_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        float qh = float(conv_pack[qk_base + kk]);
        float kh = float(conv_pack[params.key_dim + qk_base + kk]);
        q_sum = fma(qh, qh, q_sum);
        k_sum = fma(kh, kh, k_sum);
    }
    float q_inv_norm = rsqrt(q_sum + params.eps) * params.q_scale;
    float k_inv_norm = rsqrt(k_sum + params.eps);

    float beta = 1.0f / (1.0f + exp(-float(b[v_head])));
    float g_exp =
        exp(-softplus_stable(float(a[v_head]) + float(dt_bias[v_head])) * float(a_log_exp[v_head]));

    uint state_head_base = (v_head * params.k_head_dim) * params.v_head_dim + vv;
    uint state_out_base = params.value_dim + (v_head * params.k_head_dim) * params.v_head_dim + vv;
    float kv_mem = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        float k_norm = float(conv_pack[params.key_dim + qk_base + kk]) * k_inv_norm;
        float state = initial_state[state_head_base + kk * params.v_head_dim] * g_exp;
        kv_mem = fma(state, k_norm, kv_mem);
        out[state_out_base + kk * params.v_head_dim] = state;
    }

    float v_linear = float(conv_pack[params.key_dim * 2 + v_base + vv]);
    float delta = (v_linear - kv_mem) * beta;
    float out_value = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        float q_scaled = float(conv_pack[qk_base + kk]) * q_inv_norm;
        float k_norm = float(conv_pack[params.key_dim + qk_base + kk]) * k_inv_norm;
        uint state_idx = state_out_base + kk * params.v_head_dim;
        float state = fma(k_norm, delta, out[state_idx]);
        out[state_idx] = state;
        out_value = fma(state, q_scaled, out_value);
    }
    out[v_base + vv] = out_value;
}
)LDPREPAPPLY";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:581
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen linear prep/apply library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_linear_prep_decode_apply_bf16_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:582
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen linear prep/apply function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                code:583
                                                                            userInfo:@{
                                                                                NSLocalizedDescriptionKey :
                                                                                    @"Failed to create Qwen linear prep/apply pipeline"
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

id<MTLComputePipelineState> qwen_linear_decode_apply_inplace_pipeline(NSError** error_out) {
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
                                                   code:584
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"LDAPI(
#include <metal_stdlib>
using namespace metal;

struct QwenLinearPrepDecodeApplyParams {
    uint num_v_heads;
    uint num_k_heads;
    uint head_repeat;
    uint k_head_dim;
    uint v_head_dim;
    uint key_dim;
    uint value_dim;
    uint state_dim;
    uint total_threads;
    float eps;
    float q_scale;
};

static inline float softplus_stable(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

kernel void supersonic_qwen_linear_decode_apply_inplace_bf16(
    device const bfloat* conv_pack [[buffer(0)]],
    device const bfloat* a [[buffer(1)]],
    device const bfloat* b [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device const bfloat* a_log_exp [[buffer(4)]],
    device float* state [[buffer(5)]],
    device bfloat* attn_out [[buffer(6)]],
    constant QwenLinearPrepDecodeApplyParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }

    uint v_head = gid / params.v_head_dim;
    uint vv = gid - v_head * params.v_head_dim;
    uint k_head = v_head / params.head_repeat;
    uint qk_base = k_head * params.k_head_dim;
    uint v_base = v_head * params.v_head_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        float qh = float(conv_pack[qk_base + kk]);
        float kh = float(conv_pack[params.key_dim + qk_base + kk]);
        q_sum = fma(qh, qh, q_sum);
        k_sum = fma(kh, kh, k_sum);
    }
    float q_inv_norm = rsqrt(q_sum + params.eps) * params.q_scale;
    float k_inv_norm = rsqrt(k_sum + params.eps);

    float beta = 1.0f / (1.0f + exp(-float(b[v_head])));
    float g_exp =
        exp(-softplus_stable(float(a[v_head]) + float(dt_bias[v_head])) * float(a_log_exp[v_head]));

    uint state_head_base = (v_head * params.k_head_dim) * params.v_head_dim + vv;
    float kv_mem = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        uint state_idx = state_head_base + kk * params.v_head_dim;
        float k_norm = float(conv_pack[params.key_dim + qk_base + kk]) * k_inv_norm;
        float prior = state[state_idx] * g_exp;
        kv_mem = fma(prior, k_norm, kv_mem);
        state[state_idx] = prior;
    }

    float v_linear = float(conv_pack[params.key_dim * 2 + v_base + vv]);
    float delta = (v_linear - kv_mem) * beta;
    float out_value = 0.0f;
    for (uint kk = 0; kk < params.k_head_dim; ++kk) {
        float q_scaled = float(conv_pack[qk_base + kk]) * q_inv_norm;
        float k_norm = float(conv_pack[params.key_dim + qk_base + kk]) * k_inv_norm;
        uint state_idx = state_head_base + kk * params.v_head_dim;
        float updated = fma(k_norm, delta, state[state_idx]);
        state[state_idx] = updated;
        out_value = fma(updated, q_scaled, out_value);
    }
    attn_out[v_base + vv] = bfloat(out_value);
}
)LDAPI";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:585
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen linear decode apply inplace library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen_linear_decode_apply_inplace_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:586
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen linear decode apply inplace function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                code:587
                                                                            userInfo:@{
                                                                                NSLocalizedDescriptionKey :
                                                                                    @"Failed to create Qwen linear decode apply inplace pipeline"
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

id<MTLComputePipelineState> conv_state_update_bf16_pipeline(NSError** error_out) {
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
                                                   code:205
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"CSU(
#include <metal_stdlib>
using namespace metal;

struct ConvStateUpdateParams {
    uint channels;
    uint state_len;
    uint total_threads;
};

kernel void supersonic_conv_state_update_bf16(
    device bfloat* state [[buffer(0)]],
    device const bfloat* qkv [[buffer(1)]],
    constant ConvStateUpdateParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }
    uint channel = gid / params.state_len;
    uint pos = gid - channel * params.state_len;
    uint base = channel * params.state_len;
    if (pos + 1 < params.state_len) {
        state[base + pos] = state[base + pos + 1];
    } else {
        state[base + pos] = qkv[channel];
    }
}
)CSU";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:206
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile conv-state-update library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_conv_state_update_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:207
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load conv-state-update function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                code:208
                                                                            userInfo:@{
                                                                                NSLocalizedDescriptionKey :
                                                                                    @"Failed to create conv-state-update pipeline"
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

id<MTLComputePipelineState> linear_conv_value_decay_bf16_pipeline(
    NSString* function_name,
    NSError** error_out
) {
    static std::mutex mutex;
    static bool attempted_plain = false;
    static bool attempted_update = false;
    static __strong id<MTLComputePipelineState> pipeline_plain = nil;
    static __strong id<MTLComputePipelineState> pipeline_update = nil;
    static __strong NSError* build_error_plain = nil;
    static __strong NSError* build_error_update = nil;

    const bool want_update =
        [function_name isEqualToString:@"supersonic_linear_conv_value_decay_update_bf16"];
    bool& attempted = want_update ? attempted_update : attempted_plain;
    __strong id<MTLComputePipelineState>& pipeline = want_update ? pipeline_update : pipeline_plain;
    __strong NSError*& build_error = want_update ? build_error_update : build_error_plain;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:219
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"LCVD(
#include <metal_stdlib>
using namespace metal;

struct LinearConvValueDecayParams {
    uint conv_dim;
    uint state_len;
    uint kernel_size;
    uint num_heads;
    uint out_width;
    uint total_threads;
};

static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float softplus_stable(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

kernel void supersonic_linear_conv_value_decay_bf16(
    device const bfloat* mixed_qkv [[buffer(0)]],
    device const bfloat* prev_state [[buffer(1)]],
    device const bfloat* weights [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* dt_bias [[buffer(4)]],
    device const bfloat* a_log_exp [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant LinearConvValueDecayParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }
    if (gid < params.conv_dim) {
        uint c = gid;
        uint state_base = c * params.state_len;
        uint weight_base = c * params.kernel_size;
        float acc = 0.0f;
        uint history = params.kernel_size - 1;
        for (uint tap = 0; tap < params.kernel_size; ++tap) {
            int src = int(tap) - int(history);
            float x = 0.0f;
            if (src >= 0) {
                x = float(mixed_qkv[c]);
            } else {
                int state_idx = int(params.state_len) + src;
                if (state_idx >= 0) {
                    x = float(prev_state[state_base + uint(state_idx)]);
                }
            }
            acc = fma(x, float(weights[weight_base + tap]), acc);
        }
        out[c] = bfloat(silu(acc));
        return;
    }

    uint head = gid - params.conv_dim;
    if (head < params.num_heads) {
        float value = -softplus_stable(float(a[head]) + float(dt_bias[head])) * float(a_log_exp[head]);
        out[params.conv_dim + head] = bfloat(value);
    }
}

kernel void supersonic_linear_conv_value_decay_update_bf16(
    device const bfloat* mixed_qkv [[buffer(0)]],
    device bfloat* state [[buffer(1)]],
    device const bfloat* weights [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* dt_bias [[buffer(4)]],
    device const bfloat* a_log_exp [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant LinearConvValueDecayParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_threads) {
        return;
    }
    if (gid < params.conv_dim) {
        uint c = gid;
        uint state_base = c * params.state_len;
        uint weight_base = c * params.kernel_size;
        float acc = 0.0f;
        uint history = params.kernel_size - 1;
        for (uint tap = 0; tap < params.kernel_size; ++tap) {
            int src = int(tap) - int(history);
            float x = 0.0f;
            if (src >= 0) {
                x = float(mixed_qkv[c]);
            } else {
                int state_idx = int(params.state_len) + src;
                if (state_idx >= 0) {
                    x = float(state[state_base + uint(state_idx)]);
                }
            }
            acc = fma(x, float(weights[weight_base + tap]), acc);
        }
        out[c] = bfloat(silu(acc));

        for (uint pos = 0; pos < params.state_len; ++pos) {
            if (pos + 1 < params.state_len) {
                state[state_base + pos] = state[state_base + pos + 1];
            } else {
                state[state_base + pos] = mixed_qkv[c];
            }
        }
        return;
    }

    uint head = gid - params.conv_dim;
    if (head < params.num_heads) {
        float value = -softplus_stable(float(a[head]) + float(dt_bias[head])) * float(a_log_exp[head]);
        out[params.conv_dim + head] = bfloat(value);
    }
}
)LCVD";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:220
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile linear-conv-value-decay library"
                                                                   }];
                } else {
                    id<MTLFunction> function = [library newFunctionWithName:function_name];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:221
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load linear-conv-value-decay function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                code:222
                                                                            userInfo:@{
                                                                                NSLocalizedDescriptionKey :
                                                                                    @"Failed to create linear-conv-value-decay pipeline"
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
    uint block_size;
};

kernel void supersonic_l2norm_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant L2NormParams& params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }
    threadgroup float scratch[256];
    uint base = row * params.n_cols;
    float norm_sq = 0.0f;
    for (uint c = tid; c < params.n_cols; c += params.block_size) {
        float v = input[base + c];
        norm_sq = fma(v, v, norm_sq);
    }
    scratch[tid] = norm_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_norm = rsqrt(scratch[0] + params.eps);
    for (uint c = tid; c < params.n_cols; c += params.block_size) {
        out[base + c] = input[base + c] * inv_norm;
    }
}

kernel void supersonic_l2norm_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant L2NormParams& params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= params.n_rows || tid >= params.block_size) {
        return;
    }
    threadgroup float scratch[256];
    uint base = row * params.n_cols;
    float norm_sq = 0.0f;
    for (uint c = tid; c < params.n_cols; c += params.block_size) {
        float v = float(input[base + c]);
        norm_sq = fma(v, v, norm_sq);
    }
    scratch[tid] = norm_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_norm = rsqrt(scratch[0] + params.eps);
    for (uint c = tid; c < params.n_cols; c += params.block_size) {
        out[base + c] = bfloat(float(input[base + c]) * inv_norm);
    }
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

extern "C" int supersonic_metal_batch_begin() {
    @autoreleasepool {
        return metal_batch_begin();
    }
}

extern "C" int supersonic_metal_batch_flush() {
    @autoreleasepool {
        return metal_batch_flush();
    }
}

extern "C" int supersonic_metal_batch_set_label(const char* label) {
    @autoreleasepool {
        return metal_batch_set_label(label);
    }
}

extern "C" int supersonic_metal_batch_end() {
    @autoreleasepool {
        return metal_batch_end();
    }
}

extern "C" int supersonic_metal_batch_is_active() {
    return metal_batch_depth > 0 ? 1 : 0;
}

extern "C" int supersonic_metal_copy_d2d(
    const void* src_ptr,
    void* dst_ptr,
    size_t bytes
) {
    @autoreleasepool {
        if (src_ptr == nullptr || dst_ptr == nullptr || bytes == 0) {
            return 930;
        }
        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 931;
        }
        if (lookup_buffer(dst_ptr, &dst, &dst_offset) != 0) {
            return 932;
        }
        if (bytes > NSUIntegerMax || src_offset > NSUIntegerMax || dst_offset > NSUIntegerMax) {
            return 933;
        }
        return encode_blit_copy_or_submit(
            src,
            static_cast<NSUInteger>(src_offset),
            dst,
            static_cast<NSUInteger>(dst_offset),
            static_cast<NSUInteger>(bytes),
            934,
            935,
            936,
            937
        );
    }
}

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

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
            [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 76, 77, 78, 79);
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

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 200, 201, 202, 203);
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

extern "C" int supersonic_metal_full_attention_gate_bf16(
    size_t total_elems,
    const void* attn_f32_ptr,
    const void* gate_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || attn_f32_ptr == nullptr || gate_ptr == nullptr ||
            out_ptr == nullptr) {
            return 208;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = full_attention_gate_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 209;
        }

        id<MTLBuffer> attn = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> out = nil;
        size_t attn_offset = 0;
        size_t gate_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(attn_f32_ptr, &attn, &attn_offset) != 0) {
            return 210;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 211;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 212;
        }

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:attn offset:attn_offset atIndex:0];
            [encoder setBuffer:gate offset:gate_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 213, 214, 215, 216);
    }
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

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 220, 221, 222, 223);
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

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:out offset:out_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 85, 86, 87, 88);
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

        MulScalarParams params = {static_cast<uint32_t>(total_elems), scalar};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:out offset:out_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 95, 96, 97, 98);
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

        TransposeShdHsdParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(h),
            static_cast<uint32_t>(d),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:dst offset:dst_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 106, 107, 108, 109);
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

static int supersonic_metal_apply_rope_prefill_impl(
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t pos_offset,
    const void* cos_ptr,
    const void* sin_ptr,
    void* data_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (seq_len == 0 || num_heads == 0 || head_dim == 0 || rotary_dim == 0) {
            return 0;
        }
        if (seq_len > UINT32_MAX || num_heads > UINT32_MAX || head_dim > UINT32_MAX ||
            rotary_dim > UINT32_MAX || pos_offset > UINT32_MAX || cos_ptr == nullptr ||
            sin_ptr == nullptr || data_ptr == nullptr) {
            return 354;
        }
        size_t half_rot = rotary_dim / 2;
        if (half_rot == 0 || half_rot > head_dim || half_rot > UINT32_MAX) {
            return 355;
        }
        size_t total_pairs = seq_len * num_heads * half_rot;
        if (total_pairs > UINT32_MAX || total_pairs / half_rot / num_heads != seq_len) {
            return 356;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = rope_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 357;
        }

        id<MTLBuffer> data = nil;
        id<MTLBuffer> cos_table = nil;
        id<MTLBuffer> sin_table = nil;
        size_t data_offset = 0;
        size_t cos_offset = 0;
        size_t sin_offset = 0;
        if (lookup_buffer(data_ptr, &data, &data_offset) != 0) {
            return 358;
        }
        if (lookup_buffer(cos_ptr, &cos_table, &cos_offset) != 0) {
            return 359;
        }
        if (lookup_buffer(sin_ptr, &sin_table, &sin_offset) != 0) {
            return 360;
        }

        RopeParams params = {
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(rotary_dim),
            static_cast<uint32_t>(half_rot),
            static_cast<uint32_t>(pos_offset),
            static_cast<uint32_t>(total_pairs),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:data offset:data_offset atIndex:0];
            [encoder setBuffer:cos_table offset:cos_offset atIndex:1];
            [encoder setBuffer:sin_table offset:sin_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_pairs, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 361, 362, 363, 364);
    }
}

extern "C" int supersonic_metal_apply_rope_prefill_bf16(
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t pos_offset,
    const void* cos_ptr,
    const void* sin_ptr,
    void* data_ptr
) {
    return supersonic_metal_apply_rope_prefill_impl(
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        pos_offset,
        cos_ptr,
        sin_ptr,
        data_ptr,
        @"supersonic_apply_rope_prefill_bf16"
    );
}

extern "C" int supersonic_metal_apply_rope_prefill_f32(
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t rotary_dim,
    size_t pos_offset,
    const void* cos_ptr,
    const void* sin_ptr,
    void* data_ptr
) {
    return supersonic_metal_apply_rope_prefill_impl(
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        pos_offset,
        cos_ptr,
        sin_ptr,
        data_ptr,
        @"supersonic_apply_rope_prefill_f32"
    );
}

static int supersonic_metal_transpose_pad_conv_impl(
    size_t s,
    size_t c,
    size_t pad,
    const void* src_ptr,
    void* dst_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (s == 0 || c == 0) {
            return 0;
        }
        if (s > UINT32_MAX || c > UINT32_MAX || pad > UINT32_MAX || src_ptr == nullptr || dst_ptr == nullptr) {
            return 365;
        }
        size_t stride = pad + s;
        size_t total_dst = c * stride;
        if (stride > UINT32_MAX || total_dst > UINT32_MAX || total_dst / stride != c) {
            return 366;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = conv_layout_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 367;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 368;
        }
        if (lookup_buffer(dst_ptr, &dst, &dst_offset) != 0) {
            return 369;
        }

        TransposePadConvParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(c),
            static_cast<uint32_t>(pad),
            static_cast<uint32_t>(stride),
            static_cast<uint32_t>(total_dst),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:dst offset:dst_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_dst, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 370, 371, 372, 373);
    }
}

extern "C" int supersonic_metal_transpose_pad_conv_bf16(
    size_t s,
    size_t c,
    size_t pad,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_transpose_pad_conv_impl(
        s,
        c,
        pad,
        src_ptr,
        dst_ptr,
        @"supersonic_transpose_pad_conv_bf16"
    );
}

extern "C" int supersonic_metal_transpose_pad_conv_f32(
    size_t s,
    size_t c,
    size_t pad,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_transpose_pad_conv_impl(
        s,
        c,
        pad,
        src_ptr,
        dst_ptr,
        @"supersonic_transpose_pad_conv_f32"
    );
}

static int supersonic_metal_extract_conv_state_impl(
    size_t s,
    size_t c,
    size_t kern_minus_1,
    const void* src_ptr,
    void* dst_ptr,
    NSString* function_name
) {
    @autoreleasepool {
        if (c == 0 || kern_minus_1 == 0) {
            return 0;
        }
        if (s > UINT32_MAX || c > UINT32_MAX || kern_minus_1 > UINT32_MAX ||
            src_ptr == nullptr || dst_ptr == nullptr) {
            return 374;
        }
        size_t total_dst = c * kern_minus_1;
        if (total_dst > UINT32_MAX || total_dst / kern_minus_1 != c) {
            return 375;
        }
        size_t copy = std::min(s, kern_minus_1);
        size_t start = s >= copy ? s - copy : 0;
        size_t dst_start = kern_minus_1 - copy;

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = conv_layout_pipeline(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 376;
        }

        id<MTLBuffer> src = nil;
        id<MTLBuffer> dst = nil;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        if (lookup_buffer(src_ptr, &src, &src_offset) != 0) {
            return 377;
        }
        if (lookup_buffer(dst_ptr, &dst, &dst_offset) != 0) {
            return 378;
        }

        ExtractConvStateParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(c),
            static_cast<uint32_t>(kern_minus_1),
            static_cast<uint32_t>(copy),
            static_cast<uint32_t>(start),
            static_cast<uint32_t>(dst_start),
            static_cast<uint32_t>(total_dst),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:dst offset:dst_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_dst, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 379, 380, 381, 382);
    }
}

extern "C" int supersonic_metal_extract_conv_state_bf16(
    size_t s,
    size_t c,
    size_t kern_minus_1,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_extract_conv_state_impl(
        s,
        c,
        kern_minus_1,
        src_ptr,
        dst_ptr,
        @"supersonic_extract_conv_state_bf16"
    );
}

extern "C" int supersonic_metal_extract_conv_state_f32(
    size_t s,
    size_t c,
    size_t kern_minus_1,
    const void* src_ptr,
    void* dst_ptr
) {
    return supersonic_metal_extract_conv_state_impl(
        s,
        c,
        kern_minus_1,
        src_ptr,
        dst_ptr,
        @"supersonic_extract_conv_state_f32"
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

        SplitQkvParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(key_dim),
            static_cast<uint32_t>(val_dim),
            static_cast<uint32_t>(src_stride),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:q offset:q_offset atIndex:1];
            [encoder setBuffer:k offset:k_offset atIndex:2];
            [encoder setBuffer:v offset:v_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 119, 120, 121, 122);
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

        SplitQgateParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(src_stride),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:query offset:query_offset atIndex:1];
            [encoder setBuffer:gate offset:gate_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 130, 131, 132, 133);
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

        RepeatInterleaveHeadsParams params = {
            static_cast<uint32_t>(s),
            static_cast<uint32_t>(n_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(repeats),
            static_cast<uint32_t>(dst_heads),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:dst offset:dst_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 140, 141, 142, 143);
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

        ComputeBetaGParams params = {
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(nv),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:b offset:b_offset atIndex:0];
            [encoder setBuffer:a offset:a_offset atIndex:1];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:2];
            [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:3];
            [encoder setBuffer:beta offset:beta_offset atIndex:4];
            [encoder setBuffer:g offset:g_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 153, 154, 155, 156);
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

        DeltaRecurrentPrefillParams params = {
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(k_head_dim),
            static_cast<uint32_t>(v_head_dim),
            static_cast<uint32_t>(out_rows),
            static_cast<uint32_t>(total_threads),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 174, 175, 176, 177);
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

        // Qwen's real l2norm path is F32 after BF16->F32 casts. Keep that
        // accumulation order identical to the host/oracle path; the parallel
        // reduction is faster but nudges tight logit thresholds over the line.
        const bool preserve_f32_order = [function_name isEqualToString:@"supersonic_l2norm_f32"];
        NSUInteger block_size = preserve_f32_order
            ? 1
            : std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        if (block_size == 0) {
            block_size = 1;
        }
        L2NormParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            static_cast<uint32_t>(total_elems),
            static_cast<uint32_t>(block_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:out offset:out_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];

            MTLSize threadgroups = MTLSizeMake(n_rows, 1, 1);
            MTLSize threads_per_group = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 187, 188, 189, 190);
    }
}

extern "C" int supersonic_metal_linear_decode_apply_parts_f32(
    size_t num_v_heads,
    size_t num_k_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    const void* q_scaled_ptr,
    const void* k_normed_ptr,
    const void* v_linear_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* dt_bias_ptr,
    const void* a_log_exp_ptr,
    const void* initial_state_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (num_v_heads == 0 || num_k_heads == 0 || head_k_dim == 0 || head_v_dim == 0 ||
            num_v_heads % num_k_heads != 0 || q_scaled_ptr == nullptr || k_normed_ptr == nullptr ||
            v_linear_ptr == nullptr || a_ptr == nullptr || b_ptr == nullptr ||
            dt_bias_ptr == nullptr || a_log_exp_ptr == nullptr || initial_state_ptr == nullptr ||
            out_ptr == nullptr) {
            return 185;
        }
        if (num_v_heads > UINT32_MAX || num_k_heads > UINT32_MAX ||
            head_k_dim > UINT32_MAX || head_v_dim > UINT32_MAX) {
            return 186;
        }
        if (num_v_heads > SIZE_MAX / head_v_dim) {
            return 187;
        }
        size_t total_threads = num_v_heads * head_v_dim;
        if (total_threads > UINT32_MAX) {
            return 188;
        }
        if (head_k_dim != 0 && num_v_heads > SIZE_MAX / head_k_dim / head_v_dim) {
            return 189;
        }
        size_t value_dim = total_threads;
        size_t state_dim = num_v_heads * head_k_dim * head_v_dim;
        if (value_dim > UINT32_MAX || state_dim > UINT32_MAX) {
            return 190;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = linear_decode_apply_parts_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 191;
        }

        id<MTLBuffer> q_scaled = nil;
        id<MTLBuffer> k_normed = nil;
        id<MTLBuffer> v_linear = nil;
        id<MTLBuffer> a = nil;
        id<MTLBuffer> b = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log_exp = nil;
        id<MTLBuffer> initial_state = nil;
        id<MTLBuffer> out = nil;
        size_t q_scaled_offset = 0;
        size_t k_normed_offset = 0;
        size_t v_linear_offset = 0;
        size_t a_offset = 0;
        size_t b_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_exp_offset = 0;
        size_t initial_state_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(q_scaled_ptr, &q_scaled, &q_scaled_offset) != 0) return 192;
        if (lookup_buffer(k_normed_ptr, &k_normed, &k_normed_offset) != 0) return 193;
        if (lookup_buffer(v_linear_ptr, &v_linear, &v_linear_offset) != 0) return 194;
        if (lookup_buffer(a_ptr, &a, &a_offset) != 0) return 195;
        if (lookup_buffer(b_ptr, &b, &b_offset) != 0) return 196;
        if (lookup_buffer(dt_bias_ptr, &dt_bias, &dt_bias_offset) != 0) return 197;
        if (lookup_buffer(a_log_exp_ptr, &a_log_exp, &a_log_exp_offset) != 0) return 198;
        if (lookup_buffer(initial_state_ptr, &initial_state, &initial_state_offset) != 0) return 199;
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) return 200;

        LinearDecodeApplyParams params = {
            static_cast<uint32_t>(num_v_heads),
            static_cast<uint32_t>(num_k_heads),
            static_cast<uint32_t>(num_v_heads / num_k_heads),
            static_cast<uint32_t>(head_k_dim),
            static_cast<uint32_t>(head_v_dim),
            static_cast<uint32_t>(value_dim),
            static_cast<uint32_t>(state_dim),
            static_cast<uint32_t>(total_threads),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:q_scaled offset:q_scaled_offset atIndex:0];
            [encoder setBuffer:k_normed offset:k_normed_offset atIndex:1];
            [encoder setBuffer:v_linear offset:v_linear_offset atIndex:2];
            [encoder setBuffer:a offset:a_offset atIndex:3];
            [encoder setBuffer:b offset:b_offset atIndex:4];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:5];
            [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:6];
            [encoder setBuffer:initial_state offset:initial_state_offset atIndex:7];
            [encoder setBuffer:out offset:out_offset atIndex:8];
            [encoder setBytes:&params length:sizeof(params) atIndex:9];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(total_threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        }, 201, 202, 203, 204);
    }
}

extern "C" int supersonic_metal_qwen_linear_prep_bf16_f32(
    size_t key_dim,
    size_t val_dim,
    size_t num_key_heads,
    size_t key_head_dim,
    const void* conv_pack_ptr,
    void* q_bf16_ptr,
    void* k_bf16_ptr,
    void* v_bf16_ptr,
    void* q_f32_ptr,
    void* k_f32_ptr,
    void* v_f32_ptr,
    void* q_normed_ptr,
    void* q_scaled_ptr,
    void* k_normed_ptr
) {
    @autoreleasepool {
        if (key_dim == 0 || val_dim == 0 || num_key_heads == 0 || key_head_dim == 0 ||
            conv_pack_ptr == nullptr || q_bf16_ptr == nullptr || k_bf16_ptr == nullptr ||
            v_bf16_ptr == nullptr || q_f32_ptr == nullptr || k_f32_ptr == nullptr ||
            v_f32_ptr == nullptr || q_normed_ptr == nullptr || q_scaled_ptr == nullptr ||
            k_normed_ptr == nullptr) {
            return 190;
        }
        if (key_dim != num_key_heads * key_head_dim) {
            return 191;
        }
        size_t total_threads = std::max(key_dim, val_dim);
        if (key_dim > UINT32_MAX || val_dim > UINT32_MAX || num_key_heads > UINT32_MAX ||
            key_head_dim > UINT32_MAX || total_threads > UINT32_MAX) {
            return 192;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen_linear_prep_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 193;
        }

        id<MTLBuffer> conv_pack = nil;
        id<MTLBuffer> q_bf16 = nil;
        id<MTLBuffer> k_bf16 = nil;
        id<MTLBuffer> v_bf16 = nil;
        id<MTLBuffer> q_f32 = nil;
        id<MTLBuffer> k_f32 = nil;
        id<MTLBuffer> v_f32 = nil;
        id<MTLBuffer> q_normed = nil;
        id<MTLBuffer> q_scaled = nil;
        id<MTLBuffer> k_normed = nil;
        size_t conv_pack_offset = 0;
        size_t q_bf16_offset = 0;
        size_t k_bf16_offset = 0;
        size_t v_bf16_offset = 0;
        size_t q_f32_offset = 0;
        size_t k_f32_offset = 0;
        size_t v_f32_offset = 0;
        size_t q_normed_offset = 0;
        size_t q_scaled_offset = 0;
        size_t k_normed_offset = 0;
        if (lookup_buffer(conv_pack_ptr, &conv_pack, &conv_pack_offset) != 0) {
            return 194;
        }
        if (lookup_buffer(q_bf16_ptr, &q_bf16, &q_bf16_offset) != 0) {
            return 195;
        }
        if (lookup_buffer(k_bf16_ptr, &k_bf16, &k_bf16_offset) != 0) {
            return 196;
        }
        if (lookup_buffer(v_bf16_ptr, &v_bf16, &v_bf16_offset) != 0) {
            return 197;
        }
        if (lookup_buffer(q_f32_ptr, &q_f32, &q_f32_offset) != 0) {
            return 198;
        }
        if (lookup_buffer(k_f32_ptr, &k_f32, &k_f32_offset) != 0) {
            return 199;
        }
        if (lookup_buffer(v_f32_ptr, &v_f32, &v_f32_offset) != 0) {
            return 200;
        }
        if (lookup_buffer(q_normed_ptr, &q_normed, &q_normed_offset) != 0) {
            return 201;
        }
        if (lookup_buffer(q_scaled_ptr, &q_scaled, &q_scaled_offset) != 0) {
            return 202;
        }
        if (lookup_buffer(k_normed_ptr, &k_normed, &k_normed_offset) != 0) {
            return 203;
        }

        QwenLinearPrepParams params = {
            static_cast<uint32_t>(key_dim),
            static_cast<uint32_t>(val_dim),
            static_cast<uint32_t>(num_key_heads),
            static_cast<uint32_t>(key_head_dim),
            static_cast<uint32_t>(total_threads),
            1.0e-6f,
            1.0f / sqrtf(static_cast<float>(key_head_dim)),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:conv_pack offset:conv_pack_offset atIndex:0];
            [encoder setBuffer:q_bf16 offset:q_bf16_offset atIndex:1];
            [encoder setBuffer:k_bf16 offset:k_bf16_offset atIndex:2];
            [encoder setBuffer:v_bf16 offset:v_bf16_offset atIndex:3];
            [encoder setBuffer:q_f32 offset:q_f32_offset atIndex:4];
            [encoder setBuffer:k_f32 offset:k_f32_offset atIndex:5];
            [encoder setBuffer:v_f32 offset:v_f32_offset atIndex:6];
            [encoder setBuffer:q_normed offset:q_normed_offset atIndex:7];
            [encoder setBuffer:q_scaled offset:q_scaled_offset atIndex:8];
            [encoder setBuffer:k_normed offset:k_normed_offset atIndex:9];
            [encoder setBytes:&params length:sizeof(params) atIndex:10];

            NSUInteger threads_per_group =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_grid = MTLSizeMake(total_threads, 1, 1);
            MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:group];
        }, 204, 205, 206, 207);
    }
}

extern "C" int supersonic_metal_qwen_linear_prep_decode_apply_bf16_f32(
    size_t num_v_heads,
    size_t num_k_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    const void* conv_pack_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* dt_bias_ptr,
    const void* a_log_exp_ptr,
    const void* initial_state_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (num_v_heads == 0 || num_k_heads == 0 || head_k_dim == 0 || head_v_dim == 0 ||
            num_v_heads % num_k_heads != 0 || conv_pack_ptr == nullptr || a_ptr == nullptr ||
            b_ptr == nullptr || dt_bias_ptr == nullptr || a_log_exp_ptr == nullptr ||
            initial_state_ptr == nullptr || out_ptr == nullptr) {
            return 584;
        }
        if (num_v_heads > UINT32_MAX || num_k_heads > UINT32_MAX ||
            head_k_dim > UINT32_MAX || head_v_dim > UINT32_MAX) {
            return 585;
        }
        if (num_v_heads > SIZE_MAX / head_v_dim) {
            return 586;
        }
        size_t total_threads = num_v_heads * head_v_dim;
        if (total_threads > UINT32_MAX) {
            return 587;
        }
        if (num_k_heads > SIZE_MAX / head_k_dim ||
            num_v_heads > SIZE_MAX / head_k_dim / head_v_dim) {
            return 588;
        }
        size_t key_dim = num_k_heads * head_k_dim;
        size_t value_dim = total_threads;
        size_t state_dim = num_v_heads * head_k_dim * head_v_dim;
        if (key_dim > UINT32_MAX || value_dim > UINT32_MAX || state_dim > UINT32_MAX) {
            return 589;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            qwen_linear_prep_decode_apply_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 590;
        }

        id<MTLBuffer> conv_pack = nil;
        id<MTLBuffer> a = nil;
        id<MTLBuffer> b = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log_exp = nil;
        id<MTLBuffer> initial_state = nil;
        id<MTLBuffer> out = nil;
        size_t conv_pack_offset = 0;
        size_t a_offset = 0;
        size_t b_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_exp_offset = 0;
        size_t initial_state_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(conv_pack_ptr, &conv_pack, &conv_pack_offset) != 0) return 591;
        if (lookup_buffer(a_ptr, &a, &a_offset) != 0) return 592;
        if (lookup_buffer(b_ptr, &b, &b_offset) != 0) return 593;
        if (lookup_buffer(dt_bias_ptr, &dt_bias, &dt_bias_offset) != 0) return 594;
        if (lookup_buffer(a_log_exp_ptr, &a_log_exp, &a_log_exp_offset) != 0) return 595;
        if (lookup_buffer(initial_state_ptr, &initial_state, &initial_state_offset) != 0) return 596;
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) return 597;

        QwenLinearPrepDecodeApplyParams params = {
            static_cast<uint32_t>(num_v_heads),
            static_cast<uint32_t>(num_k_heads),
            static_cast<uint32_t>(num_v_heads / num_k_heads),
            static_cast<uint32_t>(head_k_dim),
            static_cast<uint32_t>(head_v_dim),
            static_cast<uint32_t>(key_dim),
            static_cast<uint32_t>(value_dim),
            static_cast<uint32_t>(state_dim),
            static_cast<uint32_t>(total_threads),
            1.0e-6f,
            1.0f / sqrtf(static_cast<float>(head_k_dim)),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:conv_pack offset:conv_pack_offset atIndex:0];
            [encoder setBuffer:a offset:a_offset atIndex:1];
            [encoder setBuffer:b offset:b_offset atIndex:2];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:3];
            [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:4];
            [encoder setBuffer:initial_state offset:initial_state_offset atIndex:5];
            [encoder setBuffer:out offset:out_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(total_threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        }, 598, 599, 600, 601);
    }
}

extern "C" int supersonic_metal_qwen_linear_decode_apply_inplace_bf16(
    size_t num_v_heads,
    size_t num_k_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    const void* conv_pack_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* dt_bias_ptr,
    const void* a_log_exp_ptr,
    void* state_ptr,
    void* attn_out_ptr
) {
    @autoreleasepool {
        if (num_v_heads == 0 || num_k_heads == 0 || head_k_dim == 0 || head_v_dim == 0 ||
            num_v_heads % num_k_heads != 0 || conv_pack_ptr == nullptr || a_ptr == nullptr ||
            b_ptr == nullptr || dt_bias_ptr == nullptr || a_log_exp_ptr == nullptr ||
            state_ptr == nullptr || attn_out_ptr == nullptr) {
            return 607;
        }
        if (num_v_heads > UINT32_MAX || num_k_heads > UINT32_MAX ||
            head_k_dim > UINT32_MAX || head_v_dim > UINT32_MAX) {
            return 608;
        }
        size_t total_threads = num_v_heads * head_v_dim;
        if (total_threads > UINT32_MAX) {
            return 609;
        }
        if (num_k_heads > SIZE_MAX / head_k_dim || num_v_heads > SIZE_MAX / head_k_dim / head_v_dim) {
            return 610;
        }
        size_t key_dim = num_k_heads * head_k_dim;
        size_t value_dim = total_threads;
        size_t state_dim = num_v_heads * head_k_dim * head_v_dim;
        if (key_dim > UINT32_MAX || value_dim > UINT32_MAX || state_dim > UINT32_MAX) {
            return 611;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            qwen_linear_decode_apply_inplace_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 612;
        }

        id<MTLBuffer> conv_pack = nil;
        id<MTLBuffer> a = nil;
        id<MTLBuffer> b = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log_exp = nil;
        id<MTLBuffer> state = nil;
        id<MTLBuffer> attn_out = nil;
        size_t conv_pack_offset = 0;
        size_t a_offset = 0;
        size_t b_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_exp_offset = 0;
        size_t state_offset = 0;
        size_t attn_out_offset = 0;
        if (lookup_buffer(conv_pack_ptr, &conv_pack, &conv_pack_offset) != 0) return 613;
        if (lookup_buffer(a_ptr, &a, &a_offset) != 0) return 614;
        if (lookup_buffer(b_ptr, &b, &b_offset) != 0) return 615;
        if (lookup_buffer(dt_bias_ptr, &dt_bias, &dt_bias_offset) != 0) return 616;
        if (lookup_buffer(a_log_exp_ptr, &a_log_exp, &a_log_exp_offset) != 0) return 617;
        if (lookup_buffer(state_ptr, &state, &state_offset) != 0) return 618;
        if (lookup_buffer(attn_out_ptr, &attn_out, &attn_out_offset) != 0) return 619;

        QwenLinearPrepDecodeApplyParams params = {
            static_cast<uint32_t>(num_v_heads),
            static_cast<uint32_t>(num_k_heads),
            static_cast<uint32_t>(num_v_heads / num_k_heads),
            static_cast<uint32_t>(head_k_dim),
            static_cast<uint32_t>(head_v_dim),
            static_cast<uint32_t>(key_dim),
            static_cast<uint32_t>(value_dim),
            static_cast<uint32_t>(state_dim),
            static_cast<uint32_t>(total_threads),
            1.0e-6f,
            1.0f / sqrtf(static_cast<float>(head_k_dim)),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:conv_pack offset:conv_pack_offset atIndex:0];
            [encoder setBuffer:a offset:a_offset atIndex:1];
            [encoder setBuffer:b offset:b_offset atIndex:2];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:3];
            [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:4];
            [encoder setBuffer:state offset:state_offset atIndex:5];
            [encoder setBuffer:attn_out offset:attn_out_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(total_threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        }, 620, 621, 622, 623);
    }
}

extern "C" int supersonic_metal_conv_state_update_bf16(
    size_t channels,
    size_t state_len,
    const void* qkv_ptr,
    void* state_ptr
) {
    @autoreleasepool {
        if (channels == 0 || state_len == 0 || qkv_ptr == nullptr || state_ptr == nullptr) {
            return 209;
        }
        if (channels > UINT32_MAX || state_len > UINT32_MAX || channels > SIZE_MAX / state_len) {
            return 210;
        }
        size_t total_threads = channels * state_len;
        if (total_threads > UINT32_MAX) {
            return 211;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = conv_state_update_bf16_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 212;
        }

        id<MTLBuffer> qkv = nil;
        id<MTLBuffer> state = nil;
        size_t qkv_offset = 0;
        size_t state_offset = 0;
        if (lookup_buffer(qkv_ptr, &qkv, &qkv_offset) != 0) return 213;
        if (lookup_buffer(state_ptr, &state, &state_offset) != 0) return 214;

        ConvStateUpdateParams params = {
            static_cast<uint32_t>(channels),
            static_cast<uint32_t>(state_len),
            static_cast<uint32_t>(total_threads),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:state offset:state_offset atIndex:0];
            [encoder setBuffer:qkv offset:qkv_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(total_threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        }, 215, 216, 217, 218);
    }
}

extern "C" int supersonic_metal_linear_conv_value_decay_bf16(
    size_t conv_dim,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv_ptr,
    const void* prev_state_ptr,
    const void* weights_ptr,
    const void* a_ptr,
    const void* dt_bias_ptr,
    const void* a_log_exp_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (conv_dim == 0 || state_len == 0 || kernel_size == 0 || num_heads == 0 ||
            mixed_qkv_ptr == nullptr || prev_state_ptr == nullptr || weights_ptr == nullptr ||
            a_ptr == nullptr || dt_bias_ptr == nullptr || a_log_exp_ptr == nullptr ||
            out_ptr == nullptr) {
            return 223;
        }
        if (kernel_size != state_len + 1) {
            return 224;
        }
        if (conv_dim > UINT32_MAX || state_len > UINT32_MAX || kernel_size > UINT32_MAX ||
            num_heads > UINT32_MAX || conv_dim > SIZE_MAX - num_heads) {
            return 225;
        }
        size_t out_width = conv_dim + num_heads;
        if (out_width > UINT32_MAX) {
            return 226;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            linear_conv_value_decay_bf16_pipeline(
                @"supersonic_linear_conv_value_decay_bf16",
                &pipeline_error);
        if (pipeline == nil) {
            return 227;
        }

        id<MTLBuffer> mixed_qkv = nil;
        id<MTLBuffer> prev_state = nil;
        id<MTLBuffer> weights = nil;
        id<MTLBuffer> a = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log_exp = nil;
        id<MTLBuffer> out = nil;
        size_t mixed_qkv_offset = 0;
        size_t prev_state_offset = 0;
        size_t weights_offset = 0;
        size_t a_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_exp_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(mixed_qkv_ptr, &mixed_qkv, &mixed_qkv_offset) != 0) return 228;
        if (lookup_buffer(prev_state_ptr, &prev_state, &prev_state_offset) != 0) return 229;
        if (lookup_buffer(weights_ptr, &weights, &weights_offset) != 0) return 230;
        if (lookup_buffer(a_ptr, &a, &a_offset) != 0) return 231;
        if (lookup_buffer(dt_bias_ptr, &dt_bias, &dt_bias_offset) != 0) return 232;
        if (lookup_buffer(a_log_exp_ptr, &a_log_exp, &a_log_exp_offset) != 0) return 233;
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) return 234;

        LinearConvValueDecayParams params = {
            static_cast<uint32_t>(conv_dim),
            static_cast<uint32_t>(state_len),
            static_cast<uint32_t>(kernel_size),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(out_width),
            static_cast<uint32_t>(out_width),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:mixed_qkv offset:mixed_qkv_offset atIndex:0];
            [encoder setBuffer:prev_state offset:prev_state_offset atIndex:1];
            [encoder setBuffer:weights offset:weights_offset atIndex:2];
            [encoder setBuffer:a offset:a_offset atIndex:3];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:4];
            [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:5];
            [encoder setBuffer:out offset:out_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(out_width, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        }, 235, 236, 237, 238);
    }
}

extern "C" int supersonic_metal_linear_conv_value_decay_update_bf16(
    size_t conv_dim,
    size_t state_len,
    size_t kernel_size,
    size_t num_heads,
    const void* mixed_qkv_ptr,
    void* state_ptr,
    const void* weights_ptr,
    const void* a_ptr,
    const void* dt_bias_ptr,
    const void* a_log_exp_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (conv_dim == 0 || state_len == 0 || kernel_size == 0 || num_heads == 0 ||
            mixed_qkv_ptr == nullptr || state_ptr == nullptr || weights_ptr == nullptr ||
            a_ptr == nullptr || dt_bias_ptr == nullptr || a_log_exp_ptr == nullptr ||
            out_ptr == nullptr) {
            return 602;
        }
        if (kernel_size != state_len + 1) {
            return 603;
        }
        if (conv_dim > UINT32_MAX || state_len > UINT32_MAX || kernel_size > UINT32_MAX ||
            num_heads > UINT32_MAX || conv_dim > SIZE_MAX - num_heads) {
            return 604;
        }
        size_t out_width = conv_dim + num_heads;
        if (out_width > UINT32_MAX) {
            return 605;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            linear_conv_value_decay_bf16_pipeline(
                @"supersonic_linear_conv_value_decay_update_bf16",
                &pipeline_error);
        if (pipeline == nil) {
            return 606;
        }

        id<MTLBuffer> mixed_qkv = nil;
        id<MTLBuffer> state = nil;
        id<MTLBuffer> weights = nil;
        id<MTLBuffer> a = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log_exp = nil;
        id<MTLBuffer> out = nil;
        size_t mixed_qkv_offset = 0;
        size_t state_offset = 0;
        size_t weights_offset = 0;
        size_t a_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_exp_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(mixed_qkv_ptr, &mixed_qkv, &mixed_qkv_offset) != 0) return 607;
        if (lookup_buffer(state_ptr, &state, &state_offset) != 0) return 608;
        if (lookup_buffer(weights_ptr, &weights, &weights_offset) != 0) return 609;
        if (lookup_buffer(a_ptr, &a, &a_offset) != 0) return 610;
        if (lookup_buffer(dt_bias_ptr, &dt_bias, &dt_bias_offset) != 0) return 611;
        if (lookup_buffer(a_log_exp_ptr, &a_log_exp, &a_log_exp_offset) != 0) return 612;
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) return 613;

        LinearConvValueDecayParams params = {
            static_cast<uint32_t>(conv_dim),
            static_cast<uint32_t>(state_len),
            static_cast<uint32_t>(kernel_size),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(out_width),
            static_cast<uint32_t>(out_width),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:mixed_qkv offset:mixed_qkv_offset atIndex:0];
            [encoder setBuffer:state offset:state_offset atIndex:1];
            [encoder setBuffer:weights offset:weights_offset atIndex:2];
            [encoder setBuffer:a offset:a_offset atIndex:3];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:4];
            [encoder setBuffer:a_log_exp offset:a_log_exp_offset atIndex:5];
            [encoder setBuffer:out offset:out_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(out_width, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        }, 614, 615, 616, 617);
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

extern "C" int supersonic_metal_embedding_lookup_bf16(
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    const void* embeddings_ptr,
    const void* indexes_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (token_count == 0 || vocab_size == 0 || hidden_size == 0 || token_count > UINT32_MAX ||
            vocab_size > UINT32_MAX || hidden_size > UINT32_MAX || embeddings_ptr == nullptr ||
            indexes_ptr == nullptr || out_ptr == nullptr) {
            return 242;
        }
        const size_t total_elems = token_count * hidden_size;
        if (hidden_size != 0 && total_elems / hidden_size != token_count) {
            return 243;
        }
        if (total_elems > UINT32_MAX) {
            return 244;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = embedding_lookup_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 245;
        }

        id<MTLBuffer> embeddings = nil;
        id<MTLBuffer> indexes = nil;
        id<MTLBuffer> out = nil;
        size_t embeddings_offset = 0;
        size_t indexes_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(embeddings_ptr, &embeddings, &embeddings_offset) != 0) {
            return 246;
        }
        if (lookup_buffer(indexes_ptr, &indexes, &indexes_offset) != 0) {
            return 247;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 248;
        }

        EmbeddingLookupParams params = {
            static_cast<uint32_t>(token_count),
            static_cast<uint32_t>(vocab_size),
            static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:embeddings offset:embeddings_offset atIndex:0];
            [encoder setBuffer:indexes offset:indexes_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 249, 250, 251, 252);
    }
}

extern "C" int supersonic_metal_matmul_rhs_transposed_bf16_gemv_m1(
    size_t n,
    size_t k,
    const void* lhs_ptr,
    const void* rhs_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (n == 0 || k == 0 || lhs_ptr == nullptr || rhs_ptr == nullptr || out_ptr == nullptr) {
            return 510;
        }
        if (n > UINT32_MAX || k > UINT32_MAX) {
            return 511;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_pipeline_bf16_gemv_m1(&pipeline_error);
        if (pipeline == nil) {
            return 512;
        }

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 513;
        }
        if (lookup_buffer(rhs_ptr, &rhs, &rhs_offset) != 0) {
            return 514;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 515;
        }

        struct MatmulGemvParams {
            uint32_t n;
            uint32_t k;
        } params = {
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
            [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            // One SIMD-group (32 threads) per output column.
            MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
            MTLSize threadgroups = MTLSizeMake(n, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 516, 517, 518, 519);
    }
}

extern "C" int supersonic_metal_matmul_rhs_transposed_bf16_gemv_m1_tiled(
    size_t n,
    size_t k,
    const void* lhs_ptr,
    const void* rhs_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (n == 0 || k == 0 || lhs_ptr == nullptr || rhs_ptr == nullptr || out_ptr == nullptr) {
            return 620;
        }
        if (n > UINT32_MAX || k > UINT32_MAX) {
            return 621;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_pipeline_bf16_gemv_m1_tiled(&pipeline_error);
        if (pipeline == nil) {
            return 622;
        }

        size_t shared_bytes = k * sizeof(float);

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 624;
        }
        if (lookup_buffer(rhs_ptr, &rhs, &rhs_offset) != 0) {
            return 625;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 626;
        }

        struct MatmulGemvParams {
            uint32_t n;
            uint32_t k;
        } params = {
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
        };

        // 32 cols per threadgroup → ceil(n / 32) threadgroups.
        size_t tg_count = (n + 31) / 32;

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
            [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];

            MTLSize threads_per_group = MTLSizeMake(1024, 1, 1);
            MTLSize threadgroups = MTLSizeMake(tg_count, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 627, 628, 629, 630);
    }
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

        MatmulParams params = {
            static_cast<uint32_t>(batch_elems),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 6, 7, 8, 9);
    }
}

extern "C" int supersonic_metal_matmul_rhs_transposed_residual_bf16(
    size_t batch_elems,
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_ptr,
    const void* rhs_ptr,
    const void* residual_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (batch_elems == 0 || m == 0 || n == 0 || k == 0 || lhs_ptr == nullptr || rhs_ptr == nullptr ||
            residual_ptr == nullptr || out_ptr == nullptr) {
            return 345;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_residual_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 346;
        }

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> residual = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t residual_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 347;
        }
        if (lookup_buffer(rhs_ptr, &rhs, &rhs_offset) != 0) {
            return 348;
        }
        if (lookup_buffer(residual_ptr, &residual, &residual_offset) != 0) {
            return 349;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 350;
        }

        MatmulParams params = {
            static_cast<uint32_t>(batch_elems),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
            [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
            [encoder setBuffer:residual offset:residual_offset atIndex:2];
            [encoder setBuffer:out offset:out_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            NSUInteger tg_width = std::min<NSUInteger>(8, std::max<NSUInteger>(1, n));
            NSUInteger tg_height =
                std::min<NSUInteger>(8, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tg_width));
            if (tg_height == 0) {
                tg_height = 1;
            }
            MTLSize threads_per_group = MTLSizeMake(tg_width, tg_height, 1);
            MTLSize threads_per_grid = MTLSizeMake(n, m, batch_elems);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 351, 352, 353, 354);
    }
}

extern "C" int supersonic_metal_matmul_rhs_transposed_int4_bf16(
    size_t batch_elems,
    size_t m,
    size_t n,
    size_t k,
    size_t group_size,
    const void* lhs_ptr,
    const void* rhs_int4_ptr,
    const void* scale_ptr,
    const void* zero_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (batch_elems == 0 || m == 0 || n == 0 || k == 0 || group_size == 0 ||
            lhs_ptr == nullptr || rhs_int4_ptr == nullptr || scale_ptr == nullptr ||
            zero_ptr == nullptr || out_ptr == nullptr) {
            return 410;
        }
        if (k % 2 != 0) {
            return 411;
        }
        if (batch_elems > UINT32_MAX || m > UINT32_MAX || n > UINT32_MAX ||
            k > UINT32_MAX || group_size > UINT32_MAX) {
            return 412;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_int4_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 413;
        }

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> sc = nil;
        id<MTLBuffer> zr = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t sc_offset = 0;
        size_t zr_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 414;
        }
        if (lookup_buffer(rhs_int4_ptr, &rhs, &rhs_offset) != 0) {
            return 415;
        }
        if (lookup_buffer(scale_ptr, &sc, &sc_offset) != 0) {
            return 416;
        }
        if (lookup_buffer(zero_ptr, &zr, &zr_offset) != 0) {
            return 417;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 418;
        }

        struct MatmulInt4Params {
            uint32_t batch_elems;
            uint32_t m;
            uint32_t n;
            uint32_t k;
            uint32_t group_size;
        } params = {
            static_cast<uint32_t>(batch_elems),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
            static_cast<uint32_t>(group_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
            [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
            [encoder setBuffer:sc offset:sc_offset atIndex:2];
            [encoder setBuffer:zr offset:zr_offset atIndex:3];
            [encoder setBuffer:out offset:out_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];

            NSUInteger tg_width = std::min<NSUInteger>(8, std::max<NSUInteger>(1, n));
            NSUInteger tg_height =
                std::min<NSUInteger>(8, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tg_width));
            if (tg_height == 0) {
                tg_height = 1;
            }
            MTLSize threads_per_group = MTLSizeMake(tg_width, tg_height, 1);
            MTLSize threads_per_grid = MTLSizeMake(n, m, batch_elems);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 419, 420, 421, 422);
    }
}

extern "C" int supersonic_metal_qwen_mlp_gate_up_bf16(
    size_t hidden_dim,
    size_t intermediate_dim,
    const void* input_ptr,
    const void* gate_weight_ptr,
    const void* up_weight_ptr,
    void* gate_out_ptr,
    void* up_out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || intermediate_dim == 0 || input_ptr == nullptr ||
            gate_weight_ptr == nullptr || up_weight_ptr == nullptr || gate_out_ptr == nullptr ||
            up_out_ptr == nullptr) {
            return 375;
        }
        if (hidden_dim > UINT32_MAX || intermediate_dim > UINT32_MAX) {
            return 376;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen_mlp_gate_up_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 377;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> gate_out = nil;
        id<MTLBuffer> up_out = nil;
        size_t input_offset = 0;
        size_t gate_weight_offset = 0;
        size_t up_weight_offset = 0;
        size_t gate_out_offset = 0;
        size_t up_out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 378;
        }
        if (lookup_buffer(gate_weight_ptr, &gate_weight, &gate_weight_offset) != 0) {
            return 379;
        }
        if (lookup_buffer(up_weight_ptr, &up_weight, &up_weight_offset) != 0) {
            return 380;
        }
        if (lookup_buffer(gate_out_ptr, &gate_out, &gate_out_offset) != 0) {
            return 381;
        }
        if (lookup_buffer(up_out_ptr, &up_out, &up_out_offset) != 0) {
            return 382;
        }

        QwenMlpParams params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(intermediate_dim),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:gate_weight offset:gate_weight_offset atIndex:1];
            [encoder setBuffer:up_weight offset:up_weight_offset atIndex:2];
            [encoder setBuffer:gate_out offset:gate_out_offset atIndex:3];
            [encoder setBuffer:up_out offset:up_out_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(intermediate_dim, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 383, 384, 385, 386);
    }
}

extern "C" int supersonic_metal_qwen_mlp_gate_up_swiglu_bf16(
    size_t hidden_dim,
    size_t intermediate_dim,
    const void* input_ptr,
    const void* gate_weight_ptr,
    const void* up_weight_ptr,
    void* mlp_out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || intermediate_dim == 0 || input_ptr == nullptr ||
            gate_weight_ptr == nullptr || up_weight_ptr == nullptr || mlp_out_ptr == nullptr) {
            return 425;
        }
        if (hidden_dim > UINT32_MAX || intermediate_dim > UINT32_MAX) {
            return 426;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen_mlp_gate_up_swiglu_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 427;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> gate_weight = nil;
        id<MTLBuffer> up_weight = nil;
        id<MTLBuffer> mlp_out = nil;
        size_t input_offset = 0;
        size_t gate_weight_offset = 0;
        size_t up_weight_offset = 0;
        size_t mlp_out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 428;
        }
        if (lookup_buffer(gate_weight_ptr, &gate_weight, &gate_weight_offset) != 0) {
            return 429;
        }
        if (lookup_buffer(up_weight_ptr, &up_weight, &up_weight_offset) != 0) {
            return 430;
        }
        if (lookup_buffer(mlp_out_ptr, &mlp_out, &mlp_out_offset) != 0) {
            return 431;
        }

        QwenMlpParams params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(intermediate_dim),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:gate_weight offset:gate_weight_offset atIndex:1];
            [encoder setBuffer:up_weight offset:up_weight_offset atIndex:2];
            [encoder setBuffer:mlp_out offset:mlp_out_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(intermediate_dim, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 432, 433, 434, 435);
    }
}

extern "C" int supersonic_metal_qwen_mlp_down_residual_bf16(
    size_t hidden_dim,
    size_t intermediate_dim,
    const void* gate_ptr,
    const void* up_ptr,
    const void* down_weight_ptr,
    const void* residual_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || intermediate_dim == 0 || gate_ptr == nullptr || up_ptr == nullptr ||
            down_weight_ptr == nullptr || residual_ptr == nullptr || out_ptr == nullptr) {
            return 387;
        }
        if (hidden_dim > UINT32_MAX || intermediate_dim > UINT32_MAX) {
            return 388;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen_mlp_down_residual_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 389;
        }

        id<MTLBuffer> gate = nil;
        id<MTLBuffer> up = nil;
        id<MTLBuffer> down_weight = nil;
        id<MTLBuffer> residual = nil;
        id<MTLBuffer> out = nil;
        size_t gate_offset = 0;
        size_t up_offset = 0;
        size_t down_weight_offset = 0;
        size_t residual_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 390;
        }
        if (lookup_buffer(up_ptr, &up, &up_offset) != 0) {
            return 391;
        }
        if (lookup_buffer(down_weight_ptr, &down_weight, &down_weight_offset) != 0) {
            return 392;
        }
        if (lookup_buffer(residual_ptr, &residual, &residual_offset) != 0) {
            return 393;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 394;
        }

        QwenMlpParams params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(intermediate_dim),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:gate offset:gate_offset atIndex:0];
            [encoder setBuffer:up offset:up_offset atIndex:1];
            [encoder setBuffer:down_weight offset:down_weight_offset atIndex:2];
            [encoder setBuffer:residual offset:residual_offset atIndex:3];
            [encoder setBuffer:out offset:out_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(hidden_dim, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 395, 396, 397, 398);
    }
}

extern "C" int supersonic_metal_qwen_linear_out_residual_f32_bf16(
    size_t hidden_dim,
    size_t num_rows,
    size_t row_dim,
    float eps,
    const void* attn_ptr,
    const void* gate_ptr,
    const void* weight_ptr,
    const void* out_proj_ptr,
    const void* residual_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || num_rows == 0 || row_dim == 0 || attn_ptr == nullptr ||
            gate_ptr == nullptr || weight_ptr == nullptr || out_proj_ptr == nullptr ||
            residual_ptr == nullptr || out_ptr == nullptr) {
            return 399;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            qwen_linear_out_residual_pipeline(
                @"supersonic_qwen_linear_out_residual_f32_bf16",
                &pipeline_error
            );
        if (pipeline == nil) {
            return 400;
        }

        id<MTLBuffer> attn = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> out_proj = nil;
        id<MTLBuffer> residual = nil;
        id<MTLBuffer> out = nil;
        size_t attn_offset = 0;
        size_t gate_offset = 0;
        size_t weight_offset = 0;
        size_t out_proj_offset = 0;
        size_t residual_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(attn_ptr, &attn, &attn_offset) != 0) {
            return 401;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 402;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 403;
        }
        if (lookup_buffer(out_proj_ptr, &out_proj, &out_proj_offset) != 0) {
            return 404;
        }
        if (lookup_buffer(residual_ptr, &residual, &residual_offset) != 0) {
            return 405;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 406;
        }

        struct QwenLinearOutParams {
            uint32_t hidden_dim;
            uint32_t num_rows;
            uint32_t row_dim;
            float eps;
        } params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(row_dim),
            eps,
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:attn offset:attn_offset atIndex:0];
            [encoder setBuffer:gate offset:gate_offset atIndex:1];
            [encoder setBuffer:weight offset:weight_offset atIndex:2];
            [encoder setBuffer:out_proj offset:out_proj_offset atIndex:3];
            [encoder setBuffer:residual offset:residual_offset atIndex:4];
            [encoder setBuffer:out offset:out_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(hidden_dim, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 407, 408, 409, 410);
    }
}

extern "C" int supersonic_metal_qwen_linear_out_residual_bf16_bf16(
    size_t hidden_dim,
    size_t num_rows,
    size_t row_dim,
    float eps,
    const void* attn_ptr,
    const void* gate_ptr,
    const void* weight_ptr,
    const void* out_proj_ptr,
    const void* residual_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || num_rows == 0 || row_dim == 0 || attn_ptr == nullptr ||
            gate_ptr == nullptr || weight_ptr == nullptr || out_proj_ptr == nullptr ||
            residual_ptr == nullptr || out_ptr == nullptr) {
            return 411;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            qwen_linear_out_residual_pipeline(
                @"supersonic_qwen_linear_out_residual_bf16_bf16",
                &pipeline_error
            );
        if (pipeline == nil) {
            return 412;
        }

        id<MTLBuffer> attn = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> out_proj = nil;
        id<MTLBuffer> residual = nil;
        id<MTLBuffer> out = nil;
        size_t attn_offset = 0;
        size_t gate_offset = 0;
        size_t weight_offset = 0;
        size_t out_proj_offset = 0;
        size_t residual_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(attn_ptr, &attn, &attn_offset) != 0) {
            return 413;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 414;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 415;
        }
        if (lookup_buffer(out_proj_ptr, &out_proj, &out_proj_offset) != 0) {
            return 416;
        }
        if (lookup_buffer(residual_ptr, &residual, &residual_offset) != 0) {
            return 417;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 418;
        }

        struct QwenLinearOutParams {
            uint32_t hidden_dim;
            uint32_t num_rows;
            uint32_t row_dim;
            float eps;
        } params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(row_dim),
            eps,
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:attn offset:attn_offset atIndex:0];
            [encoder setBuffer:gate offset:gate_offset atIndex:1];
            [encoder setBuffer:weight offset:weight_offset atIndex:2];
            [encoder setBuffer:out_proj offset:out_proj_offset atIndex:3];
            [encoder setBuffer:residual offset:residual_offset atIndex:4];
            [encoder setBuffer:out offset:out_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(hidden_dim, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 419, 420, 421, 422);
    }
}

extern "C" int supersonic_metal_qwen_full_projections_bf16(
    size_t hidden_dim,
    size_t q_proj_dim,
    size_t kv_dim,
    const void* input_ptr,
    const void* q_weight_ptr,
    const void* k_weight_ptr,
    const void* v_weight_ptr,
    void* q_out_ptr,
    void* k_out_ptr,
    void* v_out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || q_proj_dim == 0 || kv_dim == 0 || input_ptr == nullptr ||
            q_weight_ptr == nullptr || k_weight_ptr == nullptr || v_weight_ptr == nullptr ||
            q_out_ptr == nullptr || k_out_ptr == nullptr || v_out_ptr == nullptr) {
            return 405;
        }
        const size_t total_cols = q_proj_dim + kv_dim * 2;
        if (hidden_dim > UINT32_MAX || q_proj_dim > UINT32_MAX || kv_dim > UINT32_MAX ||
            total_cols > UINT32_MAX) {
            return 406;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen_full_projection_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 407;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> q_weight = nil;
        id<MTLBuffer> k_weight = nil;
        id<MTLBuffer> v_weight = nil;
        id<MTLBuffer> q_out = nil;
        id<MTLBuffer> k_out = nil;
        id<MTLBuffer> v_out = nil;
        size_t input_offset = 0;
        size_t q_weight_offset = 0;
        size_t k_weight_offset = 0;
        size_t v_weight_offset = 0;
        size_t q_out_offset = 0;
        size_t k_out_offset = 0;
        size_t v_out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 408;
        }
        if (lookup_buffer(q_weight_ptr, &q_weight, &q_weight_offset) != 0) {
            return 409;
        }
        if (lookup_buffer(k_weight_ptr, &k_weight, &k_weight_offset) != 0) {
            return 410;
        }
        if (lookup_buffer(v_weight_ptr, &v_weight, &v_weight_offset) != 0) {
            return 411;
        }
        if (lookup_buffer(q_out_ptr, &q_out, &q_out_offset) != 0) {
            return 412;
        }
        if (lookup_buffer(k_out_ptr, &k_out, &k_out_offset) != 0) {
            return 413;
        }
        if (lookup_buffer(v_out_ptr, &v_out, &v_out_offset) != 0) {
            return 414;
        }

        QwenFullProjectionParams params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(q_proj_dim),
            static_cast<uint32_t>(kv_dim),
            static_cast<uint32_t>(total_cols),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:q_weight offset:q_weight_offset atIndex:1];
            [encoder setBuffer:k_weight offset:k_weight_offset atIndex:2];
            [encoder setBuffer:v_weight offset:v_weight_offset atIndex:3];
            [encoder setBuffer:q_out offset:q_out_offset atIndex:4];
            [encoder setBuffer:k_out offset:k_out_offset atIndex:5];
            [encoder setBuffer:v_out offset:v_out_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_cols, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 415, 416, 417, 418);
    }
}

extern "C" int supersonic_metal_matmul_rhs_transposed_f32(
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
            return 310;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_pipeline_f32(&pipeline_error);
        if (pipeline == nil) {
            return 311;
        }

        id<MTLBuffer> lhs = nil;
        id<MTLBuffer> rhs = nil;
        id<MTLBuffer> out = nil;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(lhs_ptr, &lhs, &lhs_offset) != 0) {
            return 312;
        }
        if (lookup_buffer(rhs_ptr, &rhs, &rhs_offset) != 0) {
            return 313;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 314;
        }

        MatmulParams params = {
            static_cast<uint32_t>(batch_elems),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 315, 316, 317, 318);
    }
}

extern "C" int supersonic_metal_qwen_linear_projections_bf16(
    size_t hidden_dim,
    size_t qkv_dim,
    size_t val_dim,
    size_t num_value_heads,
    const void* input_ptr,
    const void* qkv_weight_ptr,
    const void* z_weight_ptr,
    const void* a_weight_ptr,
    const void* b_weight_ptr,
    void* qkv_out_ptr,
    void* z_out_ptr,
    void* a_out_ptr,
    void* b_out_ptr
) {
    @autoreleasepool {
        if (hidden_dim == 0 || qkv_dim == 0 || val_dim == 0 || num_value_heads == 0 ||
            input_ptr == nullptr || qkv_weight_ptr == nullptr || z_weight_ptr == nullptr ||
            a_weight_ptr == nullptr || b_weight_ptr == nullptr || qkv_out_ptr == nullptr ||
            z_out_ptr == nullptr || a_out_ptr == nullptr || b_out_ptr == nullptr) {
            return 320;
        }
        size_t total_cols = qkv_dim + val_dim + num_value_heads * 2;
        if (hidden_dim > UINT32_MAX || qkv_dim > UINT32_MAX || val_dim > UINT32_MAX ||
            num_value_heads > UINT32_MAX || total_cols > UINT32_MAX) {
            return 321;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen_linear_projection_pipeline(&pipeline_error);
        if (pipeline == nil) {
            return 322;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> qkv_weight = nil;
        id<MTLBuffer> z_weight = nil;
        id<MTLBuffer> a_weight = nil;
        id<MTLBuffer> b_weight = nil;
        id<MTLBuffer> qkv_out = nil;
        id<MTLBuffer> z_out = nil;
        id<MTLBuffer> a_out = nil;
        id<MTLBuffer> b_out = nil;
        size_t input_offset = 0;
        size_t qkv_weight_offset = 0;
        size_t z_weight_offset = 0;
        size_t a_weight_offset = 0;
        size_t b_weight_offset = 0;
        size_t qkv_out_offset = 0;
        size_t z_out_offset = 0;
        size_t a_out_offset = 0;
        size_t b_out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 323;
        }
        if (lookup_buffer(qkv_weight_ptr, &qkv_weight, &qkv_weight_offset) != 0) {
            return 324;
        }
        if (lookup_buffer(z_weight_ptr, &z_weight, &z_weight_offset) != 0) {
            return 325;
        }
        if (lookup_buffer(a_weight_ptr, &a_weight, &a_weight_offset) != 0) {
            return 326;
        }
        if (lookup_buffer(b_weight_ptr, &b_weight, &b_weight_offset) != 0) {
            return 327;
        }
        if (lookup_buffer(qkv_out_ptr, &qkv_out, &qkv_out_offset) != 0) {
            return 328;
        }
        if (lookup_buffer(z_out_ptr, &z_out, &z_out_offset) != 0) {
            return 329;
        }
        if (lookup_buffer(a_out_ptr, &a_out, &a_out_offset) != 0) {
            return 330;
        }
        if (lookup_buffer(b_out_ptr, &b_out, &b_out_offset) != 0) {
            return 331;
        }

        QwenLinearProjectionParams params = {
            static_cast<uint32_t>(hidden_dim),
            static_cast<uint32_t>(qkv_dim),
            static_cast<uint32_t>(val_dim),
            static_cast<uint32_t>(num_value_heads),
            static_cast<uint32_t>(total_cols),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:qkv_weight offset:qkv_weight_offset atIndex:1];
            [encoder setBuffer:z_weight offset:z_weight_offset atIndex:2];
            [encoder setBuffer:a_weight offset:a_weight_offset atIndex:3];
            [encoder setBuffer:b_weight offset:b_weight_offset atIndex:4];
            [encoder setBuffer:qkv_out offset:qkv_out_offset atIndex:5];
            [encoder setBuffer:z_out offset:z_out_offset atIndex:6];
            [encoder setBuffer:a_out offset:a_out_offset atIndex:7];
            [encoder setBuffer:b_out offset:b_out_offset atIndex:8];
            [encoder setBytes:&params length:sizeof(params) atIndex:9];

            NSUInteger threads_per_group =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_grid = MTLSizeMake(total_cols, 1, 1);
            MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:group];
        }, 332, 333, 334, 335);
    }
}

extern "C" int supersonic_metal_lm_head_argmax_bf16(
    size_t in_dim,
    size_t vocab_size,
    const void* hidden_ptr,
    const void* weight_ptr,
    void* out_index_ptr,
    void* partial_values_ptr,
    void* partial_indices_ptr
) {
    @autoreleasepool {
        if (in_dim == 0 || vocab_size == 0 || hidden_ptr == nullptr || weight_ptr == nullptr ||
            out_index_ptr == nullptr) {
            return 270;
        }
        if (in_dim > UINT32_MAX || vocab_size > UINT32_MAX) {
            return 271;
        }

        NSError* stage1_error = nil;
        NSError* stage2_error = nil;
        id<MTLComputePipelineState> stage1 =
            lm_head_argmax_pipeline(@"supersonic_lm_head_argmax_stage1_bf16", &stage1_error);
        id<MTLComputePipelineState> stage2 =
            lm_head_argmax_pipeline(@"supersonic_lm_head_argmax_stage2", &stage2_error);
        if (stage1 == nil || stage2 == nil) {
            return 272;
        }

        id<MTLBuffer> hidden = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> out_index = nil;
        id<MTLBuffer> partial_values = nil;
        id<MTLBuffer> partial_indices = nil;
        size_t hidden_offset = 0;
        size_t weight_offset = 0;
        size_t out_index_offset = 0;
        size_t partial_values_offset = 0;
        size_t partial_indices_offset = 0;
        if (lookup_buffer(hidden_ptr, &hidden, &hidden_offset) != 0) {
            return 273;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 274;
        }
        if (lookup_buffer(out_index_ptr, &out_index, &out_index_offset) != 0) {
            return 275;
        }

        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return 276;
        }

        const uint32_t block_size = 256;
        const size_t partial_count = (vocab_size + block_size - 1) / block_size;
        if (partial_count == 0 || partial_count > UINT32_MAX) {
            return 279;
        }
        if (partial_values_ptr != nullptr) {
            if (lookup_buffer(partial_values_ptr, &partial_values, &partial_values_offset) != 0) {
                return 282;
            }
        } else {
            partial_values = [device newBufferWithLength:partial_count * sizeof(float)
                                                 options:MTLResourceStorageModePrivate];
        }
        if (partial_indices_ptr != nullptr) {
            if (lookup_buffer(partial_indices_ptr, &partial_indices, &partial_indices_offset) != 0) {
                return 283;
            }
        } else {
            partial_indices = [device newBufferWithLength:partial_count * sizeof(uint32_t)
                                                  options:MTLResourceStorageModePrivate];
        }
        if (partial_values == nil || partial_indices == nil) {
            return 280;
        }

        LmHeadArgmaxParams params = {
            static_cast<uint32_t>(in_dim),
            static_cast<uint32_t>(vocab_size),
            block_size,
            static_cast<uint32_t>(partial_count),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:stage1];
            [encoder setBuffer:hidden offset:hidden_offset atIndex:0];
            [encoder setBuffer:weight offset:weight_offset atIndex:1];
            [encoder setBuffer:partial_values offset:partial_values_offset atIndex:2];
            [encoder setBuffer:partial_indices offset:partial_indices_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            MTLSize groups = MTLSizeMake(partial_count, 1, 1);
            MTLSize threads = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:stage2];
            [encoder setBuffer:partial_values offset:partial_values_offset atIndex:0];
            [encoder setBuffer:partial_indices offset:partial_indices_offset atIndex:1];
            [encoder setBuffer:out_index offset:out_index_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:threads];
        }, 276, 277, 278, 281);
    }
}

extern "C" int supersonic_metal_argmax_bf16(
    size_t n,
    const void* logits_ptr,
    void* out_index_ptr
) {
    @autoreleasepool {
        if (n == 0 || logits_ptr == nullptr || out_index_ptr == nullptr) {
            return 310;
        }
        if (n > UINT32_MAX) {
            return 311;
        }

        NSError* stage1_error = nil;
        NSError* stage2_error = nil;
        id<MTLComputePipelineState> stage1 =
            lm_head_argmax_pipeline(@"supersonic_argmax_stage1_bf16", &stage1_error);
        id<MTLComputePipelineState> stage2 =
            lm_head_argmax_pipeline(@"supersonic_lm_head_argmax_stage2", &stage2_error);
        if (stage1 == nil || stage2 == nil) {
            return 312;
        }

        id<MTLBuffer> logits = nil;
        id<MTLBuffer> out_index = nil;
        size_t logits_offset = 0;
        size_t out_index_offset = 0;
        if (lookup_buffer(logits_ptr, &logits, &logits_offset) != 0) {
            return 313;
        }
        if (lookup_buffer(out_index_ptr, &out_index, &out_index_offset) != 0) {
            return 314;
        }

        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return 315;
        }

        const uint32_t block_size = 256;
        const size_t partial_count = (n + block_size - 1) / block_size;
        if (partial_count == 0 || partial_count > UINT32_MAX) {
            return 316;
        }
        id<MTLBuffer> partial_values = [device newBufferWithLength:partial_count * sizeof(float)
                                                           options:MTLResourceStorageModePrivate];
        id<MTLBuffer> partial_indices = [device newBufferWithLength:partial_count * sizeof(uint32_t)
                                                            options:MTLResourceStorageModePrivate];
        if (partial_values == nil || partial_indices == nil) {
            return 317;
        }

        LmHeadArgmaxParams params = {
            0,
            static_cast<uint32_t>(n),
            block_size,
            static_cast<uint32_t>(partial_count),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:stage1];
            [encoder setBuffer:logits offset:logits_offset atIndex:0];
            [encoder setBuffer:partial_values offset:0 atIndex:1];
            [encoder setBuffer:partial_indices offset:0 atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(partial_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:stage2];
            [encoder setBuffer:partial_values offset:0 atIndex:0];
            [encoder setBuffer:partial_indices offset:0 atIndex:1];
            [encoder setBuffer:out_index offset:out_index_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
        }, 315, 318, 319, 320);
    }
}

extern "C" int supersonic_metal_full_attention_prefill_bf16_f32(
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t kv_stride,
    size_t head_dim,
    float scale,
    size_t seqlen_offset,
    const void* query_ptr,
    const void* key_ptr,
    const void* value_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 || kv_stride < kv_len || head_dim == 0 ||
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

        FullAttentionParams params = {
            static_cast<uint32_t>(q_heads),
            static_cast<uint32_t>(kv_heads),
            static_cast<uint32_t>(q_len),
            static_cast<uint32_t>(kv_len),
            static_cast<uint32_t>(kv_stride),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(seqlen_offset),
            scale,
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 27, 28, 29, 30);
    }
}

extern "C" int supersonic_metal_full_attention_decode_bf16_f32(
    size_t q_heads,
    size_t kv_heads,
    size_t kv_len,
    size_t kv_stride,
    size_t head_dim,
    float scale,
    const void* query_ptr,
    const void* key_ptr,
    const void* value_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (q_heads == 0 || kv_heads == 0 || kv_len == 0 || kv_stride < kv_len || head_dim == 0 ||
            head_dim > 256 || query_ptr == nullptr || key_ptr == nullptr || value_ptr == nullptr ||
            out_ptr == nullptr) {
            return 511;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            full_attention_decode_pipeline_bf16_f32(&pipeline_error);
        if (pipeline == nil) {
            return 512;
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
            return 513;
        }
        if (lookup_buffer(key_ptr, &key, &key_offset) != 0) {
            return 514;
        }
        if (lookup_buffer(value_ptr, &value, &value_offset) != 0) {
            return 515;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 516;
        }

        FullAttentionDecodeParams params = {
            static_cast<uint32_t>(q_heads),
            static_cast<uint32_t>(kv_heads),
            static_cast<uint32_t>(kv_len),
            static_cast<uint32_t>(kv_stride),
            static_cast<uint32_t>(head_dim),
            scale,
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:query offset:query_offset atIndex:0];
            [encoder setBuffer:key offset:key_offset atIndex:1];
            [encoder setBuffer:value offset:value_offset atIndex:2];
            [encoder setBuffer:out offset:out_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            MTLSize groups = MTLSizeMake(q_heads, 1, 1);
            MTLSize threads = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        }, 517, 518, 519, 520);
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

        NSUInteger block_size = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        if (block_size == 0) {
            block_size = 1;
        }
        RmsNormParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            add_unit_offset ? 1u : 0u,
            static_cast<uint32_t>(block_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:weight offset:weight_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            MTLSize threadgroups = MTLSizeMake(n_rows, 1, 1);
            MTLSize threads_per_group = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 46, 47, 48, 49);
    }
}

extern "C" int supersonic_metal_rms_norm_rope_rows_bf16(
    size_t n_rows,
    size_t n_cols,
    size_t rotary_dim,
    float eps,
    size_t pos_offset,
    const void* input_ptr,
    const void* weight_ptr,
    const void* cos_ptr,
    const void* sin_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (n_rows == 0 || n_cols == 0 || input_ptr == nullptr || weight_ptr == nullptr ||
            cos_ptr == nullptr || sin_ptr == nullptr || out_ptr == nullptr || rotary_dim > n_cols ||
            (rotary_dim & 1) != 0) {
            return 138;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = rms_norm_rope_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 139;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> cos_table = nil;
        id<MTLBuffer> sin_table = nil;
        id<MTLBuffer> out = nil;
        size_t input_offset = 0;
        size_t weight_offset = 0;
        size_t cos_offset = 0;
        size_t sin_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 140;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 141;
        }
        if (lookup_buffer(cos_ptr, &cos_table, &cos_offset) != 0) {
            return 142;
        }
        if (lookup_buffer(sin_ptr, &sin_table, &sin_offset) != 0) {
            return 143;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 144;
        }

        NSUInteger block_size = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        if (block_size == 0) {
            block_size = 1;
        }
        struct RmsNormRopeParams {
            uint32_t n_rows;
            uint32_t n_cols;
            uint32_t rotary_dim;
            uint32_t half_rot;
            uint32_t pos_offset;
            float eps;
            uint32_t block_size;
        } params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            static_cast<uint32_t>(rotary_dim),
            static_cast<uint32_t>(rotary_dim / 2),
            static_cast<uint32_t>(pos_offset),
            eps,
            static_cast<uint32_t>(block_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:weight offset:weight_offset atIndex:1];
            [encoder setBuffer:cos_table offset:cos_offset atIndex:2];
            [encoder setBuffer:sin_table offset:sin_offset atIndex:3];
            [encoder setBuffer:out offset:out_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];

            MTLSize threadgroups = MTLSizeMake(n_rows, 1, 1);
            MTLSize threads_per_group = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 145, 146, 147, 148);
    }
}

int supersonic_metal_rms_norm_rows_f32_impl(
    NSString* function_name,
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
            return 344;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = rms_norm_pipeline_f32(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 345;
        }

        id<MTLBuffer> input = nil;
        id<MTLBuffer> weight = nil;
        id<MTLBuffer> out = nil;
        size_t input_offset = 0;
        size_t weight_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(input_ptr, &input, &input_offset) != 0) {
            return 346;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 347;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 348;
        }

        NSUInteger block_size = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        if (block_size == 0) {
            block_size = 1;
        }
        RmsNormParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            add_unit_offset ? 1u : 0u,
            static_cast<uint32_t>(block_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:weight offset:weight_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            MTLSize threadgroups = MTLSizeMake(n_rows, 1, 1);
            MTLSize threads_per_group = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 349, 350, 351, 352);
    }
}

extern "C" int supersonic_metal_rms_norm_rows_f32_weight_bf16(
    size_t n_rows,
    size_t n_cols,
    float eps,
    bool add_unit_offset,
    const void* input_ptr,
    const void* weight_ptr,
    void* out_ptr
) {
    return supersonic_metal_rms_norm_rows_f32_impl(
        @"supersonic_rms_norm_rows_f32_weight_bf16",
        n_rows,
        n_cols,
        eps,
        add_unit_offset,
        input_ptr,
        weight_ptr,
        out_ptr
    );
}

extern "C" int supersonic_metal_rms_norm_rows_f32_weight_f32(
    size_t n_rows,
    size_t n_cols,
    float eps,
    bool add_unit_offset,
    const void* input_ptr,
    const void* weight_ptr,
    void* out_ptr
) {
    return supersonic_metal_rms_norm_rows_f32_impl(
        @"supersonic_rms_norm_rows_f32_weight_f32",
        n_rows,
        n_cols,
        eps,
        add_unit_offset,
        input_ptr,
        weight_ptr,
        out_ptr
    );
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

        NSUInteger block_size = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        if (block_size == 0) {
            block_size = 1;
        }
        RmsNormGatedParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            static_cast<uint32_t>(block_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:hidden offset:hidden_offset atIndex:0];
            [encoder setBuffer:gate offset:gate_offset atIndex:1];
            [encoder setBuffer:weight offset:weight_offset atIndex:2];
            [encoder setBuffer:out offset:out_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            MTLSize threadgroups = MTLSizeMake(n_rows, 1, 1);
            MTLSize threads_per_group = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 234, 235, 236, 237);
    }
}

int supersonic_metal_rms_norm_gated_f32_impl(
    NSString* function_name,
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
            return 330;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = rms_norm_gated_pipeline_f32(function_name, &pipeline_error);
        if (pipeline == nil) {
            return 331;
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
            return 332;
        }
        if (lookup_buffer(gate_ptr, &gate, &gate_offset) != 0) {
            return 333;
        }
        if (lookup_buffer(weight_ptr, &weight, &weight_offset) != 0) {
            return 334;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 335;
        }

        NSUInteger block_size = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        if (block_size == 0) {
            block_size = 1;
        }
        RmsNormGatedParams params = {
            static_cast<uint32_t>(n_rows),
            static_cast<uint32_t>(n_cols),
            eps,
            static_cast<uint32_t>(block_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:hidden offset:hidden_offset atIndex:0];
            [encoder setBuffer:gate offset:gate_offset atIndex:1];
            [encoder setBuffer:weight offset:weight_offset atIndex:2];
            [encoder setBuffer:out offset:out_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            MTLSize threadgroups = MTLSizeMake(n_rows, 1, 1);
            MTLSize threads_per_group = MTLSizeMake(block_size, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 336, 337, 338, 339);
    }
}

extern "C" int supersonic_metal_rms_norm_gated_f32_weight_bf16(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden_ptr,
    const void* gate_ptr,
    const void* weight_ptr,
    void* out_ptr
) {
    return supersonic_metal_rms_norm_gated_f32_impl(
        @"supersonic_rms_norm_gated_f32_weight_bf16",
        n_rows,
        n_cols,
        eps,
        hidden_ptr,
        gate_ptr,
        weight_ptr,
        out_ptr
    );
}

extern "C" int supersonic_metal_rms_norm_gated_f32_weight_f32(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden_ptr,
    const void* gate_ptr,
    const void* weight_ptr,
    void* out_ptr
) {
    return supersonic_metal_rms_norm_gated_f32_impl(
        @"supersonic_rms_norm_gated_f32_weight_f32",
        n_rows,
        n_cols,
        eps,
        hidden_ptr,
        gate_ptr,
        weight_ptr,
        out_ptr
    );
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

        LinearConvParams params = {
            static_cast<uint32_t>(conv_dim),
            static_cast<uint32_t>(total_len),
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(kernel_size),
        };

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
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
        }, 66, 67, 68, 69);
    }
}
