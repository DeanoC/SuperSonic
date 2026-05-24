#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#if SUPERSONIC_HAVE_MTL4_MPP
#import <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

double wait_command_buffer_ms(id<MTLCommandBuffer> command_buffer, MetalClock::time_point start) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return 0.0;
    }
    const double gpu_elapsed_ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    if (std::isfinite(gpu_elapsed_ms) && gpu_elapsed_ms > 0.0) {
        return gpu_elapsed_ms;
    }
    return std::chrono::duration<double, std::milli>(MetalClock::now() - start).count();
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

struct Qwen36FfnInt4Params {
    uint32_t hidden;
    uint32_t num_experts;
    uint32_t moe_intermediate;
    uint32_t shared_intermediate;
    uint32_t top_k;
    uint32_t group_size;
    uint32_t off_h_norm;
    uint32_t off_topk_val;
    uint32_t off_topk_idx;
    uint32_t off_sg_scalar;
    uint32_t off_shared_mid;
    uint32_t off_shared_out;
    uint32_t off_expert_mid;
    uint32_t off_moe_out;
};

struct Qwen36FfnExpertGateUpTiledParams {
    uint hidden;
    uint moe_intermediate;
    uint top_k;
    uint group_size;
    uint off_h_norm;
    uint off_topk_idx;
    uint off_expert_mid;
};

struct Qwen36FfnExpertPackParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t top_k;
    uint32_t off_topk_idx;
};

struct Qwen36BatchedFfnExpertParams {
    uint32_t n_tokens;
    uint32_t top_k;
    uint32_t hidden;
    uint32_t moe_intermediate;
    uint32_t group_size;
};

struct Qwen36RouterTopkParams {
    uint32_t n_tokens;
    uint32_t num_experts;
    uint32_t top_k;
};

struct Qwen36LinearInt4Params {
    uint32_t hidden;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t head_k_dim;
    uint32_t head_v_dim;
    uint32_t conv_kernel_dim;
    uint32_t group_size;
    uint32_t key_dim;
    uint32_t val_dim;
    uint32_t qkv_dim;
    uint32_t kstate;
    uint32_t has_conv1d_bias;
    uint32_t off_qkv_raw;
    uint32_t off_z_raw;
    uint32_t off_a_raw;
    uint32_t off_b_raw;
    uint32_t off_q_normed;
    uint32_t off_k_normed;
    uint32_t off_q_rep;
    uint32_t off_k_rep;
    uint32_t off_beta;
    uint32_t off_g;
    uint32_t off_rec_out;
    float rms_norm_eps;
    float q_scale;
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

struct RowScalarSigmoidParams {
    uint32_t rows;
    uint32_t cols;
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
int encode_or_submit_labeled_maybe_wait(
    EncodeFn encode,
    const std::string& label,
    int queue_error,
    int command_buffer_error,
    int encoder_error,
    int completion_error,
    bool wait_for_completion
) {
    if (metal_batch_depth > 0 && metal_batch_state != nullptr) {
        int status = metal_batch_ensure_compute_encoder();
        if (status != 0) {
            return status;
        }
        if (!label.empty()) {
            metal_batch_state->label = label;
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
    if (!wait_for_completion) {
        std::string label_copy = label;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status == MTLCommandBufferStatusCompleted) {
                record_command_buffer_gpu_profile(completed, label_copy);
            }
        }];
    }
    auto commit_start = MetalClock::now();
    [command_buffer commit];
    record_runtime_profile("command_buffer_commit", commit_start);
    if (!wait_for_completion) {
        return 0;
    }
    auto wait_start = MetalClock::now();
    [command_buffer waitUntilCompleted];
    record_runtime_profile("command_buffer_wait", wait_start);

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return completion_error;
    }
    record_command_buffer_gpu_profile(command_buffer, label);
    return 0;
}

template <typename EncodeFn>
int encode_or_submit_labeled(
    EncodeFn encode,
    const std::string& label,
    int queue_error,
    int command_buffer_error,
    int encoder_error,
    int completion_error
) {
    return encode_or_submit_labeled_maybe_wait(
        encode,
        label,
        queue_error,
        command_buffer_error,
        encoder_error,
        completion_error,
        true
    );
}

template <typename EncodeFn>
int encode_or_submit(
    EncodeFn encode,
    int queue_error,
    int command_buffer_error,
    int encoder_error,
    int completion_error
) {
    return encode_or_submit_labeled(
        encode,
        "",
        queue_error,
        command_buffer_error,
        encoder_error,
        completion_error
    );
}

template <typename EncodeFn>
int encode_or_submit_labeled_async(
    EncodeFn encode,
    const std::string& label,
    int queue_error,
    int command_buffer_error,
    int encoder_error,
    int completion_error
) {
    return encode_or_submit_labeled_maybe_wait(
        encode,
        label,
        queue_error,
        command_buffer_error,
        encoder_error,
        completion_error,
        false
    );
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

bool qwen36_ffn_phase_profile_enabled() {
    return NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"] != nil;
}

bool qwen36_ffn_batch_phase_profile_enabled() {
    return NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"] != nil;
}

int flush_metal_batch_after_qwen36_ffn_profile_phase() {
    if (!qwen36_ffn_batch_phase_profile_enabled() || metal_batch_depth <= 0 || metal_batch_state == nullptr) {
        return 0;
    }
    return metal_batch_flush();
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

#if SUPERSONIC_HAVE_MTL4_MPP
id<MTLComputePipelineState> mpp_mtl4_pipeline_from_source(
    NSString* source,
    NSString* name,
    NSUInteger threads_per_threadgroup
) {
    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library = [metal_device() newLibraryWithSource:source options:options error:&error];
    if (library == nil || error != nil) {
        if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
            NSLog(@"SuperSonic MPP pilot Metal4 compile failed for %@: %@", name, error);
        }
        return nil;
    }

    MTL4LibraryFunctionDescriptor* function_desc = [[MTL4LibraryFunctionDescriptor alloc] init];
    function_desc.name = name;
    function_desc.library = library;

    MTL4ComputePipelineDescriptor* pipeline_desc = [[MTL4ComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunctionDescriptor = function_desc;
    pipeline_desc.maxTotalThreadsPerThreadgroup = threads_per_threadgroup;
    pipeline_desc.requiredThreadsPerThreadgroup = MTLSizeMake(threads_per_threadgroup, 1, 1);

    MTL4CompilerDescriptor* compiler_desc = [[MTL4CompilerDescriptor alloc] init];
    id<MTL4Compiler> compiler = [metal_device() newCompilerWithDescriptor:compiler_desc error:&error];
    if (compiler == nil || error != nil) {
        if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
            NSLog(@"SuperSonic MPP pilot Metal4 compiler failed for %@: %@", name, error);
        }
        return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [compiler newComputePipelineStateWithDescriptor:pipeline_desc compilerTaskOptions:nil error:&error];
    if (pipeline == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"SuperSonic MPP pilot Metal4 pipeline failed for %@: %@", name, error);
    }
    return pipeline;
}

MTLTensorDescriptor* mpp_tensor_descriptor(NSUInteger dim0, NSUInteger dim1, MTLTensorDataType data_type) {
    NSInteger dims[2] = {
        static_cast<NSInteger>(dim0),
        static_cast<NSInteger>(dim1),
    };
    NSInteger strides[2] = {
        1,
        static_cast<NSInteger>(dim0),
    };
    MTLTensorDescriptor* desc = [[MTLTensorDescriptor alloc] init];
    desc.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
    desc.strides = [[MTLTensorExtents alloc] initWithRank:2 values:strides];
    desc.dataType = data_type;
    desc.usage = MTLTensorUsageCompute | MTLTensorUsageMachineLearning;
    desc.storageMode = MTLStorageModeShared;
    return desc;
}

MTLTensorDescriptor* mpp_device_tensor_descriptor(NSUInteger dim0, NSUInteger dim1, MTLTensorDataType data_type) {
    MTLTensorDescriptor* desc = mpp_tensor_descriptor(dim0, dim1, data_type);
    desc.strides = nil;
    return desc;
}

id<MTLTensor> mpp_tensor_from_device(id<MTLDevice> device, MTLTensorDescriptor* desc) {
    NSError* error = nil;
    id<MTLTensor> tensor = [device newTensorWithDescriptor:desc error:&error];
    if (tensor == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"SuperSonic MPP pilot tensor creation failed: %@", error);
    }
    return tensor;
}

void mpp_tensor_replace_all_f16(id<MTLTensor> tensor, const uint16_t* values, NSUInteger dim0, NSUInteger dim1) {
    NSInteger origin_values[2] = {0, 0};
    NSInteger dim_values[2] = {
        static_cast<NSInteger>(dim0),
        static_cast<NSInteger>(dim1),
    };
    NSInteger stride_values[2] = {
        1,
        static_cast<NSInteger>(dim0),
    };
    MTLTensorExtents* origin = [[MTLTensorExtents alloc] initWithRank:2 values:origin_values];
    MTLTensorExtents* dims = [[MTLTensorExtents alloc] initWithRank:2 values:dim_values];
    MTLTensorExtents* strides = [[MTLTensorExtents alloc] initWithRank:2 values:stride_values];
    [tensor replaceSliceOrigin:origin sliceDimensions:dims withBytes:values strides:strides];
}

float mpp_tensor_first_f32(id<MTLTensor> tensor) {
    NSInteger origin_values[2] = {0, 0};
    NSInteger dim_values[2] = {1, 1};
    NSInteger stride_values[2] = {1, 1};
    MTLTensorExtents* origin = [[MTLTensorExtents alloc] initWithRank:2 values:origin_values];
    MTLTensorExtents* dims = [[MTLTensorExtents alloc] initWithRank:2 values:dim_values];
    MTLTensorExtents* strides = [[MTLTensorExtents alloc] initWithRank:2 values:stride_values];
    float value = 0.0f;
    [tensor getBytes:&value strides:strides fromSliceOrigin:origin sliceDimensions:dims];
    return value;
}

id<MTL4ArgumentTable> mpp_mtl4_argument_table(id<MTLDevice> device, NSUInteger buffer_bind_count) {
    MTL4ArgumentTableDescriptor* desc = [[MTL4ArgumentTableDescriptor alloc] init];
    desc.maxBufferBindCount = buffer_bind_count;
    desc.initializeBindings = YES;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:desc error:&error];
    if (table == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"SuperSonic MPP pilot argument table failed: %@", error);
    }
    return table;
}

bool mpp_wait_for_mtl4_queue(id<MTL4CommandQueue> queue, id<MTLSharedEvent> event, uint64_t value) {
    [queue signalEvent:event value:value];
    return [event waitUntilSignaledValue:value timeoutMS:30000];
}

bool mpp_encode_gemm_mtl4(
    id<MTLDevice> device,
    id<MTL4CommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    id<MTL4ArgumentTable> args,
    uint32_t tg_x,
    uint32_t tg_y,
    NSUInteger threads_per_threadgroup,
    id<MTLSharedEvent> event,
    uint64_t signal_value
) {
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> command_buffer = [device newCommandBuffer];
    if (allocator == nil || command_buffer == nil) {
        return false;
    }
    [command_buffer beginCommandBufferWithAllocator:allocator];
    id<MTL4ComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        [command_buffer endCommandBuffer];
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:args];
    [encoder dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
    [encoder endEncoding];
    [command_buffer endCommandBuffer];
    id<MTL4CommandBuffer> buffers[1] = { command_buffer };
    [queue commit:buffers count:1];
    return mpp_wait_for_mtl4_queue(queue, event, signal_value);
}
#endif  // SUPERSONIC_HAVE_MTL4_MPP

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

// SIMD-group INT4 GEMV (M=1, batch=1). Same shape as the BF16 GEMV but
// dequants nibbles on the fly. Each SIMD-group (32 threads) handles one
// output column; lanes stride through K with `kk += 32`, dequant via
// `bf16_round_rne(nibble * scale - zero * scale)`, then `simd_sum`.
// Replaces the per-cell sequential-K kernel from Step 3 for the M=1
// decode case — every projection in `metal_v2_decode_step` hits it,
// not just lm_head.
id<MTLComputePipelineState> matmul_pipeline_int4_bf16_gemv_m1(NSError** error_out) {
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
                                                   code:701
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"INT4GEMV(
#include <metal_stdlib>
using namespace metal;

struct MatmulInt4GemvParams {
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

kernel void supersonic_matmul_rhs_transposed_int4_bf16_gemv_m1(
    device const bfloat* lhs [[buffer(0)]],
    device const uchar* rhs_int4 [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device const bfloat* zero [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant MatmulInt4GemvParams& params [[buffer(5)]],
    uint col [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (col >= params.n) {
        return;
    }
    uint k = params.k;
    uint k_packed = k / 2u;
    uint group_size = params.group_size;
    uint scale_cols = (k + group_size - 1u) / group_size;
    uint scale_row = col / group_size;
    uint rhs_base = col * k_packed;

    float partial = 0.0f;
    for (uint kk = lane; kk < k; kk += 32u) {
        uint byte_idx = kk >> 1u;
        uint packed_byte = uint(rhs_int4[rhs_base + byte_idx]);
        uint nibble = (kk & 1u) != 0u ? ((packed_byte >> 4u) & 0xFu) : (packed_byte & 0xFu);
        uint sc_idx = scale_row * scale_cols + (kk / group_size);
        float s = float(scale[sc_idx]);
        float z = float(zero[sc_idx]);
        float w = bf16_round_rne_finite(float(nibble) * s - z * s);
        partial += float(lhs[kk]) * w;
    }
    float sum = simd_sum(partial);
    if (lane == 0) {
        out[col] = bfloat(sum);
    }
}
)INT4GEMV";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:702
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile int4 gemv library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_int4_bf16_gemv_m1"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:703
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load int4 gemv function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:704
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create int4 gemv pipeline"
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

// Tiled SIMD-group INT4 GEMV. Each threadgroup handles 32 output columns.
// All 1024 threads cooperate to load `lhs` (K elements) into threadgroup
// memory once, then 32 SIMD-groups each compute one output column from that
// shared cache (with on-the-fly nibble dequant). Used for the M=1 BF16 INT4
// path when K * 4 bytes fits in threadgroup memory (~K <= 4096).
id<MTLComputePipelineState> matmul_pipeline_int4_bf16_gemv_m1_tiled(NSError** error_out) {
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
                                                   code:801
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"INT4GEMVTILED(
#include <metal_stdlib>
using namespace metal;

struct MatmulInt4GemvParams {
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

kernel void supersonic_matmul_rhs_transposed_int4_bf16_gemv_m1_tiled(
    device const bfloat* lhs [[buffer(0)]],
    device const uchar* rhs_int4 [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device const bfloat* zero [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant MatmulInt4GemvParams& params [[buffer(5)]],
    threadgroup float* shared_lhs [[threadgroup(0)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint k = params.k;
    uint k_packed = k / 2u;
    uint group_size = params.group_size;
    uint scale_cols = (k + group_size - 1u) / group_size;

    // Cooperative load of lhs[0..K] into shared memory.
    for (uint i = thread_id; i < k; i += 1024u) {
        shared_lhs[i] = float(lhs[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 32 SIMD-groups × 32 cols per threadgroup.
    uint col = tg_id * 32u + simd_id;
    if (col >= params.n) {
        return;
    }
    uint scale_row = col / group_size;
    uint rhs_base = col * k_packed;
    float partial = 0.0f;
    for (uint kk = simd_lane; kk < k; kk += 32u) {
        uint byte_idx = kk >> 1u;
        uint packed_byte = uint(rhs_int4[rhs_base + byte_idx]);
        uint nibble = (kk & 1u) != 0u ? ((packed_byte >> 4u) & 0xFu) : (packed_byte & 0xFu);
        uint sc_idx = scale_row * scale_cols + (kk / group_size);
        float s = float(scale[sc_idx]);
        float z = float(zero[sc_idx]);
        float w = bf16_round_rne_finite(float(nibble) * s - z * s);
        partial += shared_lhs[kk] * w;
    }
    float sum = simd_sum(partial);
    if (simd_lane == 0) {
        out[col] = bfloat(sum);
    }
}
)INT4GEMVTILED";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:802
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile int4 gemv tiled library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_matmul_rhs_transposed_int4_bf16_gemv_m1_tiled"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:803
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load int4 gemv tiled function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:804
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create int4 gemv tiled pipeline"
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

struct Qwen36LinearInt4Pipelines {
    __strong id<MTLComputePipelineState> input_norm = nil;
    __strong id<MTLComputePipelineState> projections = nil;
    __strong id<MTLComputePipelineState> conv_silu_state = nil;
    __strong id<MTLComputePipelineState> qk_norm_repeat = nil;
    __strong id<MTLComputePipelineState> recurrent_update = nil;
    __strong id<MTLComputePipelineState> output_gate_norm = nil;
    __strong id<MTLComputePipelineState> out_proj_finalize = nil;
};

Qwen36LinearInt4Pipelines qwen36_linear_int4_pipelines(NSError** error_out) {
    static std::mutex mutex;
    static bool attempted = false;
    static Qwen36LinearInt4Pipelines pipelines;
    static __strong NSError* build_error = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:1100
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QWEN36LINEAR(
#include <metal_stdlib>
using namespace metal;

struct Qwen36LinearInt4Params {
    uint hidden;
    uint num_k_heads;
    uint num_v_heads;
    uint head_k_dim;
    uint head_v_dim;
    uint conv_kernel_dim;
    uint group_size;
    uint key_dim;
    uint val_dim;
    uint qkv_dim;
    uint kstate;
    uint has_conv1d_bias;
    uint off_qkv_raw;
    uint off_z_raw;
    uint off_a_raw;
    uint off_b_raw;
    uint off_q_normed;
    uint off_k_normed;
    uint off_q_rep;
    uint off_k_rep;
    uint off_beta;
    uint off_g;
    uint off_rec_out;
    float rms_norm_eps;
    float q_scale;
};

inline float bf16_round_rne_finite(float x) {
    uint bits = as_type<uint>(x);
    uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    return as_type<float>(bits);
}

inline float silu(float x) {
    return x * (1.0f / (1.0f + exp(-x)));
}

inline float int4_weight_2d(
    device const uchar* packed,
    device const bfloat* scale,
    device const bfloat* zero,
    uint row,
    uint col,
    uint cols,
    uint group_size
) {
    uint byte_cols = (cols + 1u) / 2u;
    uint scale_cols = (cols + group_size - 1u) / group_size;
    uint packed_byte = uint(packed[row * byte_cols + (col >> 1u)]);
    uint nibble = (col & 1u) != 0u ? ((packed_byte >> 4u) & 0xFu) : (packed_byte & 0xFu);
    uint scale_idx = (row / group_size) * scale_cols + (col / group_size);
    float s = float(scale[scale_idx]);
    float z = float(zero[scale_idx]);
    return bf16_round_rne_finite(float(nibble) * s - z * s);
}

inline float int4_weight_pair_dot_2d(
    device const uchar* packed,
    device const bfloat* scale,
    device const bfloat* zero,
    uint row,
    uint byte_col,
    uint cols,
    uint group_size,
    float x0,
    float x1
) {
    uint col = byte_col << 1u;
    uint byte_cols = (cols + 1u) / 2u;
    uint packed_byte = uint(packed[row * byte_cols + byte_col]);
    uint scale_cols = (cols + group_size - 1u) / group_size;
    uint scale_idx = (row / group_size) * scale_cols + (col / group_size);
    float s = float(scale[scale_idx]);
    float z = float(zero[scale_idx]);
    float w0 = bf16_round_rne_finite(float(packed_byte & 0xFu) * s - z * s);
    float w1 = bf16_round_rne_finite(float((packed_byte >> 4u) & 0xFu) * s - z * s);
    return w0 * x0 + w1 * x1;
}

kernel void supersonic_qwen36_linear_input_norm(
    device const bfloat* input_hidden [[buffer(0)]],
    device const bfloat* input_norm_w [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant Qwen36LinearInt4Params& params [[buffer(3)]],
    threadgroup float* partials [[threadgroup(0)]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    float partial = 0.0f;
    for (uint col = thread_id; col < params.hidden; col += 1024u) {
        float v = float(input_hidden[col]);
        partial += v * v;
    }
    float simd_partial = simd_sum(partial);
    if (simd_lane == 0) {
        partials[simd_id] = simd_partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_id == 0) {
        float x = simd_lane < 32u ? partials[simd_lane] : 0.0f;
        float sum = simd_sum(x);
        if (simd_lane == 0) {
            partials[0] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = partials[0];
    float inv_rms = 1.0f / sqrt(total / float(params.hidden) + params.rms_norm_eps);
    for (uint col = thread_id; col < params.hidden; col += 1024u) {
        float v = float(input_hidden[col]);
        float w = float(input_norm_w[col]);
        output[col] = bfloat(bf16_round_rne_finite(v * inv_rms * (1.0f + w)));
    }
}

kernel void supersonic_qwen36_linear_projections(
    device const bfloat* x_norm [[buffer(0)]],
    device const uchar* in_proj_qkv [[buffer(1)]],
    device const bfloat* in_proj_qkv_scale [[buffer(2)]],
    device const bfloat* in_proj_qkv_zero [[buffer(3)]],
    device const uchar* in_proj_z [[buffer(4)]],
    device const bfloat* in_proj_z_scale [[buffer(5)]],
    device const bfloat* in_proj_z_zero [[buffer(6)]],
    device const bfloat* in_proj_a [[buffer(7)]],
    device const bfloat* in_proj_b [[buffer(8)]],
    device float* workspace [[buffer(9)]],
    constant Qwen36LinearInt4Params& params [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint ab_base = params.qkv_dim + params.val_dim;
    uint total_rows = ab_base + 2u * params.num_v_heads;
    if (row >= total_rows) {
        return;
    }
    float partial = 0.0f;
    if (row < params.qkv_dim) {
        uint byte_cols = (params.hidden + 1u) / 2u;
        for (uint byte_col = lane; byte_col < byte_cols; byte_col += 32u) {
            uint col = byte_col << 1u;
            float x1 = (col + 1u) < params.hidden ? float(x_norm[col + 1u]) : 0.0f;
            partial += int4_weight_pair_dot_2d(
                in_proj_qkv, in_proj_qkv_scale, in_proj_qkv_zero,
                row, byte_col, params.hidden, params.group_size,
                float(x_norm[col]), x1
            );
        }
        float acc = simd_sum(partial);
        if (lane == 0) {
            workspace[params.off_qkv_raw + row] = bf16_round_rne_finite(acc);
        }
    } else if (row < ab_base) {
        uint z_row = row - params.qkv_dim;
        uint byte_cols = (params.hidden + 1u) / 2u;
        for (uint byte_col = lane; byte_col < byte_cols; byte_col += 32u) {
            uint col = byte_col << 1u;
            float x1 = (col + 1u) < params.hidden ? float(x_norm[col + 1u]) : 0.0f;
            partial += int4_weight_pair_dot_2d(
                in_proj_z, in_proj_z_scale, in_proj_z_zero,
                z_row, byte_col, params.hidden, params.group_size,
                float(x_norm[col]), x1
            );
        }
        float acc = simd_sum(partial);
        if (lane == 0) {
            workspace[params.off_z_raw + z_row] = bf16_round_rne_finite(acc);
        }
    } else {
        uint ab_row = row - ab_base;
        bool is_b = ab_row >= params.num_v_heads;
        uint head = is_b ? ab_row - params.num_v_heads : ab_row;
        for (uint col = lane; col < params.hidden; col += 32u) {
            uint idx = head * params.hidden + col;
            float w = is_b ? float(in_proj_b[idx]) : float(in_proj_a[idx]);
            partial += w * float(x_norm[col]);
        }
        float acc = simd_sum(partial);
        if (lane == 0) {
            uint out_off = is_b ? params.off_b_raw + head : params.off_a_raw + head;
            workspace[out_off] = bf16_round_rne_finite(acc);
        }
    }
}

kernel void supersonic_qwen36_linear_conv_silu_state(
    device float* workspace [[buffer(0)]],
    device const bfloat* conv1d_w [[buffer(1)]],
    device const bfloat* conv1d_bias [[buffer(2)]],
    device bfloat* conv_state [[buffer(3)]],
    constant Qwen36LinearInt4Params& params [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    if (ch >= params.qkv_dim) {
        return;
    }
    float new_qkv = workspace[params.off_qkv_raw + ch];
    float acc = 0.0f;
    for (uint t = 0; t < params.kstate; ++t) {
        acc += float(conv_state[ch * params.kstate + t]) *
               float(conv1d_w[ch * params.conv_kernel_dim + t]);
    }
    acc += new_qkv * float(conv1d_w[ch * params.conv_kernel_dim + params.kstate]);
    if (params.has_conv1d_bias != 0u) {
        acc += float(conv1d_bias[ch]);
    }
    float conv_out = bf16_round_rne_finite(acc);
    workspace[params.off_qkv_raw + ch] = bf16_round_rne_finite(silu(conv_out));

    if (params.kstate > 0u) {
        for (uint t = 0; t + 1u < params.kstate; ++t) {
            conv_state[ch * params.kstate + t] = conv_state[ch * params.kstate + t + 1u];
        }
        conv_state[ch * params.kstate + (params.kstate - 1u)] = bfloat(new_qkv);
    }
}

kernel void supersonic_qwen36_linear_qk_norm_repeat(
    device float* workspace [[buffer(0)]],
    constant Qwen36LinearInt4Params& params [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (head >= params.num_k_heads) {
        return;
    }
    uint q_src = params.off_qkv_raw + head * params.head_k_dim;
    uint k_src = params.off_qkv_raw + params.key_dim + head * params.head_k_dim;
    float q_partial = 0.0f;
    float k_partial = 0.0f;
    for (uint i = lane; i < params.head_k_dim; i += 32u) {
        float q = workspace[q_src + i];
        float k = workspace[k_src + i];
        q_partial += q * q;
        k_partial += k * k;
    }
    float q_ss = simd_sum(q_partial);
    float k_ss = simd_sum(k_partial);
    float q_denom = bf16_round_rne_finite(max(bf16_round_rne_finite(sqrt(q_ss)), 1.0e-6f));
    float k_denom = bf16_round_rne_finite(max(bf16_round_rne_finite(sqrt(k_ss)), 1.0e-6f));
    uint repeat = params.num_v_heads / params.num_k_heads;
    for (uint i = lane; i < params.head_k_dim; i += 32u) {
        float qn = bf16_round_rne_finite(workspace[q_src + i] / q_denom);
        float kn = bf16_round_rne_finite(workspace[k_src + i] / k_denom);
        float qs = bf16_round_rne_finite(qn * params.q_scale);
        for (uint r = 0; r < repeat; ++r) {
            uint vhead = head * repeat + r;
            workspace[params.off_q_rep + vhead * params.head_k_dim + i] = qs;
            workspace[params.off_k_rep + vhead * params.head_k_dim + i] = kn;
        }
    }
}

kernel void supersonic_qwen36_linear_recurrent_update(
    device float* workspace [[buffer(0)]],
    device const bfloat* dt_bias [[buffer(1)]],
    device const bfloat* a_log [[buffer(2)]],
    device float* recurrent_state [[buffer(3)]],
    constant Qwen36LinearInt4Params& params [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint lane = tid.x;
    uint j = tg.x;
    uint head = tg.y;
    if (head >= params.num_v_heads || j >= params.head_v_dim) {
        return;
    }
    float a_v = workspace[params.off_a_raw + head];
    float b_v = workspace[params.off_b_raw + head];
    float softplus = log(1.0f + exp(a_v + float(dt_bias[head])));
    float beta = 1.0f / (1.0f + exp(-b_v));
    float gstep = exp(-softplus * exp(float(a_log[head])));
    if (j == 0 && lane == 0) {
        workspace[params.off_beta + head] = beta;
        workspace[params.off_g + head] = -softplus * exp(float(a_log[head]));
    }

    uint state_base = head * params.head_k_dim * params.head_v_dim;
    uint k_base = params.off_k_rep + head * params.head_k_dim;
    uint q_base = params.off_q_rep + head * params.head_k_dim;
    uint v_base = params.off_qkv_raw + 2u * params.key_dim + head * params.head_v_dim;
    float kv_partial = 0.0f;
    for (uint i = lane; i < params.head_k_dim; i += 32u) {
        uint state_idx = state_base + i * params.head_v_dim + j;
        recurrent_state[state_idx] *= gstep;
        kv_partial += recurrent_state[state_idx] * workspace[k_base + i];
    }
    float kv_mem = simd_sum(kv_partial);
    float delta = (workspace[v_base + j] - kv_mem) * beta;
    float rec_partial = 0.0f;
    for (uint i = lane; i < params.head_k_dim; i += 32u) {
        uint state_idx = state_base + i * params.head_v_dim + j;
        float updated = recurrent_state[state_idx] + workspace[k_base + i] * delta;
        recurrent_state[state_idx] = updated;
        rec_partial += updated * workspace[q_base + i];
    }
    float rec = simd_sum(rec_partial);
    if (lane == 0) {
        workspace[params.off_rec_out + head * params.head_v_dim + j] =
            bf16_round_rne_finite(rec);
    }
}

kernel void supersonic_qwen36_linear_output_gate_norm(
    device float* workspace [[buffer(0)]],
    device const bfloat* norm_w [[buffer(1)]],
    constant Qwen36LinearInt4Params& params [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (head >= params.num_v_heads) {
        return;
    }
    uint rec_base = params.off_rec_out + head * params.head_v_dim;
    uint z_base = params.off_z_raw + head * params.head_v_dim;
    float partial = 0.0f;
    for (uint j = lane; j < params.head_v_dim; j += 32u) {
        float v = workspace[rec_base + j];
        partial += v * v;
    }
    float mean_sq = simd_sum(partial);
    float inv = 1.0f / sqrt(mean_sq / float(params.head_v_dim) + params.rms_norm_eps);
    for (uint j = lane; j < params.head_v_dim; j += 32u) {
        float rec = workspace[rec_base + j];
        float on = bf16_round_rne_finite(rec * inv * float(norm_w[j]));
        float z = workspace[z_base + j];
        float z_silu = bf16_round_rne_finite(silu(z));
        workspace[rec_base + j] = bf16_round_rne_finite(on * z_silu);
    }
}

kernel void supersonic_qwen36_linear_out_proj_finalize(
    device const bfloat* input_hidden [[buffer(0)]],
    device const uchar* out_proj [[buffer(1)]],
    device const bfloat* out_proj_scale [[buffer(2)]],
    device const bfloat* out_proj_zero [[buffer(3)]],
    device float* workspace [[buffer(4)]],
    device bfloat* output [[buffer(5)]],
    constant Qwen36LinearInt4Params& params [[buffer(6)]],
    device bfloat* final_output [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= params.hidden) {
        return;
    }
    float partial = 0.0f;
    uint byte_cols = (params.val_dim + 1u) / 2u;
    for (uint byte_col = lane; byte_col < byte_cols; byte_col += 32u) {
        uint col = byte_col << 1u;
        float x1 = (col + 1u) < params.val_dim ? workspace[params.off_rec_out + col + 1u] : 0.0f;
        partial += int4_weight_pair_dot_2d(
            out_proj, out_proj_scale, out_proj_zero,
            row, byte_col, params.val_dim, params.group_size,
            workspace[params.off_rec_out + col],
            x1
        );
    }
    float acc = simd_sum(partial);
    if (lane == 0) {
        float o_out = bf16_round_rne_finite(acc);
        float residual = bf16_round_rne_finite(float(input_hidden[row]) + o_out);
        final_output[row] = bfloat(residual);
    }
}
)QWEN36LINEAR";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:1101
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile qwen36 linear int4 library"
                                                                   }];
                } else {
                    NSArray<NSString*>* names = @[
                        @"supersonic_qwen36_linear_input_norm",
                        @"supersonic_qwen36_linear_projections",
                        @"supersonic_qwen36_linear_conv_silu_state",
                        @"supersonic_qwen36_linear_qk_norm_repeat",
                        @"supersonic_qwen36_linear_recurrent_update",
                        @"supersonic_qwen36_linear_output_gate_norm",
                        @"supersonic_qwen36_linear_out_proj_finalize",
                    ];
                    for (NSString* name in names) {
                        id<MTLFunction> function = [library newFunctionWithName:name];
                        if (function == nil) {
                            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                               code:1102
                                                           userInfo:@{
                                                               NSLocalizedDescriptionKey :
                                                                   [NSString stringWithFormat:@"Failed to load %@", name]
                                                           }];
                            break;
                        }
                        NSError* pipeline_error = nil;
                        id<MTLComputePipelineState> pipeline =
                            [device newComputePipelineStateWithFunction:function error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:1103
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     [NSString stringWithFormat:@"Failed to create %@", name]
                                                                             }];
                            break;
                        }
                        if ([name isEqualToString:@"supersonic_qwen36_linear_input_norm"]) {
                            pipelines.input_norm = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_linear_projections"]) {
                            pipelines.projections = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_linear_conv_silu_state"]) {
                            pipelines.conv_silu_state = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_linear_qk_norm_repeat"]) {
                            pipelines.qk_norm_repeat = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_linear_recurrent_update"]) {
                            pipelines.recurrent_update = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_linear_output_gate_norm"]) {
                            pipelines.output_gate_norm = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_linear_out_proj_finalize"]) {
                            pipelines.out_proj_finalize = pipeline;
                        }
                    }
                }
            }
        }
    }

    if ((pipelines.input_norm == nil || pipelines.projections == nil ||
         pipelines.conv_silu_state == nil || pipelines.qk_norm_repeat == nil ||
         pipelines.recurrent_update == nil || pipelines.output_gate_norm == nil ||
         pipelines.out_proj_finalize == nil) &&
        error_out != nullptr) {
        *error_out = build_error;
    }
    if (build_error != nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE"] != nil) {
        NSLog(@"SuperSonic Qwen3.6 linear INT4 Metal pipeline error: %@", build_error);
    }
    return pipelines;
}

struct Qwen36FfnInt4Pipelines {
    __strong id<MTLComputePipelineState> router_topk = nil;
    __strong id<MTLComputePipelineState> router_stage5 = nil;
    __strong id<MTLComputePipelineState> shared_gate_up = nil;
    __strong id<MTLComputePipelineState> shared_scalar = nil;
    __strong id<MTLComputePipelineState> shared_down = nil;
    __strong id<MTLComputePipelineState> expert_gate_up = nil;
    __strong id<MTLComputePipelineState> expert_gate_up_tiled = nil;
    __strong id<MTLComputePipelineState> expert_down_finalize = nil;
    __strong id<MTLComputePipelineState> expert_down_finalize_tiled = nil;
    __strong id<MTLComputePipelineState> batched_expert_gate_up_tiled = nil;
    __strong id<MTLComputePipelineState> batched_expert_down_combine_tiled = nil;
    __strong id<MTLComputePipelineState> expert_pack_u8 = nil;
    __strong id<MTLComputePipelineState> expert_pack_bf16_pair = nil;
    __strong id<MTLComputePipelineState> expert_pack_remap_topk = nil;
    __strong id<MTLComputePipelineState> expert_mps_silu = nil;
    __strong id<MTLComputePipelineState> expert_mps_finalize = nil;
    __strong id<MTLComputePipelineState> expert_mps_transcode_hnorm = nil;
    __strong id<MTLComputePipelineState> expert_mps_transcode_gate_up = nil;
    __strong id<MTLComputePipelineState> expert_mps_transcode_down = nil;
};

Qwen36FfnInt4Pipelines qwen36_ffn_int4_pipelines(NSError** error_out) {
    static std::mutex mutex;
    static bool attempted = false;
    static Qwen36FfnInt4Pipelines pipelines;
    static __strong NSError* build_error = nil;

    std::lock_guard<std::mutex> lock(mutex);
    if (!attempted) {
        attempted = true;
        @autoreleasepool {
            id<MTLDevice> device = metal_device();
            if (device == nil) {
                build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                   code:950
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"QWEN36FFN(
#include <metal_stdlib>
using namespace metal;

struct Qwen36FfnInt4Params {
    uint hidden;
    uint num_experts;
    uint moe_intermediate;
    uint shared_intermediate;
    uint top_k;
    uint group_size;
    uint off_h_norm;
    uint off_topk_val;
    uint off_topk_idx;
    uint off_sg_scalar;
    uint off_shared_mid;
    uint off_shared_out;
    uint off_expert_mid;
    uint off_moe_out;
};

struct Qwen36FfnExpertGateUpTiledParams {
    uint hidden;
    uint moe_intermediate;
    uint top_k;
    uint group_size;
    uint off_h_norm;
    uint off_topk_idx;
    uint off_expert_mid;
};

struct Qwen36FfnExpertPackParams {
    uint rows;
    uint cols;
    uint top_k;
    uint off_topk_idx;
};

struct Qwen36BatchedFfnExpertParams {
    uint n_tokens;
    uint top_k;
    uint hidden;
    uint moe_intermediate;
    uint group_size;
};

struct Qwen36RouterTopkParams {
    uint n_tokens;
    uint num_experts;
    uint top_k;
};

inline float bf16_round_rne_finite(float x) {
    uint bits = as_type<uint>(x);
    uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    return as_type<float>(bits);
}

inline float silu(float x) {
    return x * (1.0f / (1.0f + exp(-x)));
}

inline float int4_weight_2d(
    device const uchar* packed,
    device const bfloat* scale,
    device const bfloat* zero,
    uint row,
    uint col,
    uint cols,
    uint group_size
) {
    uint byte_cols = (cols + 1u) / 2u;
    uint scale_cols = (cols + group_size - 1u) / group_size;
    uint packed_byte = uint(packed[row * byte_cols + (col >> 1u)]);
    uint nibble = (col & 1u) != 0u ? ((packed_byte >> 4u) & 0xFu) : (packed_byte & 0xFu);
    uint scale_idx = (row / group_size) * scale_cols + (col / group_size);
    float s = float(scale[scale_idx]);
    float z = float(zero[scale_idx]);
    return bf16_round_rne_finite(float(nibble) * s - z * s);
}

inline float int4_dot_2d_host_order(
    device const uchar* packed,
    device const bfloat* scale,
    device const bfloat* zero,
    device const float* x,
    uint row,
    uint cols,
    uint group_size
) {
    float acc = 0.0f;
    uint byte_cols = (cols + 1u) / 2u;
    uint scale_cols = (cols + group_size - 1u) / group_size;
    uint packed_base = row * byte_cols;
    uint scale_base = (row / group_size) * scale_cols;
    for (uint scale_col = 0u; scale_col < scale_cols; ++scale_col) {
        uint group_start = scale_col * group_size;
        uint group_end = min(cols, group_start + group_size);
        uint scale_idx = scale_base + scale_col;
        float s = float(scale[scale_idx]);
        float z = float(zero[scale_idx]);
        float zs = z * s;
        uint col = group_start;
        uint byte_idx = packed_base + group_start / 2u;
        while (col + 1u < group_end) {
            uint packed_byte = uint(packed[byte_idx]);
            float w0 = bf16_round_rne_finite(float(packed_byte & 0xFu) * s - zs);
            float w1 = bf16_round_rne_finite(float((packed_byte >> 4u) & 0xFu) * s - zs);
            acc += w0 * x[col] + w1 * x[col + 1u];
            col += 2u;
            byte_idx += 1u;
        }
        if (col < group_end) {
            uint packed_byte = uint(packed[byte_idx]);
            float w = bf16_round_rne_finite(float(packed_byte & 0xFu) * s - zs);
            acc += w * x[col];
        }
    }
    return acc;
}

inline float int4_weight_expert(
    device const uchar* packed,
    device const bfloat* scale,
    device const bfloat* zero,
    uint expert,
    uint row,
    uint rows,
    uint col,
    uint cols,
    uint group_size
) {
    uint byte_cols = (cols + 1u) / 2u;
    uint scale_rows = (rows + group_size - 1u) / group_size;
    uint scale_cols = (cols + group_size - 1u) / group_size;
    uint packed_base = (expert * rows + row) * byte_cols;
    uint scale_base = (expert * scale_rows + (row / group_size)) * scale_cols;
    uint packed_byte = uint(packed[packed_base + (col >> 1u)]);
    uint nibble = (col & 1u) != 0u ? ((packed_byte >> 4u) & 0xFu) : (packed_byte & 0xFu);
    uint scale_idx = scale_base + (col / group_size);
    float s = float(scale[scale_idx]);
    float z = float(zero[scale_idx]);
    return bf16_round_rne_finite(float(nibble) * s - z * s);
}

kernel void supersonic_qwen36_router_softmax_topk_bf16(
    device const bfloat* logits [[buffer(0)]],
    device uint* topk_idx [[buffer(1)]],
    device bfloat* topk_weight [[buffer(2)]],
    constant Qwen36RouterTopkParams& params [[buffer(3)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (token >= params.n_tokens || params.num_experts > 256u || params.top_k > 16u) {
        return;
    }

    float v = -INFINITY;
    if (tid < params.num_experts) {
        v = float(logits[token * params.num_experts + tid]);
    }
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];

    float e = (tid < params.num_experts) ? exp(v - row_max) : 0.0f;
    scratch[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / scratch[0];

    scratch[tid] = (tid < params.num_experts) ? bf16_round_rne_finite(e * inv_sum) : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float sum_k = 0.0f;
        for (uint k = 0u; k < params.top_k; ++k) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < params.num_experts; ++expert) {
                float prob = scratch[expert];
                if (prob > best_val || (prob == best_val && expert < best_idx)) {
                    best_val = prob;
                    best_idx = expert;
                }
            }
            scratch[256u + k] = best_val;
            topk_idx[token * params.top_k + k] = best_idx;
            sum_k += best_val;
            scratch[best_idx] = -INFINITY;
        }

        float inv_k = 1.0f / sum_k;
        for (uint k = 0u; k < params.top_k; ++k) {
            topk_weight[token * params.top_k + k] =
                bfloat(bf16_round_rne_finite(scratch[256u + k] * inv_k));
        }
    }
}

kernel void supersonic_qwen36_ffn_router_stage5(
    device const bfloat* input_hidden [[buffer(0)]],
    device const bfloat* post_attn_norm [[buffer(1)]],
    device const bfloat* gate [[buffer(2)]],
    device float* workspace [[buffer(3)]],
    device uint* output_idx [[buffer(4)]],
    constant Qwen36FfnInt4Params& params [[buffer(5)]],
    constant float& rms_norm_eps [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (params.hidden > 2048u || params.num_experts > 256u || params.top_k > 16u) {
        return;
    }

    if (tid == 0u) {
        float mean_sq = 0.0f;
        for (uint col = 0u; col < params.hidden; ++col) {
            float v = float(input_hidden[col]);
            mean_sq += v * v;
        }
        float inv_rms = rsqrt((mean_sq / float(params.hidden)) + rms_norm_eps);
        for (uint col = 0u; col < params.hidden; ++col) {
            workspace[params.off_h_norm + col] = bf16_round_rne_finite(
                float(input_hidden[col]) * inv_rms * (1.0f + float(post_attn_norm[col]))
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float logit = -INFINITY;
    if (tid < params.num_experts) {
        float acc = 0.0f;
        uint row_base = tid * params.hidden;
        for (uint col = 0u; col < params.hidden; ++col) {
            acc += float(gate[row_base + col]) * workspace[params.off_h_norm + col];
        }
        logit = bf16_round_rne_finite(acc);
        workspace[params.hidden + tid] = logit;
    }
    scratch[tid] = logit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];

    float e = (tid < params.num_experts) ? exp(logit - row_max) : 0.0f;
    scratch[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / scratch[0];

    scratch[tid] = (tid < params.num_experts) ? bf16_round_rne_finite(e * inv_sum) : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float sum_k = 0.0f;
        for (uint k = 0u; k < params.top_k; ++k) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < params.num_experts; ++expert) {
                float prob = scratch[expert];
                if (prob > best_val || (prob == best_val && expert < best_idx)) {
                    best_val = prob;
                    best_idx = expert;
                }
            }
            scratch[256u + k] = best_val;
            output_idx[k] = best_idx;
            sum_k += best_val;
            scratch[best_idx] = -INFINITY;
            workspace[params.off_topk_idx + k] = as_type<float>(best_idx);
        }

        float inv_k = 1.0f / sum_k;
        for (uint k = 0u; k < params.top_k; ++k) {
            workspace[params.off_topk_val + k] =
                bf16_round_rne_finite(scratch[256u + k] * inv_k);
        }
    }
}

kernel void supersonic_qwen36_ffn_shared_gate_up(
    device float* workspace [[buffer(0)]],
    device const uchar* shared_gate_proj [[buffer(1)]],
    device const bfloat* shared_gate_scale [[buffer(2)]],
    device const bfloat* shared_gate_zero [[buffer(3)]],
    device const uchar* shared_up_proj [[buffer(4)]],
    device const bfloat* shared_up_scale [[buffer(5)]],
    device const bfloat* shared_up_zero [[buffer(6)]],
    constant Qwen36FfnInt4Params& params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= params.shared_intermediate) {
        return;
    }
    if (lane == 0) {
        device const float* h_norm = workspace + params.off_h_norm;
        float gate = int4_dot_2d_host_order(
            shared_gate_proj, shared_gate_scale, shared_gate_zero,
            h_norm, row, params.hidden, params.group_size
        );
        float up = int4_dot_2d_host_order(
            shared_up_proj, shared_up_scale, shared_up_zero,
            h_norm, row, params.hidden, params.group_size
        );
        workspace[params.off_shared_mid + row] = silu(gate) * up;
    }
}

kernel void supersonic_qwen36_ffn_shared_scalar(
    device float* workspace [[buffer(0)]],
    device const bfloat* shared_expert_gate [[buffer(1)]],
    constant Qwen36FfnInt4Params& params [[buffer(2)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (lane == 0) {
        float sum = 0.0f;
        for (uint col = 0u; col < params.hidden; ++col) {
            sum += float(shared_expert_gate[col]) * workspace[params.off_h_norm + col];
        }
        workspace[params.off_sg_scalar] = 1.0f / (1.0f + exp(-sum));
    }
}

kernel void supersonic_qwen36_ffn_shared_down(
    device float* workspace [[buffer(0)]],
    device const uchar* shared_down_proj [[buffer(1)]],
    device const bfloat* shared_down_scale [[buffer(2)]],
    device const bfloat* shared_down_zero [[buffer(3)]],
    constant Qwen36FfnInt4Params& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= params.hidden) {
        return;
    }
    if (lane == 0) {
        device const float* shared_mid = workspace + params.off_shared_mid;
        float acc = int4_dot_2d_host_order(
            shared_down_proj, shared_down_scale, shared_down_zero,
            shared_mid, row, params.shared_intermediate, params.group_size
        );
        float gated = workspace[params.off_sg_scalar] * acc;
        workspace[params.off_shared_out + row] = bf16_round_rne_finite(gated);
    }
}

kernel void supersonic_qwen36_ffn_expert_gate_up(
    device float* workspace [[buffer(0)]],
    device const uchar* gate_up_proj [[buffer(1)]],
    device const bfloat* gate_up_scale [[buffer(2)]],
    device const bfloat* gate_up_zero [[buffer(3)]],
    constant Qwen36FfnInt4Params& params [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint row = tg.x;
    uint group = tg.y;
    uint lane = tid.x;
    if (row >= params.moe_intermediate || group >= params.top_k) {
        return;
    }
    uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (uint col = lane; col < params.hidden; col += 32u) {
        float x = workspace[params.off_h_norm + col];
        gate_partial += int4_weight_expert(
            gate_up_proj, gate_up_scale, gate_up_zero,
            expert, row, 2u * params.moe_intermediate, col, params.hidden, params.group_size
        ) * x;
        up_partial += int4_weight_expert(
            gate_up_proj, gate_up_scale, gate_up_zero,
            expert, params.moe_intermediate + row, 2u * params.moe_intermediate,
            col, params.hidden, params.group_size
        ) * x;
    }
    float gate = simd_sum(gate_partial);
    float up = simd_sum(up_partial);
    if (lane == 0) {
        workspace[params.off_expert_mid + group * params.moe_intermediate + row] =
            silu(gate) * up;
    }
}

kernel void supersonic_qwen36_ffn_expert_gate_up_tiled(
    device float* workspace [[buffer(0)]],
    device const uchar* gate_up_proj [[buffer(1)]],
    device const bfloat* gate_up_scale [[buffer(2)]],
    device const bfloat* gate_up_zero [[buffer(3)]],
    constant Qwen36FfnExpertGateUpTiledParams& params [[buffer(4)]],
    threadgroup float* partials [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tg.x;
    uint group = tg.y;
    if (row >= params.moe_intermediate || group >= params.top_k) {
        return;
    }
    uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
    uint rows = 2u * params.moe_intermediate;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (uint col = thread_id; col < params.hidden; col += 256u) {
        float x = workspace[params.off_h_norm + col];
        gate_partial += int4_weight_expert(
            gate_up_proj, gate_up_scale, gate_up_zero,
            expert, row, rows, col, params.hidden, params.group_size
        ) * x;
        up_partial += int4_weight_expert(
            gate_up_proj, gate_up_scale, gate_up_zero,
            expert, params.moe_intermediate + row, rows,
            col, params.hidden, params.group_size
        ) * x;
    }

    float gate_simd = simd_sum(gate_partial);
    float up_simd = simd_sum(up_partial);
    if (lane == 0u) {
        partials[simd_id] = gate_simd;
        partials[8u + simd_id] = up_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u) {
        float gate_in = lane < 8u ? partials[lane] : 0.0f;
        float up_in = lane < 8u ? partials[8u + lane] : 0.0f;
        float gate = simd_sum(gate_in);
        float up = simd_sum(up_in);
        if (lane == 0u) {
            workspace[params.off_expert_mid + group * params.moe_intermediate + row] =
                silu(gate) * up;
        }
    }
}

kernel void supersonic_qwen36_ffn_expert_down_finalize(
    device float* workspace [[buffer(0)]],
    device const bfloat* input_hidden [[buffer(1)]],
    device const uchar* down_proj [[buffer(2)]],
    device const bfloat* down_scale [[buffer(3)]],
    device const bfloat* down_zero [[buffer(4)]],
    device bfloat* output [[buffer(5)]],
    constant Qwen36FfnInt4Params& params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= params.hidden) {
        return;
    }
    float moe_acc = 0.0f;
    for (uint group = 0; group < params.top_k; ++group) {
        uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
        float partial = 0.0f;
        for (uint col = lane; col < params.moe_intermediate; col += 32u) {
            float w = int4_weight_expert(
                down_proj, down_scale, down_zero,
                expert, row, params.hidden, col, params.moe_intermediate, params.group_size
            );
            partial += w * workspace[params.off_expert_mid + group * params.moe_intermediate + col];
        }
        float down = simd_sum(partial);
        moe_acc += workspace[params.off_topk_val + group] * down;
    }
    if (lane == 0) {
        float moe = bf16_round_rne_finite(moe_acc);
        workspace[params.off_moe_out + row] = moe;
        float final = bf16_round_rne_finite(
            float(input_hidden[row]) + moe + workspace[params.off_shared_out + row]
        );
        output[row] = bfloat(final);
    }
}

kernel void supersonic_qwen36_ffn_expert_down_finalize_tiled(
    device float* workspace [[buffer(0)]],
    device const bfloat* input_hidden [[buffer(1)]],
    device const uchar* down_proj [[buffer(2)]],
    device const bfloat* down_scale [[buffer(3)]],
    device const bfloat* down_zero [[buffer(4)]],
    device bfloat* output [[buffer(5)]],
    constant Qwen36FfnInt4Params& params [[buffer(6)]],
    threadgroup float* partials [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    if (row >= params.hidden) {
        return;
    }
    float moe_acc = 0.0f;
    for (uint group = 0; group < params.top_k; ++group) {
        uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
        float partial = 0.0f;
        for (uint col = thread_id; col < params.moe_intermediate; col += 256u) {
            float w = int4_weight_expert(
                down_proj, down_scale, down_zero,
                expert, row, params.hidden, col, params.moe_intermediate, params.group_size
            );
            partial += w * workspace[params.off_expert_mid + group * params.moe_intermediate + col];
        }

        float simd_partial = simd_sum(partial);
        if (lane == 0u) {
            partials[simd_id] = simd_partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u) {
            float down_in = lane < 8u ? partials[lane] : 0.0f;
            float down = simd_sum(down_in);
            if (lane == 0u) {
                moe_acc += workspace[params.off_topk_val + group] * down;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_id == 0u && lane == 0u) {
        float moe = bf16_round_rne_finite(moe_acc);
        workspace[params.off_moe_out + row] = moe;
        float final = bf16_round_rne_finite(
            float(input_hidden[row]) + moe + workspace[params.off_shared_out + row]
        );
        output[row] = bfloat(final);
    }
}

kernel void supersonic_qwen36_batched_ffn_expert_gate_up_tiled(
    device const bfloat* h_norm [[buffer(0)]],
    device const uint* topk_idx [[buffer(1)]],
    device const uchar* gate_up_proj [[buffer(2)]],
    device const bfloat* gate_up_scale [[buffer(3)]],
    device const bfloat* gate_up_zero [[buffer(4)]],
    device float* expert_mid [[buffer(5)]],
    constant Qwen36BatchedFfnExpertParams& params [[buffer(6)]],
    threadgroup float* partials [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tg.x;
    uint group = tg.y;
    uint token = tg.z;
    if (row >= params.moe_intermediate ||
        group >= params.top_k ||
        token >= params.n_tokens) {
        return;
    }
    uint expert = topk_idx[token * params.top_k + group];
    uint rows = 2u * params.moe_intermediate;
    uint token_base = token * params.hidden;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (uint col = thread_id; col < params.hidden; col += 256u) {
        float x = bf16_round_rne_finite(float(h_norm[token_base + col]));
        gate_partial += int4_weight_expert(
            gate_up_proj, gate_up_scale, gate_up_zero,
            expert, row, rows, col, params.hidden, params.group_size
        ) * x;
        up_partial += int4_weight_expert(
            gate_up_proj, gate_up_scale, gate_up_zero,
            expert, params.moe_intermediate + row, rows,
            col, params.hidden, params.group_size
        ) * x;
    }

    float gate_simd = simd_sum(gate_partial);
    float up_simd = simd_sum(up_partial);
    if (lane == 0u) {
        partials[simd_id] = gate_simd;
        partials[8u + simd_id] = up_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u) {
        float gate_in = lane < 8u ? partials[lane] : 0.0f;
        float up_in = lane < 8u ? partials[8u + lane] : 0.0f;
        float gate = simd_sum(gate_in);
        float up = simd_sum(up_in);
        if (lane == 0u) {
            uint dst = (token * params.top_k + group) * params.moe_intermediate + row;
            expert_mid[dst] = bf16_round_rne_finite(silu(gate) * up);
        }
    }
}

kernel void supersonic_qwen36_batched_ffn_expert_down_combine_tiled(
    device const uint* topk_idx [[buffer(0)]],
    device const bfloat* topk_weight [[buffer(1)]],
    device const uchar* down_proj [[buffer(2)]],
    device const bfloat* down_scale [[buffer(3)]],
    device const bfloat* down_zero [[buffer(4)]],
    device const float* expert_mid [[buffer(5)]],
    device bfloat* combined [[buffer(6)]],
    constant Qwen36BatchedFfnExpertParams& params [[buffer(7)]],
    threadgroup float* partials [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tg.x;
    uint token = tg.y;
    if (row >= params.hidden || token >= params.n_tokens) {
        return;
    }

    float moe_acc = 0.0f;
    for (uint group = 0u; group < params.top_k; ++group) {
        uint expert = topk_idx[token * params.top_k + group];
        float partial = 0.0f;
        uint mid_base = (token * params.top_k + group) * params.moe_intermediate;
        for (uint col = thread_id; col < params.moe_intermediate; col += 256u) {
            float w = int4_weight_expert(
                down_proj, down_scale, down_zero,
                expert, row, params.hidden, col, params.moe_intermediate, params.group_size
            );
            partial += w * expert_mid[mid_base + col];
        }

        float simd_partial = simd_sum(partial);
        if (lane == 0u) {
            partials[simd_id] = simd_partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u) {
            float down_in = lane < 8u ? partials[lane] : 0.0f;
            float down = simd_sum(down_in);
            if (lane == 0u) {
                moe_acc += float(topk_weight[token * params.top_k + group]) * down;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_id == 0u) {
        combined[token * params.hidden + row] = bfloat(bf16_round_rne_finite(moe_acc));
    }
}

kernel void supersonic_qwen36_ffn_pack_active_u8(
    device const float* workspace [[buffer(0)]],
    device const uchar* src [[buffer(1)]],
    device uchar* dst [[buffer(2)]],
    constant Qwen36FfnExpertPackParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint elems_per_group = params.rows * params.cols;
    uint total = params.top_k * elems_per_group;
    if (gid >= total || elems_per_group == 0u) {
        return;
    }
    uint group = gid / elems_per_group;
    uint rem = gid - group * elems_per_group;
    uint row = rem / params.cols;
    uint col = rem - row * params.cols;
    uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
    dst[(group * params.rows + row) * params.cols + col] =
        src[(expert * params.rows + row) * params.cols + col];
}

kernel void supersonic_qwen36_ffn_pack_active_bf16_pair(
    device const float* workspace [[buffer(0)]],
    device const bfloat* scale_src [[buffer(1)]],
    device const bfloat* zero_src [[buffer(2)]],
    device bfloat* scale_dst [[buffer(3)]],
    device bfloat* zero_dst [[buffer(4)]],
    constant Qwen36FfnExpertPackParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint elems_per_group = params.rows * params.cols;
    uint total = params.top_k * elems_per_group;
    if (gid >= total || elems_per_group == 0u) {
        return;
    }
    uint group = gid / elems_per_group;
    uint rem = gid - group * elems_per_group;
    uint row = rem / params.cols;
    uint col = rem - row * params.cols;
    uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
    uint dst_idx = (group * params.rows + row) * params.cols + col;
    uint src_idx = (expert * params.rows + row) * params.cols + col;
    scale_dst[dst_idx] = scale_src[src_idx];
    zero_dst[dst_idx] = zero_src[src_idx];
}

kernel void supersonic_qwen36_ffn_pack_active_remap_topk(
    device float* workspace [[buffer(0)]],
    constant Qwen36FfnExpertPackParams& params [[buffer(1)]],
    uint group [[thread_position_in_grid]]
) {
    if (group < params.top_k) {
        workspace[params.off_topk_idx + group] = as_type<float>(group);
    }
}

kernel void supersonic_qwen36_ffn_mps_expert_silu(
    device const half* gate_up_out [[buffer(0)]],
    device half* down_lhs [[buffer(1)]],
    constant Qwen36FfnInt4Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint group = gid.y;
    if (row >= params.moe_intermediate || group >= params.top_k) {
        return;
    }
    uint base = group * 2u * params.moe_intermediate;
    float gate = float(gate_up_out[base + row]);
    float up = float(gate_up_out[base + params.moe_intermediate + row]);
    down_lhs[group * params.moe_intermediate + row] = half(silu(gate) * up);
}

kernel void supersonic_qwen36_ffn_mps_expert_finalize(
    device float* workspace [[buffer(0)]],
    device const bfloat* input_hidden [[buffer(1)]],
    device const half* down_out [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant Qwen36FfnInt4Params& params [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= params.hidden) {
        return;
    }
    float moe_acc = 0.0f;
    for (uint group = 0; group < params.top_k; ++group) {
        moe_acc += workspace[params.off_topk_val + group] *
            float(down_out[group * params.hidden + row]);
    }
    float moe = bf16_round_rne_finite(moe_acc);
    workspace[params.off_moe_out + row] = moe;
    float final = bf16_round_rne_finite(
        float(input_hidden[row]) + moe + workspace[params.off_shared_out + row]
    );
    output[row] = bfloat(final);
}

kernel void supersonic_qwen36_ffn_mps_transcode_hnorm(
    device const float* workspace [[buffer(0)]],
    device half* h_norm_out [[buffer(1)]],
    constant Qwen36FfnInt4Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint group = gid.y;
    if (col >= params.hidden || group >= params.top_k) {
        return;
    }
    h_norm_out[group * params.hidden + col] = half(workspace[params.off_h_norm + col]);
}

kernel void supersonic_qwen36_ffn_mps_transcode_gate_up_lut(
    device const float* workspace [[buffer(0)]],
    device const uchar* gate_up_proj [[buffer(1)]],
    device const bfloat* gate_up_scale [[buffer(2)]],
    device const bfloat* gate_up_zero [[buffer(3)]],
    device half* gate_up_rhs [[buffer(4)]],
    constant Qwen36FfnInt4Params& params [[buffer(5)]],
    threadgroup half* lut [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint row = tg.x;
    uint group = tg.y;
    uint rows = 2u * params.moe_intermediate;
    if (row >= rows || group >= params.top_k) {
        return;
    }
    uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
    uint byte_cols = (params.hidden + 1u) / 2u;
    uint scale_rows = (rows + params.group_size - 1u) / params.group_size;
    uint scale_cols = (params.hidden + params.group_size - 1u) / params.group_size;
    uint packed_base = (expert * rows + row) * byte_cols;
    uint scale_base = (expert * scale_rows + (row / params.group_size)) * scale_cols;
    uint dst_base = group * params.hidden * rows + row;
    uint pairs_per_group = params.group_size / 2u;

    for (uint scale_col = 0u; scale_col < scale_cols; ++scale_col) {
        float s = float(gate_up_scale[scale_base + scale_col]);
        float z = float(gate_up_zero[scale_base + scale_col]);
        if (tid < 16u) {
            lut[tid] = half(bf16_round_rne_finite(float(tid) * s - z * s));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < pairs_per_group) {
            uint col0 = scale_col * params.group_size + 2u * tid;
            if (col0 < params.hidden) {
                uchar packed = gate_up_proj[packed_base + col0 / 2u];
                gate_up_rhs[dst_base + col0 * rows] = lut[uint(packed & 0x0Fu)];
                uint col1 = col0 + 1u;
                if (col1 < params.hidden) {
                    gate_up_rhs[dst_base + col1 * rows] = lut[uint((packed >> 4u) & 0x0Fu)];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void supersonic_qwen36_ffn_mps_transcode_down_lut(
    device const float* workspace [[buffer(0)]],
    device const uchar* down_proj [[buffer(1)]],
    device const bfloat* down_scale [[buffer(2)]],
    device const bfloat* down_zero [[buffer(3)]],
    device half* down_rhs [[buffer(4)]],
    constant Qwen36FfnInt4Params& params [[buffer(5)]],
    threadgroup half* lut [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint row = tg.x;
    uint group = tg.y;
    if (row >= params.hidden || group >= params.top_k) {
        return;
    }
    uint expert = as_type<uint>(workspace[params.off_topk_idx + group]);
    uint byte_cols = (params.moe_intermediate + 1u) / 2u;
    uint scale_rows = (params.hidden + params.group_size - 1u) / params.group_size;
    uint scale_cols = (params.moe_intermediate + params.group_size - 1u) / params.group_size;
    uint packed_base = (expert * params.hidden + row) * byte_cols;
    uint scale_base = (expert * scale_rows + (row / params.group_size)) * scale_cols;
    uint dst_base = group * params.moe_intermediate * params.hidden + row;
    uint pairs_per_group = params.group_size / 2u;

    for (uint scale_col = 0u; scale_col < scale_cols; ++scale_col) {
        float s = float(down_scale[scale_base + scale_col]);
        float z = float(down_zero[scale_base + scale_col]);
        if (tid < 16u) {
            lut[tid] = half(bf16_round_rne_finite(float(tid) * s - z * s));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < pairs_per_group) {
            uint col0 = scale_col * params.group_size + 2u * tid;
            if (col0 < params.moe_intermediate) {
                uchar packed = down_proj[packed_base + col0 / 2u];
                down_rhs[dst_base + col0 * params.hidden] = lut[uint(packed & 0x0Fu)];
                uint col1 = col0 + 1u;
                if (col1 < params.moe_intermediate) {
                    down_rhs[dst_base + col1 * params.hidden] = lut[uint((packed >> 4u) & 0x0Fu)];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)QWEN36FFN";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:951
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen3.6 FFN INT4 library"
                                                                   }];
                } else {
                    NSArray<NSString*>* names = @[
                        @"supersonic_qwen36_router_softmax_topk_bf16",
                        @"supersonic_qwen36_ffn_router_stage5",
                        @"supersonic_qwen36_ffn_shared_gate_up",
                        @"supersonic_qwen36_ffn_shared_scalar",
                        @"supersonic_qwen36_ffn_shared_down",
                        @"supersonic_qwen36_ffn_expert_gate_up",
                        @"supersonic_qwen36_ffn_expert_gate_up_tiled",
                        @"supersonic_qwen36_ffn_expert_down_finalize",
                        @"supersonic_qwen36_ffn_expert_down_finalize_tiled",
                        @"supersonic_qwen36_batched_ffn_expert_gate_up_tiled",
                        @"supersonic_qwen36_batched_ffn_expert_down_combine_tiled",
                        @"supersonic_qwen36_ffn_pack_active_u8",
                        @"supersonic_qwen36_ffn_pack_active_bf16_pair",
                        @"supersonic_qwen36_ffn_pack_active_remap_topk",
                        @"supersonic_qwen36_ffn_mps_expert_silu",
                        @"supersonic_qwen36_ffn_mps_expert_finalize",
                        @"supersonic_qwen36_ffn_mps_transcode_hnorm",
                        @"supersonic_qwen36_ffn_mps_transcode_gate_up_lut",
                        @"supersonic_qwen36_ffn_mps_transcode_down_lut",
                    ];
                    for (NSString* name in names) {
                        id<MTLFunction> function = [library newFunctionWithName:name];
                        if (function == nil) {
                            build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                               code:952
                                                           userInfo:@{
                                                               NSLocalizedDescriptionKey :
                                                                   [NSString stringWithFormat:@"Failed to load %@", name]
                                                           }];
                            break;
                        }
                        NSError* pipeline_error = nil;
                        id<MTLComputePipelineState> pipeline =
                            [device newComputePipelineStateWithFunction:function
                                                                   error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:953
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     [NSString stringWithFormat:@"Failed to create %@", name]
                                                                             }];
                            break;
                        }
                        if ([name isEqualToString:@"supersonic_qwen36_router_softmax_topk_bf16"]) {
                            pipelines.router_topk = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_router_stage5"]) {
                            pipelines.router_stage5 = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_shared_gate_up"]) {
                            pipelines.shared_gate_up = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_shared_scalar"]) {
                            pipelines.shared_scalar = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_shared_down"]) {
                            pipelines.shared_down = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_expert_gate_up"]) {
                            pipelines.expert_gate_up = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_expert_gate_up_tiled"]) {
                            pipelines.expert_gate_up_tiled = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_expert_down_finalize"]) {
                            pipelines.expert_down_finalize = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_expert_down_finalize_tiled"]) {
                            pipelines.expert_down_finalize_tiled = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_batched_ffn_expert_gate_up_tiled"]) {
                            pipelines.batched_expert_gate_up_tiled = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_batched_ffn_expert_down_combine_tiled"]) {
                            pipelines.batched_expert_down_combine_tiled = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_pack_active_u8"]) {
                            pipelines.expert_pack_u8 = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_pack_active_bf16_pair"]) {
                            pipelines.expert_pack_bf16_pair = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_pack_active_remap_topk"]) {
                            pipelines.expert_pack_remap_topk = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_mps_expert_silu"]) {
                            pipelines.expert_mps_silu = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_mps_expert_finalize"]) {
                            pipelines.expert_mps_finalize = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_mps_transcode_hnorm"]) {
                            pipelines.expert_mps_transcode_hnorm = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_mps_transcode_gate_up_lut"]) {
                            pipelines.expert_mps_transcode_gate_up = pipeline;
                        } else if ([name isEqualToString:@"supersonic_qwen36_ffn_mps_transcode_down_lut"]) {
                            pipelines.expert_mps_transcode_down = pipeline;
                        }
                    }
                }
            }
        }
    }

    bool ok = pipelines.router_topk != nil && pipelines.router_stage5 != nil &&
        pipelines.shared_gate_up != nil && pipelines.shared_scalar != nil &&
        pipelines.shared_down != nil && pipelines.expert_gate_up != nil &&
        pipelines.expert_down_finalize != nil && pipelines.expert_down_finalize_tiled != nil &&
        pipelines.batched_expert_gate_up_tiled != nil &&
        pipelines.batched_expert_down_combine_tiled != nil &&
        pipelines.expert_pack_u8 != nil && pipelines.expert_pack_bf16_pair != nil &&
        pipelines.expert_pack_remap_topk != nil &&
        pipelines.expert_mps_silu != nil &&
        pipelines.expert_mps_finalize != nil && pipelines.expert_mps_transcode_hnorm != nil &&
        pipelines.expert_mps_transcode_gate_up != nil && pipelines.expert_mps_transcode_down != nil;
    if (!ok && build_error != nil &&
        NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"SuperSonic Qwen3.6 FFN INT4 Metal pipeline error: %@", build_error);
    }
    if (!ok && error_out != nullptr) {
        *error_out = build_error;
    }
    return pipelines;
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

id<MTLComputePipelineState> full_attention_tmajor_pipeline_bf16_f32(NSError** error_out) {
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
                static const char* kSource = R"FATTNTMAJOR(
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

kernel void supersonic_full_attention_prefill_tmajor_bf16_f32(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* key [[buffer(1)]],
    device const bfloat* value [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant FullAttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint q_head = gid.y;
    uint q_pos = gid.z;
    if (q_head >= params.q_heads || q_pos >= params.q_len || d >= params.head_dim) {
        return;
    }

    uint num_kv_groups = params.q_heads / params.kv_heads;
    uint kv_head = q_head / num_kv_groups;
    uint max_attend = min(params.seqlen_offset + q_pos + 1, params.kv_len);
    uint query_base = (q_pos * params.q_heads + q_head) * params.head_dim;

    float max_score = -INFINITY;
    for (uint kv_pos = 0; kv_pos < max_attend; ++kv_pos) {
        uint key_base = (kv_pos * params.kv_heads + kv_head) * params.head_dim;
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
        uint key_base = (kv_pos * params.kv_heads + kv_head) * params.head_dim;
        float dot = 0.0f;
        for (uint kk = 0; kk < params.head_dim; ++kk) {
            dot += float(query[query_base + kk]) * float(key[key_base + kk]);
        }
        float weight = exp((dot * params.scale) - max_score);
        denom += weight;
        uint value_base = (kv_pos * params.kv_heads + kv_head) * params.head_dim;
        numer += weight * float(value[value_base + d]);
    }

    out[(q_pos * params.q_heads + q_head) * params.head_dim + d] = numer / denom;
}
)FATTNTMAJOR";
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
                                                                           @"Failed to compile time-major full-attention library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_full_attention_prefill_tmajor_bf16_f32"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:213
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load time-major full-attention function"
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
                                                                                     @"Failed to create time-major full-attention pipeline"
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

id<MTLComputePipelineState> qwen36_ffn_residual_add_pipeline_bf16(NSError** error_out) {
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
                                                   code:771
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"Q36FFNRES(
#include <metal_stdlib>
using namespace metal;

struct Qwen36FfnResidualAddParams {
    uint total_elems;
};

kernel void supersonic_qwen36_ffn_residual_add_bf16(
    device bfloat* residual [[buffer(0)]],
    device const bfloat* combined [[buffer(1)]],
    device const bfloat* shared [[buffer(2)]],
    constant Qwen36FfnResidualAddParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    bfloat with_routed = bfloat(float(residual[gid]) + float(combined[gid]));
    residual[gid] = bfloat(float(with_routed) + float(shared[gid]));
}
)Q36FFNRES";
                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:772
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile Qwen3.6 FFN residual-add library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_qwen36_ffn_residual_add_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:773
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load Qwen3.6 FFN residual-add function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:774
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create Qwen3.6 FFN residual-add pipeline"
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

id<MTLComputePipelineState> sigmoid_mul_row_scalar_pipeline_bf16(NSError** error_out) {
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
                                                   code:1841
                                               userInfo:@{NSLocalizedDescriptionKey : @"No Metal device"}];
            } else {
                static const char* kSource = R"SIGMROWS(
#include <metal_stdlib>
using namespace metal;

struct RowScalarSigmoidParams {
    uint rows;
    uint cols;
    uint total_elems;
};

kernel void supersonic_sigmoid_mul_row_scalar_bf16(
    device const bfloat* data [[buffer(0)]],
    device const bfloat* row_gate [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant RowScalarSigmoidParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elems) {
        return;
    }
    uint row = gid / params.cols;
    if (row >= params.rows) {
        return;
    }
    float sig = 1.0f / (1.0f + exp(-float(row_gate[row])));
    out[gid] = bfloat(float(data[gid]) * sig);
}
)SIGMROWS";

                NSString* source = [NSString stringWithUTF8String:kSource];
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                configure_precise_math(options);
                NSError* library_error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:source
                                                              options:options
                                                                error:&library_error];
                if (library == nil || library_error != nil) {
                    build_error = library_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                       code:1842
                                                                   userInfo:@{
                                                                       NSLocalizedDescriptionKey :
                                                                           @"Failed to compile row-scalar sigmoid-mul library"
                                                                   }];
                } else {
                    id<MTLFunction> function =
                        [library newFunctionWithName:@"supersonic_sigmoid_mul_row_scalar_bf16"];
                    if (function == nil) {
                        build_error = [NSError errorWithDomain:@"SuperSonicMetal"
                                                           code:1843
                                                       userInfo:@{
                                                           NSLocalizedDescriptionKey :
                                                               @"Failed to load row-scalar sigmoid-mul function"
                                                       }];
                    } else {
                        NSError* pipeline_error = nil;
                        pipeline = [device newComputePipelineStateWithFunction:function
                                                                         error:&pipeline_error];
                        if (pipeline == nil || pipeline_error != nil) {
                            build_error = pipeline_error ?: [NSError errorWithDomain:@"SuperSonicMetal"
                                                                                 code:1844
                                                                             userInfo:@{
                                                                                 NSLocalizedDescriptionKey :
                                                                                     @"Failed to create row-scalar sigmoid-mul pipeline"
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

extern "C" int supersonic_metal_queue_sync() {
    @autoreleasepool {
        if (metal_batch_depth > 0) {
            return metal_batch_flush();
        }
        id<MTLCommandQueue> queue = metal_queue();
        if (queue == nil) {
            return 931;
        }
        auto command_buffer_start = MetalClock::now();
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        record_runtime_profile("command_buffer_create", command_buffer_start);
        if (command_buffer == nil) {
            return 932;
        }
        auto commit_start = MetalClock::now();
        [command_buffer commit];
        record_runtime_profile("command_buffer_commit", commit_start);
        auto wait_start = MetalClock::now();
        [command_buffer waitUntilCompleted];
        record_runtime_profile("command_buffer_wait", wait_start);
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return 933;
        }
        record_command_buffer_gpu_profile(command_buffer, "queue_sync");
        return 0;
    }
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

extern "C" int supersonic_metal_qwen36_ffn_residual_add_bf16(
    size_t total_elems,
    void* residual_ptr,
    const void* combined_ptr,
    const void* shared_ptr
) {
    @autoreleasepool {
        if (total_elems == 0) {
            return 0;
        }
        if (total_elems > UINT32_MAX || residual_ptr == nullptr ||
            combined_ptr == nullptr || shared_ptr == nullptr) {
            return 781;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = qwen36_ffn_residual_add_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 782;
        }

        id<MTLBuffer> residual = nil;
        id<MTLBuffer> combined = nil;
        id<MTLBuffer> shared = nil;
        size_t residual_offset = 0;
        size_t combined_offset = 0;
        size_t shared_offset = 0;
        if (lookup_buffer(residual_ptr, &residual, &residual_offset) != 0) {
            return 783;
        }
        if (lookup_buffer(combined_ptr, &combined, &combined_offset) != 0) {
            return 784;
        }
        if (lookup_buffer(shared_ptr, &shared, &shared_offset) != 0) {
            return 785;
        }

        ElementwiseParams params = {static_cast<uint32_t>(total_elems)};

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:residual offset:residual_offset atIndex:0];
            [encoder setBuffer:combined offset:combined_offset atIndex:1];
            [encoder setBuffer:shared offset:shared_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 786, 787, 788, 789);
    }
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

extern "C" int supersonic_metal_sigmoid_mul_row_scalar_bf16(
    size_t rows,
    size_t cols,
    const void* data_ptr,
    const void* row_gate_ptr,
    void* out_ptr
) {
    @autoreleasepool {
        if (rows == 0 || cols == 0) {
            return 0;
        }
        if (rows > UINT32_MAX || cols > UINT32_MAX ||
            data_ptr == nullptr || row_gate_ptr == nullptr || out_ptr == nullptr) {
            return 1845;
        }
        const size_t total_elems = rows * cols;
        if (cols != 0 && total_elems / cols != rows) {
            return 1846;
        }
        if (total_elems > UINT32_MAX) {
            return 1847;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = sigmoid_mul_row_scalar_pipeline_bf16(&pipeline_error);
        if (pipeline == nil) {
            return 1848;
        }

        id<MTLBuffer> data = nil;
        id<MTLBuffer> row_gate = nil;
        id<MTLBuffer> out = nil;
        size_t data_offset = 0;
        size_t row_gate_offset = 0;
        size_t out_offset = 0;
        if (lookup_buffer(data_ptr, &data, &data_offset) != 0) {
            return 1849;
        }
        if (lookup_buffer(row_gate_ptr, &row_gate, &row_gate_offset) != 0) {
            return 1850;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 1851;
        }

        RowScalarSigmoidParams params = {
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols),
            static_cast<uint32_t>(total_elems),
        };

        return encode_or_submit_labeled([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:data offset:data_offset atIndex:0];
            [encoder setBuffer:row_gate offset:row_gate_offset atIndex:1];
            [encoder setBuffer:out offset:out_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
            MTLSize threads_per_group = MTLSizeMake(tg_width, 1, 1);
            MTLSize threads_per_grid = MTLSizeMake(total_elems, 1, 1);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, "sigmoid_mul_row_scalar", 1852, 1853, 1854, 1855);
    }
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

extern "C" int supersonic_metal_matmul_rhs_transposed_int4_bf16_gemv_m1(
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
        if (n == 0 || k == 0 || group_size == 0 || lhs_ptr == nullptr ||
            rhs_int4_ptr == nullptr || scale_ptr == nullptr || zero_ptr == nullptr ||
            out_ptr == nullptr) {
            return 720;
        }
        if (k % 2 != 0) {
            return 721;
        }
        if (n > UINT32_MAX || k > UINT32_MAX || group_size > UINT32_MAX) {
            return 722;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = matmul_pipeline_int4_bf16_gemv_m1(&pipeline_error);
        if (pipeline == nil) {
            return 723;
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
            return 724;
        }
        if (lookup_buffer(rhs_int4_ptr, &rhs, &rhs_offset) != 0) {
            return 725;
        }
        if (lookup_buffer(scale_ptr, &sc, &sc_offset) != 0) {
            return 726;
        }
        if (lookup_buffer(zero_ptr, &zr, &zr_offset) != 0) {
            return 727;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 728;
        }

        struct MatmulInt4GemvParams {
            uint32_t n;
            uint32_t k;
            uint32_t group_size;
        } params = {
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

            // One SIMD-group (32 threads) per output column.
            MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
            MTLSize threadgroups = MTLSizeMake(n, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 729, 730, 731, 732);
    }
}

extern "C" int supersonic_metal_matmul_rhs_transposed_int4_bf16_gemv_m1_tiled(
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
        if (n == 0 || k == 0 || group_size == 0 || lhs_ptr == nullptr ||
            rhs_int4_ptr == nullptr || scale_ptr == nullptr || zero_ptr == nullptr ||
            out_ptr == nullptr) {
            return 820;
        }
        if (k % 2 != 0) {
            return 821;
        }
        if (n > UINT32_MAX || k > UINT32_MAX || group_size > UINT32_MAX) {
            return 822;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            matmul_pipeline_int4_bf16_gemv_m1_tiled(&pipeline_error);
        if (pipeline == nil) {
            return 823;
        }

        size_t shared_bytes = k * sizeof(float);

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
            return 824;
        }
        if (lookup_buffer(rhs_int4_ptr, &rhs, &rhs_offset) != 0) {
            return 825;
        }
        if (lookup_buffer(scale_ptr, &sc, &sc_offset) != 0) {
            return 826;
        }
        if (lookup_buffer(zero_ptr, &zr, &zr_offset) != 0) {
            return 827;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 828;
        }

        struct MatmulInt4GemvParams {
            uint32_t n;
            uint32_t k;
            uint32_t group_size;
        } params = {
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
            static_cast<uint32_t>(group_size),
        };

        // 32 cols per threadgroup → ceil(n / 32) threadgroups.
        size_t tg_count = (n + 31) / 32;

        return encode_or_submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:lhs offset:lhs_offset atIndex:0];
            [encoder setBuffer:rhs offset:rhs_offset atIndex:1];
            [encoder setBuffer:sc offset:sc_offset atIndex:2];
            [encoder setBuffer:zr offset:zr_offset atIndex:3];
            [encoder setBuffer:out offset:out_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];

            MTLSize threads_per_group = MTLSizeMake(1024, 1, 1);
            MTLSize threadgroups = MTLSizeMake(tg_count, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        }, 829, 830, 831, 832);
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

extern "C" int supersonic_metal_qwen36_linear_int4_stage5(
    size_t hidden,
    size_t num_k_heads,
    size_t num_v_heads,
    size_t head_k_dim,
    size_t head_v_dim,
    size_t conv_kernel_dim,
    size_t group_size,
    float rms_norm_eps,
    const void* input_hidden_ptr,
    const void* input_norm_w_ptr,
    const void* in_proj_qkv_ptr,
    const void* in_proj_qkv_scale_ptr,
    const void* in_proj_qkv_zero_ptr,
    const void* in_proj_z_ptr,
    const void* in_proj_z_scale_ptr,
    const void* in_proj_z_zero_ptr,
    const void* in_proj_a_ptr,
    const void* in_proj_b_ptr,
    const void* conv1d_w_ptr,
    const void* conv1d_bias_ptr,
    const void* dt_bias_ptr,
    const void* a_log_ptr,
    const void* norm_w_ptr,
    const void* out_proj_ptr,
    const void* out_proj_scale_ptr,
    const void* out_proj_zero_ptr,
    void* conv_state_ptr,
    void* recurrent_state_ptr,
    void* workspace_ptr,
    void* output_ptr,
    void* final_output_ptr,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || num_k_heads == 0 || num_v_heads == 0 ||
            head_k_dim == 0 || head_v_dim == 0 || conv_kernel_dim == 0 ||
            group_size == 0 || input_hidden_ptr == nullptr ||
            input_norm_w_ptr == nullptr || in_proj_qkv_ptr == nullptr ||
            in_proj_qkv_scale_ptr == nullptr || in_proj_qkv_zero_ptr == nullptr ||
            in_proj_z_ptr == nullptr || in_proj_z_scale_ptr == nullptr ||
            in_proj_z_zero_ptr == nullptr || in_proj_a_ptr == nullptr ||
            in_proj_b_ptr == nullptr || conv1d_w_ptr == nullptr ||
            dt_bias_ptr == nullptr || a_log_ptr == nullptr ||
            norm_w_ptr == nullptr || out_proj_ptr == nullptr ||
            out_proj_scale_ptr == nullptr || out_proj_zero_ptr == nullptr ||
            conv_state_ptr == nullptr || recurrent_state_ptr == nullptr ||
            workspace_ptr == nullptr || output_ptr == nullptr ||
            final_output_ptr == nullptr) {
            return 1120;
        }
        if (hidden > UINT32_MAX || num_k_heads > UINT32_MAX ||
            num_v_heads > UINT32_MAX || head_k_dim > UINT32_MAX ||
            head_v_dim > UINT32_MAX || conv_kernel_dim > UINT32_MAX ||
            group_size > UINT32_MAX) {
            return 1121;
        }
        if (num_v_heads % num_k_heads != 0 || conv_kernel_dim < 1 ||
            hidden % 2 != 0 || head_k_dim % 2 != 0 || head_v_dim % 2 != 0) {
            return 1122;
        }
        if (num_k_heads > SIZE_MAX / head_k_dim ||
            num_v_heads > SIZE_MAX / head_v_dim) {
            return 1123;
        }
        size_t key_dim = num_k_heads * head_k_dim;
        size_t val_dim = num_v_heads * head_v_dim;
        if (key_dim > (SIZE_MAX - val_dim) / 2) {
            return 1124;
        }
        size_t qkv_dim = 2 * key_dim + val_dim;
        size_t total_projection_rows = qkv_dim + val_dim + 2 * num_v_heads;
        if (key_dim > UINT32_MAX || val_dim > UINT32_MAX ||
            qkv_dim > UINT32_MAX || total_projection_rows > UINT32_MAX) {
            return 1125;
        }

        NSError* pipeline_error = nil;
        Qwen36LinearInt4Pipelines pipelines = qwen36_linear_int4_pipelines(&pipeline_error);
        if (pipelines.input_norm == nil || pipelines.projections == nil ||
            pipelines.conv_silu_state == nil || pipelines.qk_norm_repeat == nil ||
            pipelines.recurrent_update == nil || pipelines.output_gate_norm == nil ||
            pipelines.out_proj_finalize == nil) {
            return 1126;
        }

        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> input_norm_w = nil;
        id<MTLBuffer> in_proj_qkv = nil;
        id<MTLBuffer> in_proj_qkv_scale = nil;
        id<MTLBuffer> in_proj_qkv_zero = nil;
        id<MTLBuffer> in_proj_z = nil;
        id<MTLBuffer> in_proj_z_scale = nil;
        id<MTLBuffer> in_proj_z_zero = nil;
        id<MTLBuffer> in_proj_a = nil;
        id<MTLBuffer> in_proj_b = nil;
        id<MTLBuffer> conv1d_w = nil;
        id<MTLBuffer> conv1d_bias = nil;
        id<MTLBuffer> dt_bias = nil;
        id<MTLBuffer> a_log = nil;
        id<MTLBuffer> norm_w = nil;
        id<MTLBuffer> out_proj = nil;
        id<MTLBuffer> out_proj_scale = nil;
        id<MTLBuffer> out_proj_zero = nil;
        id<MTLBuffer> conv_state = nil;
        id<MTLBuffer> recurrent_state = nil;
        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> output = nil;
        id<MTLBuffer> final_output = nil;
        size_t input_hidden_offset = 0;
        size_t input_norm_w_offset = 0;
        size_t in_proj_qkv_offset = 0;
        size_t in_proj_qkv_scale_offset = 0;
        size_t in_proj_qkv_zero_offset = 0;
        size_t in_proj_z_offset = 0;
        size_t in_proj_z_scale_offset = 0;
        size_t in_proj_z_zero_offset = 0;
        size_t in_proj_a_offset = 0;
        size_t in_proj_b_offset = 0;
        size_t conv1d_w_offset = 0;
        size_t conv1d_bias_offset = 0;
        size_t dt_bias_offset = 0;
        size_t a_log_offset = 0;
        size_t norm_w_offset = 0;
        size_t out_proj_offset = 0;
        size_t out_proj_scale_offset = 0;
        size_t out_proj_zero_offset = 0;
        size_t conv_state_offset = 0;
        size_t recurrent_state_offset = 0;
        size_t workspace_offset = 0;
        size_t output_offset = 0;
        size_t final_output_offset = 0;

        auto lookup_required = [](const void* ptr, id<MTLBuffer>* buffer, size_t* offset, int code) -> int {
            return lookup_buffer(ptr, buffer, offset) == 0 ? 0 : code;
        };
        int status = 0;
        if ((status = lookup_required(input_hidden_ptr, &input_hidden, &input_hidden_offset, 1127)) != 0) return status;
        if ((status = lookup_required(input_norm_w_ptr, &input_norm_w, &input_norm_w_offset, 1128)) != 0) return status;
        if ((status = lookup_required(in_proj_qkv_ptr, &in_proj_qkv, &in_proj_qkv_offset, 1129)) != 0) return status;
        if ((status = lookup_required(in_proj_qkv_scale_ptr, &in_proj_qkv_scale, &in_proj_qkv_scale_offset, 1130)) != 0) return status;
        if ((status = lookup_required(in_proj_qkv_zero_ptr, &in_proj_qkv_zero, &in_proj_qkv_zero_offset, 1131)) != 0) return status;
        if ((status = lookup_required(in_proj_z_ptr, &in_proj_z, &in_proj_z_offset, 1132)) != 0) return status;
        if ((status = lookup_required(in_proj_z_scale_ptr, &in_proj_z_scale, &in_proj_z_scale_offset, 1133)) != 0) return status;
        if ((status = lookup_required(in_proj_z_zero_ptr, &in_proj_z_zero, &in_proj_z_zero_offset, 1134)) != 0) return status;
        if ((status = lookup_required(in_proj_a_ptr, &in_proj_a, &in_proj_a_offset, 1135)) != 0) return status;
        if ((status = lookup_required(in_proj_b_ptr, &in_proj_b, &in_proj_b_offset, 1136)) != 0) return status;
        if ((status = lookup_required(conv1d_w_ptr, &conv1d_w, &conv1d_w_offset, 1137)) != 0) return status;
        if (conv1d_bias_ptr != nullptr) {
            if ((status = lookup_required(conv1d_bias_ptr, &conv1d_bias, &conv1d_bias_offset, 1138)) != 0) return status;
        } else {
            conv1d_bias = conv1d_w;
            conv1d_bias_offset = conv1d_w_offset;
        }
        if ((status = lookup_required(dt_bias_ptr, &dt_bias, &dt_bias_offset, 1139)) != 0) return status;
        if ((status = lookup_required(a_log_ptr, &a_log, &a_log_offset, 1140)) != 0) return status;
        if ((status = lookup_required(norm_w_ptr, &norm_w, &norm_w_offset, 1141)) != 0) return status;
        if ((status = lookup_required(out_proj_ptr, &out_proj, &out_proj_offset, 1142)) != 0) return status;
        if ((status = lookup_required(out_proj_scale_ptr, &out_proj_scale, &out_proj_scale_offset, 1143)) != 0) return status;
        if ((status = lookup_required(out_proj_zero_ptr, &out_proj_zero, &out_proj_zero_offset, 1144)) != 0) return status;
        if ((status = lookup_required(conv_state_ptr, &conv_state, &conv_state_offset, 1145)) != 0) return status;
        if ((status = lookup_required(recurrent_state_ptr, &recurrent_state, &recurrent_state_offset, 1146)) != 0) return status;
        if ((status = lookup_required(workspace_ptr, &workspace, &workspace_offset, 1147)) != 0) return status;
        if ((status = lookup_required(output_ptr, &output, &output_offset, 1148)) != 0) return status;
        if ((status = lookup_required(final_output_ptr, &final_output, &final_output_offset, 1181)) != 0) return status;

        uint32_t off_qkv_raw = 0u;
        uint32_t off_z_raw = static_cast<uint32_t>(qkv_dim);
        uint32_t off_a_raw = static_cast<uint32_t>(qkv_dim + val_dim);
        uint32_t off_b_raw = static_cast<uint32_t>(qkv_dim + val_dim + num_v_heads);
        uint32_t off_q_normed = static_cast<uint32_t>(qkv_dim + val_dim + 2 * num_v_heads);
        uint32_t off_k_normed = off_q_normed + static_cast<uint32_t>(key_dim);
        uint32_t off_q_rep = off_k_normed + static_cast<uint32_t>(key_dim);
        uint32_t off_k_rep = off_q_rep + static_cast<uint32_t>(num_v_heads * head_k_dim);
        uint32_t off_beta = off_k_rep + static_cast<uint32_t>(num_v_heads * head_k_dim);
        uint32_t off_g = off_beta + static_cast<uint32_t>(num_v_heads);
        uint32_t off_rec_out = off_g + static_cast<uint32_t>(num_v_heads);

        Qwen36LinearInt4Params params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(num_k_heads),
            static_cast<uint32_t>(num_v_heads),
            static_cast<uint32_t>(head_k_dim),
            static_cast<uint32_t>(head_v_dim),
            static_cast<uint32_t>(conv_kernel_dim),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(key_dim),
            static_cast<uint32_t>(val_dim),
            static_cast<uint32_t>(qkv_dim),
            static_cast<uint32_t>(conv_kernel_dim - 1),
            conv1d_bias_ptr != nullptr ? 1u : 0u,
            off_qkv_raw,
            off_z_raw,
            off_a_raw,
            off_b_raw,
            off_q_normed,
            off_k_normed,
            off_q_rep,
            off_k_rep,
            off_beta,
            off_g,
            off_rec_out,
            rms_norm_eps,
            1.0f / sqrtf(static_cast<float>(head_k_dim)),
        };

        auto encode_input_norm = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.input_norm];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:0];
            [encoder setBuffer:input_norm_w offset:input_norm_w_offset atIndex:1];
            [encoder setBuffer:output offset:output_offset atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        };
        auto encode_projections = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.projections];
            [encoder setBuffer:output offset:output_offset atIndex:0];
            [encoder setBuffer:in_proj_qkv offset:in_proj_qkv_offset atIndex:1];
            [encoder setBuffer:in_proj_qkv_scale offset:in_proj_qkv_scale_offset atIndex:2];
            [encoder setBuffer:in_proj_qkv_zero offset:in_proj_qkv_zero_offset atIndex:3];
            [encoder setBuffer:in_proj_z offset:in_proj_z_offset atIndex:4];
            [encoder setBuffer:in_proj_z_scale offset:in_proj_z_scale_offset atIndex:5];
            [encoder setBuffer:in_proj_z_zero offset:in_proj_z_zero_offset atIndex:6];
            [encoder setBuffer:in_proj_a offset:in_proj_a_offset atIndex:7];
            [encoder setBuffer:in_proj_b offset:in_proj_b_offset atIndex:8];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:9];
            [encoder setBytes:&params length:sizeof(params) atIndex:10];
            [encoder dispatchThreadgroups:MTLSizeMake(total_projection_rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_conv = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.conv_silu_state];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:conv1d_w offset:conv1d_w_offset atIndex:1];
            [encoder setBuffer:conv1d_bias offset:conv1d_bias_offset atIndex:2];
            [encoder setBuffer:conv_state offset:conv_state_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            NSUInteger tg_width =
                std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipelines.conv_silu_state.maxTotalThreadsPerThreadgroup));
            [encoder dispatchThreads:MTLSizeMake(qkv_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        };
        auto encode_qk_norm_repeat = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.qk_norm_repeat];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBytes:&params length:sizeof(params) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_recurrent = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.recurrent_update];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:dt_bias offset:dt_bias_offset atIndex:1];
            [encoder setBuffer:a_log offset:a_log_offset atIndex:2];
            [encoder setBuffer:recurrent_state offset:recurrent_state_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(head_v_dim, num_v_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_output_gate_norm = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.output_gate_norm];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:norm_w offset:norm_w_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_out_proj_finalize = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.out_proj_finalize];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:0];
            [encoder setBuffer:out_proj offset:out_proj_offset atIndex:1];
            [encoder setBuffer:out_proj_scale offset:out_proj_scale_offset atIndex:2];
            [encoder setBuffer:out_proj_zero offset:out_proj_zero_offset atIndex:3];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:4];
            [encoder setBuffer:output offset:output_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];
            [encoder setBuffer:final_output offset:final_output_offset atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };

        bool split_profile = NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE"] != nil;
        if (split_profile) {
            if ((status = encode_or_submit_labeled(encode_input_norm, "qwen36_linear_int4_input_norm", 1149, 1150, 1151, 1152)) != 0) return status;
            if ((status = encode_or_submit_labeled(encode_projections, "qwen36_linear_int4_projections", 1153, 1154, 1155, 1156)) != 0) return status;
            if ((status = encode_or_submit_labeled(encode_conv, "qwen36_linear_int4_conv_silu_state", 1157, 1158, 1159, 1160)) != 0) return status;
            if ((status = encode_or_submit_labeled(encode_qk_norm_repeat, "qwen36_linear_int4_qk_norm_repeat", 1161, 1162, 1163, 1164)) != 0) return status;
            if ((status = encode_or_submit_labeled(encode_recurrent, "qwen36_linear_int4_recurrent_update", 1165, 1166, 1167, 1168)) != 0) return status;
            if ((status = encode_or_submit_labeled(encode_output_gate_norm, "qwen36_linear_int4_output_gate_norm", 1169, 1170, 1171, 1172)) != 0) return status;
            return encode_or_submit_labeled(encode_out_proj_finalize, "qwen36_linear_int4_out_proj_finalize", 1173, 1174, 1175, 1176);
        }

        auto encode_stage = [&](id<MTLComputeCommandEncoder> encoder) {
            encode_input_norm(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_projections(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_conv(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_qk_norm_repeat(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_recurrent(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_output_gate_norm(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_out_proj_finalize(encoder);
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode_stage,
                "qwen36_linear_int4_stage5",
                1177,
                1178,
                1179,
                1180
            );
        }
        return encode_or_submit_labeled_async(
            encode_stage,
            "qwen36_linear_int4_stage5",
            1177,
            1178,
            1179,
            1180
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_gate_up_tiled(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    size_t group_size,
    void* workspace_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    size_t off_h_norm,
    size_t off_topk_idx,
    size_t off_expert_mid,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || group_size == 0 ||
            workspace_ptr == nullptr || gate_up_proj_ptr == nullptr ||
            gate_up_scale_ptr == nullptr || gate_up_zero_ptr == nullptr) {
            return 1181;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX ||
            top_k > UINT32_MAX || group_size > UINT32_MAX ||
            off_h_norm > UINT32_MAX || off_topk_idx > UINT32_MAX ||
            off_expert_mid > UINT32_MAX) {
            return 1182;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0) {
            return 1183;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_gate_up_tiled == nil) {
            return 1184;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        size_t workspace_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1185;
        if (lookup_buffer(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset) != 0) return 1186;
        if (lookup_buffer(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset) != 0) return 1187;
        if (lookup_buffer(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset) != 0) return 1188;

        Qwen36FfnExpertGateUpTiledParams params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_idx),
            static_cast<uint32_t>(off_expert_mid),
        };

        auto encode = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_gate_up_tiled];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:1];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:2];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode,
                "qwen36_ffn_int4_expert_gate_up_tiled",
                1189,
                1190,
                1191,
                1192
            );
        }
        return encode_or_submit_labeled_async(
            encode,
            "qwen36_ffn_int4_expert_gate_up_tiled",
            1189,
            1190,
            1191,
            1192
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_gate_up_down_finalize_tiled(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    size_t group_size,
    void* workspace_ptr,
    const void* input_hidden_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* output_ptr,
    size_t off_h_norm,
    size_t off_topk_val,
    size_t off_topk_idx,
    size_t off_shared_out,
    size_t off_expert_mid,
    size_t off_moe_out,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || group_size == 0 ||
            workspace_ptr == nullptr || input_hidden_ptr == nullptr ||
            gate_up_proj_ptr == nullptr || gate_up_scale_ptr == nullptr ||
            gate_up_zero_ptr == nullptr || down_proj_ptr == nullptr ||
            down_scale_ptr == nullptr || down_zero_ptr == nullptr || output_ptr == nullptr) {
            return 1193;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX ||
            top_k > UINT32_MAX || group_size > UINT32_MAX ||
            off_h_norm > UINT32_MAX || off_topk_val > UINT32_MAX ||
            off_topk_idx > UINT32_MAX || off_shared_out > UINT32_MAX ||
            off_expert_mid > UINT32_MAX || off_moe_out > UINT32_MAX) {
            return 1194;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0) {
            return 1195;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_gate_up_tiled == nil || pipelines.expert_down_finalize == nil) {
            return 1196;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        id<MTLBuffer> down_proj = nil;
        id<MTLBuffer> down_scale = nil;
        id<MTLBuffer> down_zero = nil;
        id<MTLBuffer> output = nil;
        size_t workspace_offset = 0;
        size_t input_hidden_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        size_t down_proj_offset = 0;
        size_t down_scale_offset = 0;
        size_t down_zero_offset = 0;
        size_t output_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1197;
        if (lookup_buffer(input_hidden_ptr, &input_hidden, &input_hidden_offset) != 0) return 1198;
        if (lookup_buffer(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset) != 0) return 1199;
        if (lookup_buffer(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset) != 0) return 1200;
        if (lookup_buffer(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset) != 0) return 1201;
        if (lookup_buffer(down_proj_ptr, &down_proj, &down_proj_offset) != 0) return 1202;
        if (lookup_buffer(down_scale_ptr, &down_scale, &down_scale_offset) != 0) return 1203;
        if (lookup_buffer(down_zero_ptr, &down_zero, &down_zero_offset) != 0) return 1204;
        if (lookup_buffer(output_ptr, &output, &output_offset) != 0) return 1205;

        Qwen36FfnExpertGateUpTiledParams gate_params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_idx),
            static_cast<uint32_t>(off_expert_mid),
        };
        Qwen36FfnInt4Params down_params = {
            static_cast<uint32_t>(hidden),
            0u,
            static_cast<uint32_t>(moe_intermediate),
            0u,
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_val),
            static_cast<uint32_t>(off_topk_idx),
            0u,
            0u,
            static_cast<uint32_t>(off_shared_out),
            static_cast<uint32_t>(off_expert_mid),
            static_cast<uint32_t>(off_moe_out),
        };

        auto encode = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_gate_up_tiled];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:1];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:2];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:3];
            [encoder setBytes:&gate_params length:sizeof(gate_params) atIndex:4];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_down_finalize];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
            [encoder setBuffer:down_proj offset:down_proj_offset atIndex:2];
            [encoder setBuffer:down_scale offset:down_scale_offset atIndex:3];
            [encoder setBuffer:down_zero offset:down_zero_offset atIndex:4];
            [encoder setBuffer:output offset:output_offset atIndex:5];
            [encoder setBytes:&down_params length:sizeof(down_params) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode,
                "qwen36_ffn_int4_expert_gate_up_down_finalize_tiled",
                1206,
                1207,
                1208,
                1209
            );
        }
        return encode_or_submit_labeled_async(
            encode,
            "qwen36_ffn_int4_expert_gate_up_down_finalize_tiled",
            1206,
            1207,
            1208,
            1209
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_direct_gather_stage5(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    size_t group_size,
    void* workspace_ptr,
    const void* input_hidden_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* output_ptr,
    size_t off_h_norm,
    size_t off_topk_val,
    size_t off_topk_idx,
    size_t off_shared_out,
    size_t off_expert_mid,
    size_t off_moe_out,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || group_size == 0 ||
            workspace_ptr == nullptr || input_hidden_ptr == nullptr ||
            gate_up_proj_ptr == nullptr || gate_up_scale_ptr == nullptr ||
            gate_up_zero_ptr == nullptr || down_proj_ptr == nullptr ||
            down_scale_ptr == nullptr || down_zero_ptr == nullptr || output_ptr == nullptr) {
            return 1253;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX ||
            top_k > UINT32_MAX || group_size > UINT32_MAX ||
            off_h_norm > UINT32_MAX || off_topk_val > UINT32_MAX ||
            off_topk_idx > UINT32_MAX || off_shared_out > UINT32_MAX ||
            off_expert_mid > UINT32_MAX || off_moe_out > UINT32_MAX) {
            return 1254;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0 ||
            (hidden % group_size) != 0 || (moe_intermediate % group_size) != 0 ||
            ((2 * moe_intermediate) % group_size) != 0) {
            return 1255;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_gate_up_tiled == nil ||
            pipelines.expert_down_finalize == nil) {
            return 1256;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        id<MTLBuffer> down_proj = nil;
        id<MTLBuffer> down_scale = nil;
        id<MTLBuffer> down_zero = nil;
        id<MTLBuffer> output = nil;
        size_t workspace_offset = 0;
        size_t input_hidden_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        size_t down_proj_offset = 0;
        size_t down_scale_offset = 0;
        size_t down_zero_offset = 0;
        size_t output_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1257;
        if (lookup_buffer(input_hidden_ptr, &input_hidden, &input_hidden_offset) != 0) return 1258;
        if (lookup_buffer(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset) != 0) return 1259;
        if (lookup_buffer(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset) != 0) return 1260;
        if (lookup_buffer(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset) != 0) return 1261;
        if (lookup_buffer(down_proj_ptr, &down_proj, &down_proj_offset) != 0) return 1262;
        if (lookup_buffer(down_scale_ptr, &down_scale, &down_scale_offset) != 0) return 1263;
        if (lookup_buffer(down_zero_ptr, &down_zero, &down_zero_offset) != 0) return 1264;
        if (lookup_buffer(output_ptr, &output, &output_offset) != 0) return 1265;

        Qwen36FfnExpertGateUpTiledParams gate_params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_idx),
            static_cast<uint32_t>(off_expert_mid),
        };
        Qwen36FfnInt4Params down_params = {
            static_cast<uint32_t>(hidden),
            0u,
            static_cast<uint32_t>(moe_intermediate),
            0u,
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_val),
            static_cast<uint32_t>(off_topk_idx),
            0u,
            0u,
            static_cast<uint32_t>(off_shared_out),
            static_cast<uint32_t>(off_expert_mid),
            static_cast<uint32_t>(off_moe_out),
        };

        auto encode = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_gate_up_tiled];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:1];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:2];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:3];
            [encoder setBytes:&gate_params length:sizeof(gate_params) atIndex:4];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_down_finalize];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
            [encoder setBuffer:down_proj offset:down_proj_offset atIndex:2];
            [encoder setBuffer:down_scale offset:down_scale_offset atIndex:3];
            [encoder setBuffer:down_zero offset:down_zero_offset atIndex:4];
            [encoder setBuffer:output offset:output_offset atIndex:5];
            [encoder setBytes:&down_params length:sizeof(down_params) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode,
                "qwen36_ffn_int4_expert_direct_gather_stage5",
                1266,
                1267,
                1268,
                1269
            );
        }
        return encode_or_submit_labeled_async(
            encode,
            "qwen36_ffn_int4_expert_direct_gather_stage5",
            1266,
            1267,
            1268,
            1269
        );
    }
}

extern "C" int supersonic_metal_qwen36_batched_ffn_grouped_expert_direct(
    size_t n_tokens,
    size_t top_k,
    size_t hidden,
    size_t moe_intermediate,
    size_t group_size,
    const void* x_norm_ptr,
    const void* topk_idx_ptr,
    const void* topk_weight_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* expert_mid_ptr,
    void* combined_ptr,
    int wait_for_completion
) {
    @autoreleasepool {
        if (n_tokens == 0 || top_k == 0 || hidden == 0 || moe_intermediate == 0 ||
            group_size == 0 || x_norm_ptr == nullptr || topk_idx_ptr == nullptr ||
            topk_weight_ptr == nullptr || gate_up_proj_ptr == nullptr ||
            gate_up_scale_ptr == nullptr || gate_up_zero_ptr == nullptr ||
            down_proj_ptr == nullptr || down_scale_ptr == nullptr ||
            down_zero_ptr == nullptr || expert_mid_ptr == nullptr || combined_ptr == nullptr) {
            return 1260;
        }
        if (n_tokens > UINT32_MAX || top_k > UINT32_MAX || hidden > UINT32_MAX ||
            moe_intermediate > UINT32_MAX || group_size > UINT32_MAX) {
            return 1261;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0 || group_size == 0) {
            return 1262;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.batched_expert_gate_up_tiled == nil ||
            pipelines.batched_expert_down_combine_tiled == nil) {
            return 1263;
        }

        id<MTLBuffer> x_norm = nil;
        id<MTLBuffer> topk_idx = nil;
        id<MTLBuffer> topk_weight = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        id<MTLBuffer> down_proj = nil;
        id<MTLBuffer> down_scale = nil;
        id<MTLBuffer> down_zero = nil;
        id<MTLBuffer> expert_mid = nil;
        id<MTLBuffer> combined = nil;
        size_t x_norm_offset = 0;
        size_t topk_idx_offset = 0;
        size_t topk_weight_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        size_t down_proj_offset = 0;
        size_t down_scale_offset = 0;
        size_t down_zero_offset = 0;
        size_t expert_mid_offset = 0;
        size_t combined_offset = 0;
        auto lookup_required = [](const void* ptr, id<MTLBuffer>* buffer, size_t* offset, int code) -> int {
            return lookup_buffer(ptr, buffer, offset) == 0 ? 0 : code;
        };
        int status = 0;
        if ((status = lookup_required(x_norm_ptr, &x_norm, &x_norm_offset, 1264)) != 0) return status;
        if ((status = lookup_required(topk_idx_ptr, &topk_idx, &topk_idx_offset, 1265)) != 0) return status;
        if ((status = lookup_required(topk_weight_ptr, &topk_weight, &topk_weight_offset, 1266)) != 0) return status;
        if ((status = lookup_required(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset, 1267)) != 0) return status;
        if ((status = lookup_required(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset, 1268)) != 0) return status;
        if ((status = lookup_required(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset, 1269)) != 0) return status;
        if ((status = lookup_required(down_proj_ptr, &down_proj, &down_proj_offset, 1270)) != 0) return status;
        if ((status = lookup_required(down_scale_ptr, &down_scale, &down_scale_offset, 1271)) != 0) return status;
        if ((status = lookup_required(down_zero_ptr, &down_zero, &down_zero_offset, 1272)) != 0) return status;
        if ((status = lookup_required(expert_mid_ptr, &expert_mid, &expert_mid_offset, 1273)) != 0) return status;
        if ((status = lookup_required(combined_ptr, &combined, &combined_offset, 1274)) != 0) return status;

        Qwen36BatchedFfnExpertParams params = {
            static_cast<uint32_t>(n_tokens),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(group_size),
        };

        auto encode_gate_up = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.batched_expert_gate_up_tiled];
            [encoder setBuffer:x_norm offset:x_norm_offset atIndex:0];
            [encoder setBuffer:topk_idx offset:topk_idx_offset atIndex:1];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:2];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:3];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:4];
            [encoder setBuffer:expert_mid offset:expert_mid_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, n_tokens)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto encode_down_combine = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.batched_expert_down_combine_tiled];
            [encoder setBuffer:topk_idx offset:topk_idx_offset atIndex:0];
            [encoder setBuffer:topk_weight offset:topk_weight_offset atIndex:1];
            [encoder setBuffer:down_proj offset:down_proj_offset atIndex:2];
            [encoder setBuffer:down_scale offset:down_scale_offset atIndex:3];
            [encoder setBuffer:down_zero offset:down_zero_offset atIndex:4];
            [encoder setBuffer:expert_mid offset:expert_mid_offset atIndex:5];
            [encoder setBuffer:combined offset:combined_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, n_tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        bool split_profile = qwen36_ffn_phase_profile_enabled();
        if (split_profile) {
            if ((status = encode_or_submit_labeled(
                    encode_gate_up,
                    "qwen36_batched_prefill_grouped_expert_gate_up",
                    1275,
                    1276,
                    1277,
                    1278)) != 0) {
                return status;
            }
            return encode_or_submit_labeled(
                encode_down_combine,
                "qwen36_batched_prefill_grouped_expert_down_combine",
                1279,
                1280,
                1281,
                1282);
        }

        auto encode_stage = [&](id<MTLComputeCommandEncoder> encoder) {
            encode_gate_up(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_down_combine(encoder);
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode_stage,
                "qwen36_batched_prefill_grouped_expert_direct",
                1283,
                1284,
                1285,
                1286);
        }
        return encode_or_submit_labeled_async(
            encode_stage,
            "qwen36_batched_prefill_grouped_expert_direct",
            1283,
            1284,
            1285,
            1286);
    }
}

extern "C" int supersonic_metal_qwen36_router_softmax_topk_bf16(
    size_t n_tokens,
    size_t num_experts,
    size_t top_k,
    const void* logits_ptr,
    void* topk_idx_ptr,
    void* topk_weight_ptr
) {
    @autoreleasepool {
        if (n_tokens == 0 || num_experts == 0 || top_k == 0 || logits_ptr == nullptr ||
            topk_idx_ptr == nullptr || topk_weight_ptr == nullptr) {
            return 1290;
        }
        if (n_tokens > UINT32_MAX || num_experts > 256 || top_k > 16) {
            return 1291;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.router_topk == nil) {
            return 1292;
        }

        id<MTLBuffer> logits = nil;
        id<MTLBuffer> topk_idx = nil;
        id<MTLBuffer> topk_weight = nil;
        size_t logits_offset = 0;
        size_t topk_idx_offset = 0;
        size_t topk_weight_offset = 0;
        if (lookup_buffer(logits_ptr, &logits, &logits_offset) != 0) {
            return 1293;
        }
        if (lookup_buffer(topk_idx_ptr, &topk_idx, &topk_idx_offset) != 0) {
            return 1294;
        }
        if (lookup_buffer(topk_weight_ptr, &topk_weight, &topk_weight_offset) != 0) {
            return 1295;
        }

        Qwen36RouterTopkParams params = {
            static_cast<uint32_t>(n_tokens),
            static_cast<uint32_t>(num_experts),
            static_cast<uint32_t>(top_k),
        };

        return encode_or_submit_labeled(
            [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:pipelines.router_topk];
                [encoder setBuffer:logits offset:logits_offset atIndex:0];
                [encoder setBuffer:topk_idx offset:topk_idx_offset atIndex:1];
                [encoder setBuffer:topk_weight offset:topk_weight_offset atIndex:2];
                [encoder setBytes:&params length:sizeof(params) atIndex:3];
                [encoder setThreadgroupMemoryLength:(256 + 16) * sizeof(float) atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            },
            "qwen36_batched_prefill_router_softmax_topk",
            1296,
            1297,
            1298,
            1299
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_gpu_pack_gate_up_down_finalize_tiled(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    size_t group_size,
    void* workspace_ptr,
    const void* input_hidden_ptr,
    const void* gate_up_proj_src_ptr,
    const void* gate_up_scale_src_ptr,
    const void* gate_up_zero_src_ptr,
    const void* down_proj_src_ptr,
    const void* down_scale_src_ptr,
    const void* down_zero_src_ptr,
    void* gate_up_proj_dst_ptr,
    void* gate_up_scale_dst_ptr,
    void* gate_up_zero_dst_ptr,
    void* down_proj_dst_ptr,
    void* down_scale_dst_ptr,
    void* down_zero_dst_ptr,
    void* output_ptr,
    size_t off_h_norm,
    size_t off_topk_val,
    size_t off_topk_idx,
    size_t off_shared_out,
    size_t off_expert_mid,
    size_t off_moe_out,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || group_size == 0 ||
            workspace_ptr == nullptr || input_hidden_ptr == nullptr ||
            gate_up_proj_src_ptr == nullptr || gate_up_scale_src_ptr == nullptr ||
            gate_up_zero_src_ptr == nullptr || down_proj_src_ptr == nullptr ||
            down_scale_src_ptr == nullptr || down_zero_src_ptr == nullptr ||
            gate_up_proj_dst_ptr == nullptr || gate_up_scale_dst_ptr == nullptr ||
            gate_up_zero_dst_ptr == nullptr || down_proj_dst_ptr == nullptr ||
            down_scale_dst_ptr == nullptr || down_zero_dst_ptr == nullptr || output_ptr == nullptr) {
            return 1230;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX ||
            top_k > UINT32_MAX || group_size > UINT32_MAX ||
            off_h_norm > UINT32_MAX || off_topk_val > UINT32_MAX ||
            off_topk_idx > UINT32_MAX || off_shared_out > UINT32_MAX ||
            off_expert_mid > UINT32_MAX || off_moe_out > UINT32_MAX) {
            return 1231;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0 ||
            (hidden % group_size) != 0 || (moe_intermediate % group_size) != 0 ||
            ((2 * moe_intermediate) % group_size) != 0) {
            return 1232;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_pack_u8 == nil || pipelines.expert_pack_bf16_pair == nil ||
            pipelines.expert_pack_remap_topk == nil || pipelines.expert_gate_up_tiled == nil ||
            pipelines.expert_down_finalize == nil) {
            return 1233;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> gate_up_proj_src = nil;
        id<MTLBuffer> gate_up_scale_src = nil;
        id<MTLBuffer> gate_up_zero_src = nil;
        id<MTLBuffer> down_proj_src = nil;
        id<MTLBuffer> down_scale_src = nil;
        id<MTLBuffer> down_zero_src = nil;
        id<MTLBuffer> gate_up_proj_dst = nil;
        id<MTLBuffer> gate_up_scale_dst = nil;
        id<MTLBuffer> gate_up_zero_dst = nil;
        id<MTLBuffer> down_proj_dst = nil;
        id<MTLBuffer> down_scale_dst = nil;
        id<MTLBuffer> down_zero_dst = nil;
        id<MTLBuffer> output = nil;
        size_t workspace_offset = 0;
        size_t input_hidden_offset = 0;
        size_t gate_up_proj_src_offset = 0;
        size_t gate_up_scale_src_offset = 0;
        size_t gate_up_zero_src_offset = 0;
        size_t down_proj_src_offset = 0;
        size_t down_scale_src_offset = 0;
        size_t down_zero_src_offset = 0;
        size_t gate_up_proj_dst_offset = 0;
        size_t gate_up_scale_dst_offset = 0;
        size_t gate_up_zero_dst_offset = 0;
        size_t down_proj_dst_offset = 0;
        size_t down_scale_dst_offset = 0;
        size_t down_zero_dst_offset = 0;
        size_t output_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1234;
        if (lookup_buffer(input_hidden_ptr, &input_hidden, &input_hidden_offset) != 0) return 1235;
        if (lookup_buffer(gate_up_proj_src_ptr, &gate_up_proj_src, &gate_up_proj_src_offset) != 0) return 1236;
        if (lookup_buffer(gate_up_scale_src_ptr, &gate_up_scale_src, &gate_up_scale_src_offset) != 0) return 1237;
        if (lookup_buffer(gate_up_zero_src_ptr, &gate_up_zero_src, &gate_up_zero_src_offset) != 0) return 1238;
        if (lookup_buffer(down_proj_src_ptr, &down_proj_src, &down_proj_src_offset) != 0) return 1239;
        if (lookup_buffer(down_scale_src_ptr, &down_scale_src, &down_scale_src_offset) != 0) return 1240;
        if (lookup_buffer(down_zero_src_ptr, &down_zero_src, &down_zero_src_offset) != 0) return 1241;
        if (lookup_buffer(gate_up_proj_dst_ptr, &gate_up_proj_dst, &gate_up_proj_dst_offset) != 0) return 1242;
        if (lookup_buffer(gate_up_scale_dst_ptr, &gate_up_scale_dst, &gate_up_scale_dst_offset) != 0) return 1243;
        if (lookup_buffer(gate_up_zero_dst_ptr, &gate_up_zero_dst, &gate_up_zero_dst_offset) != 0) return 1244;
        if (lookup_buffer(down_proj_dst_ptr, &down_proj_dst, &down_proj_dst_offset) != 0) return 1245;
        if (lookup_buffer(down_scale_dst_ptr, &down_scale_dst, &down_scale_dst_offset) != 0) return 1246;
        if (lookup_buffer(down_zero_dst_ptr, &down_zero_dst, &down_zero_dst_offset) != 0) return 1247;
        if (lookup_buffer(output_ptr, &output, &output_offset) != 0) return 1248;

        Qwen36FfnExpertPackParams gate_up_u8_params = {
            static_cast<uint32_t>(2 * moe_intermediate),
            static_cast<uint32_t>(hidden / 2),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(off_topk_idx),
        };
        Qwen36FfnExpertPackParams gate_up_sidecar_params = {
            static_cast<uint32_t>((2 * moe_intermediate) / group_size),
            static_cast<uint32_t>(hidden / group_size),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(off_topk_idx),
        };
        Qwen36FfnExpertPackParams down_u8_params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate / 2),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(off_topk_idx),
        };
        Qwen36FfnExpertPackParams down_sidecar_params = {
            static_cast<uint32_t>(hidden / group_size),
            static_cast<uint32_t>(moe_intermediate / group_size),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(off_topk_idx),
        };
        Qwen36FfnExpertGateUpTiledParams gate_params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_idx),
            static_cast<uint32_t>(off_expert_mid),
        };
        Qwen36FfnInt4Params down_params = {
            static_cast<uint32_t>(hidden),
            0u,
            static_cast<uint32_t>(moe_intermediate),
            0u,
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            static_cast<uint32_t>(off_topk_val),
            static_cast<uint32_t>(off_topk_idx),
            0u,
            0u,
            static_cast<uint32_t>(off_shared_out),
            static_cast<uint32_t>(off_expert_mid),
            static_cast<uint32_t>(off_moe_out),
        };

        auto encode = [&](id<MTLComputeCommandEncoder> encoder) {
            MTLSize threads = MTLSizeMake(256, 1, 1);
            auto encode_pack_u8 = [&](id<MTLBuffer> src, size_t src_offset,
                                      id<MTLBuffer> dst, size_t dst_offset,
                                      Qwen36FfnExpertPackParams& params) {
                size_t total = static_cast<size_t>(params.rows) * params.cols * params.top_k;
                [encoder setComputePipelineState:pipelines.expert_pack_u8];
                [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
                [encoder setBuffer:src offset:src_offset atIndex:1];
                [encoder setBuffer:dst offset:dst_offset atIndex:2];
                [encoder setBytes:&params length:sizeof(params) atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:threads];
            };
            auto encode_pack_bf16_pair = [&](id<MTLBuffer> scale_src, size_t scale_src_offset,
                                             id<MTLBuffer> zero_src, size_t zero_src_offset,
                                             id<MTLBuffer> scale_dst, size_t scale_dst_offset,
                                             id<MTLBuffer> zero_dst, size_t zero_dst_offset,
                                             Qwen36FfnExpertPackParams& params) {
                size_t total = static_cast<size_t>(params.rows) * params.cols * params.top_k;
                [encoder setComputePipelineState:pipelines.expert_pack_bf16_pair];
                [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
                [encoder setBuffer:scale_src offset:scale_src_offset atIndex:1];
                [encoder setBuffer:zero_src offset:zero_src_offset atIndex:2];
                [encoder setBuffer:scale_dst offset:scale_dst_offset atIndex:3];
                [encoder setBuffer:zero_dst offset:zero_dst_offset atIndex:4];
                [encoder setBytes:&params length:sizeof(params) atIndex:5];
                [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:threads];
            };

            encode_pack_u8(
                gate_up_proj_src, gate_up_proj_src_offset,
                gate_up_proj_dst, gate_up_proj_dst_offset,
                gate_up_u8_params
            );
            encode_pack_bf16_pair(
                gate_up_scale_src, gate_up_scale_src_offset,
                gate_up_zero_src, gate_up_zero_src_offset,
                gate_up_scale_dst, gate_up_scale_dst_offset,
                gate_up_zero_dst, gate_up_zero_dst_offset,
                gate_up_sidecar_params
            );
            encode_pack_u8(
                down_proj_src, down_proj_src_offset,
                down_proj_dst, down_proj_dst_offset,
                down_u8_params
            );
            encode_pack_bf16_pair(
                down_scale_src, down_scale_src_offset,
                down_zero_src, down_zero_src_offset,
                down_scale_dst, down_scale_dst_offset,
                down_zero_dst, down_zero_dst_offset,
                down_sidecar_params
            );
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_pack_remap_topk];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBytes:&down_sidecar_params length:sizeof(down_sidecar_params) atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(top_k, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_gate_up_tiled];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj_dst offset:gate_up_proj_dst_offset atIndex:1];
            [encoder setBuffer:gate_up_scale_dst offset:gate_up_scale_dst_offset atIndex:2];
            [encoder setBuffer:gate_up_zero_dst offset:gate_up_zero_dst_offset atIndex:3];
            [encoder setBytes:&gate_params length:sizeof(gate_params) atIndex:4];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_down_finalize];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
            [encoder setBuffer:down_proj_dst offset:down_proj_dst_offset atIndex:2];
            [encoder setBuffer:down_scale_dst offset:down_scale_dst_offset atIndex:3];
            [encoder setBuffer:down_zero_dst offset:down_zero_dst_offset atIndex:4];
            [encoder setBuffer:output offset:output_offset atIndex:5];
            [encoder setBytes:&down_params length:sizeof(down_params) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode,
                "qwen36_ffn_int4_expert_gpu_pack_gate_up_down_finalize_tiled",
                1249,
                1250,
                1251,
                1252
            );
        }
        return encode_or_submit_labeled_async(
            encode,
            "qwen36_ffn_int4_expert_gpu_pack_gate_up_down_finalize_tiled",
            1249,
            1250,
            1251,
            1252
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_int4_stage5(
    size_t hidden,
    size_t num_experts,
    size_t moe_intermediate,
    size_t shared_intermediate,
    size_t top_k,
    size_t group_size,
    const void* input_hidden_ptr,
    const void* shared_expert_gate_ptr,
    const void* shared_gate_proj_ptr,
    const void* shared_gate_scale_ptr,
    const void* shared_gate_zero_ptr,
    const void* shared_up_proj_ptr,
    const void* shared_up_scale_ptr,
    const void* shared_up_zero_ptr,
    const void* shared_down_proj_ptr,
    const void* shared_down_scale_ptr,
    const void* shared_down_zero_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* workspace_ptr,
    void* output_ptr,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || num_experts == 0 || moe_intermediate == 0 ||
            shared_intermediate == 0 || top_k == 0 || group_size == 0 ||
            input_hidden_ptr == nullptr || shared_expert_gate_ptr == nullptr ||
            shared_gate_proj_ptr == nullptr || shared_gate_scale_ptr == nullptr ||
            shared_gate_zero_ptr == nullptr || shared_up_proj_ptr == nullptr ||
            shared_up_scale_ptr == nullptr || shared_up_zero_ptr == nullptr ||
            shared_down_proj_ptr == nullptr || shared_down_scale_ptr == nullptr ||
            shared_down_zero_ptr == nullptr || gate_up_proj_ptr == nullptr ||
            gate_up_scale_ptr == nullptr || gate_up_zero_ptr == nullptr ||
            down_proj_ptr == nullptr || down_scale_ptr == nullptr ||
            down_zero_ptr == nullptr || workspace_ptr == nullptr || output_ptr == nullptr) {
            return 960;
        }
        if (hidden > UINT32_MAX || num_experts > UINT32_MAX ||
            moe_intermediate > UINT32_MAX || shared_intermediate > UINT32_MAX ||
            top_k > UINT32_MAX || group_size > UINT32_MAX) {
            return 961;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0 ||
            (shared_intermediate % 2) != 0) {
            return 962;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.shared_gate_up == nil || pipelines.shared_scalar == nil ||
            pipelines.shared_down == nil || pipelines.expert_gate_up_tiled == nil ||
            pipelines.expert_down_finalize == nil) {
            return 963;
        }

        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> shared_expert_gate = nil;
        id<MTLBuffer> shared_gate_proj = nil;
        id<MTLBuffer> shared_gate_scale = nil;
        id<MTLBuffer> shared_gate_zero = nil;
        id<MTLBuffer> shared_up_proj = nil;
        id<MTLBuffer> shared_up_scale = nil;
        id<MTLBuffer> shared_up_zero = nil;
        id<MTLBuffer> shared_down_proj = nil;
        id<MTLBuffer> shared_down_scale = nil;
        id<MTLBuffer> shared_down_zero = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        id<MTLBuffer> down_proj = nil;
        id<MTLBuffer> down_scale = nil;
        id<MTLBuffer> down_zero = nil;
        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> output = nil;
        size_t input_hidden_offset = 0;
        size_t shared_expert_gate_offset = 0;
        size_t shared_gate_proj_offset = 0;
        size_t shared_gate_scale_offset = 0;
        size_t shared_gate_zero_offset = 0;
        size_t shared_up_proj_offset = 0;
        size_t shared_up_scale_offset = 0;
        size_t shared_up_zero_offset = 0;
        size_t shared_down_proj_offset = 0;
        size_t shared_down_scale_offset = 0;
        size_t shared_down_zero_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        size_t down_proj_offset = 0;
        size_t down_scale_offset = 0;
        size_t down_zero_offset = 0;
        size_t workspace_offset = 0;
        size_t output_offset = 0;

        auto lookup_required = [](const void* ptr, id<MTLBuffer>* buffer, size_t* offset, int code) -> int {
            return lookup_buffer(ptr, buffer, offset) == 0 ? 0 : code;
        };
        int status = 0;
        if ((status = lookup_required(input_hidden_ptr, &input_hidden, &input_hidden_offset, 964)) != 0) return status;
        if ((status = lookup_required(shared_expert_gate_ptr, &shared_expert_gate, &shared_expert_gate_offset, 965)) != 0) return status;
        if ((status = lookup_required(shared_gate_proj_ptr, &shared_gate_proj, &shared_gate_proj_offset, 966)) != 0) return status;
        if ((status = lookup_required(shared_gate_scale_ptr, &shared_gate_scale, &shared_gate_scale_offset, 967)) != 0) return status;
        if ((status = lookup_required(shared_gate_zero_ptr, &shared_gate_zero, &shared_gate_zero_offset, 968)) != 0) return status;
        if ((status = lookup_required(shared_up_proj_ptr, &shared_up_proj, &shared_up_proj_offset, 969)) != 0) return status;
        if ((status = lookup_required(shared_up_scale_ptr, &shared_up_scale, &shared_up_scale_offset, 970)) != 0) return status;
        if ((status = lookup_required(shared_up_zero_ptr, &shared_up_zero, &shared_up_zero_offset, 971)) != 0) return status;
        if ((status = lookup_required(shared_down_proj_ptr, &shared_down_proj, &shared_down_proj_offset, 972)) != 0) return status;
        if ((status = lookup_required(shared_down_scale_ptr, &shared_down_scale, &shared_down_scale_offset, 973)) != 0) return status;
        if ((status = lookup_required(shared_down_zero_ptr, &shared_down_zero, &shared_down_zero_offset, 974)) != 0) return status;
        if ((status = lookup_required(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset, 975)) != 0) return status;
        if ((status = lookup_required(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset, 976)) != 0) return status;
        if ((status = lookup_required(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset, 977)) != 0) return status;
        if ((status = lookup_required(down_proj_ptr, &down_proj, &down_proj_offset, 978)) != 0) return status;
        if ((status = lookup_required(down_scale_ptr, &down_scale, &down_scale_offset, 979)) != 0) return status;
        if ((status = lookup_required(down_zero_ptr, &down_zero, &down_zero_offset, 980)) != 0) return status;
        if ((status = lookup_required(workspace_ptr, &workspace, &workspace_offset, 981)) != 0) return status;
        if ((status = lookup_required(output_ptr, &output, &output_offset, 982)) != 0) return status;

        uint32_t off_h_norm = 0u;
        uint32_t off_topk_val = static_cast<uint32_t>(hidden + 2 * num_experts);
        uint32_t off_topk_idx = static_cast<uint32_t>(hidden + 2 * num_experts + top_k);
        uint32_t off_sg_scalar = static_cast<uint32_t>(hidden + 2 * num_experts + 2 * top_k);
        uint32_t off_sgp = off_sg_scalar + 1u;
        uint32_t off_sup = off_sgp + static_cast<uint32_t>(shared_intermediate);
        uint32_t off_shared_mid = off_sup + static_cast<uint32_t>(shared_intermediate);
        uint32_t off_shared_out = off_shared_mid + static_cast<uint32_t>(shared_intermediate);
        uint32_t off_expert_gu = off_shared_out + static_cast<uint32_t>(hidden);
        uint32_t off_expert_mid = off_expert_gu + static_cast<uint32_t>(top_k * 2 * moe_intermediate);
        uint32_t off_expert_stack = off_expert_mid + static_cast<uint32_t>(top_k * moe_intermediate);
        uint32_t off_moe_out = off_expert_stack + static_cast<uint32_t>(top_k * hidden);

        Qwen36FfnInt4Params params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(num_experts),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(shared_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            off_h_norm,
            off_topk_val,
            off_topk_idx,
            off_sg_scalar,
            off_shared_mid,
            off_shared_out,
            off_expert_mid,
            off_moe_out,
        };
        Qwen36FfnExpertGateUpTiledParams expert_gate_params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            off_h_norm,
            off_topk_idx,
            off_expert_mid,
        };

        auto encode_shared_gate_up = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.shared_gate_up];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:shared_gate_proj offset:shared_gate_proj_offset atIndex:1];
            [encoder setBuffer:shared_gate_scale offset:shared_gate_scale_offset atIndex:2];
            [encoder setBuffer:shared_gate_zero offset:shared_gate_zero_offset atIndex:3];
            [encoder setBuffer:shared_up_proj offset:shared_up_proj_offset atIndex:4];
            [encoder setBuffer:shared_up_scale offset:shared_up_scale_offset atIndex:5];
            [encoder setBuffer:shared_up_zero offset:shared_up_zero_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(shared_intermediate, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_shared_scalar = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.shared_scalar];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:shared_expert_gate offset:shared_expert_gate_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_shared_down = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.shared_down];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:shared_down_proj offset:shared_down_proj_offset atIndex:1];
            [encoder setBuffer:shared_down_scale offset:shared_down_scale_offset atIndex:2];
            [encoder setBuffer:shared_down_zero offset:shared_down_zero_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_expert_gate_up = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_gate_up_tiled];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:1];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:2];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:3];
            [encoder setBytes:&expert_gate_params length:sizeof(expert_gate_params) atIndex:4];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto encode_expert_down_finalize = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_down_finalize];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
            [encoder setBuffer:down_proj offset:down_proj_offset atIndex:2];
            [encoder setBuffer:down_scale offset:down_scale_offset atIndex:3];
            [encoder setBuffer:down_zero offset:down_zero_offset atIndex:4];
            [encoder setBuffer:output offset:output_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };

        bool split_profile = qwen36_ffn_phase_profile_enabled();
        if (split_profile) {
            auto submit_profile_phase = [&](auto encode_fn, const std::string& label,
                                            int queue_error, int command_buffer_error,
                                            int encoder_error, int completion_error) -> int {
                int phase_status = encode_or_submit_labeled(
                    encode_fn,
                    label,
                    queue_error,
                    command_buffer_error,
                    encoder_error,
                    completion_error
                );
                if (phase_status != 0) {
                    return phase_status;
                }
                return flush_metal_batch_after_qwen36_ffn_profile_phase();
            };
            if ((status = submit_profile_phase(encode_shared_gate_up, "qwen36_ffn_int4_shared_gate_up", 983, 984, 985, 986)) != 0) return status;
            if ((status = submit_profile_phase(encode_shared_scalar, "qwen36_ffn_int4_shared_gate_scalar", 987, 988, 989, 990)) != 0) return status;
            if ((status = submit_profile_phase(encode_shared_down, "qwen36_ffn_int4_shared_down", 991, 992, 993, 994)) != 0) return status;
            if ((status = submit_profile_phase(encode_expert_gate_up, "qwen36_ffn_int4_expert_gate_up_tiled_stage5", 995, 996, 997, 998)) != 0) return status;
            return submit_profile_phase(
                encode_expert_down_finalize,
                "qwen36_ffn_int4_expert_down_finalize",
                999,
                1000,
                1001,
                1002
            );
        }

        auto encode_stage = [&](id<MTLComputeCommandEncoder> encoder) {
            encode_shared_gate_up(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_shared_scalar(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_shared_down(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_expert_gate_up(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_expert_down_finalize(encoder);
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode_stage,
                "qwen36_ffn_int4_stage5",
                1003,
                1004,
                1005,
                1006
            );
        }
        return encode_or_submit_labeled_async(
            encode_stage,
            "qwen36_ffn_int4_stage5",
            1003,
            1004,
            1005,
            1006
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_int4_stage5_with_router(
    size_t hidden,
    size_t num_experts,
    size_t moe_intermediate,
    size_t shared_intermediate,
    size_t top_k,
    size_t group_size,
    float rms_norm_eps,
    const void* input_hidden_ptr,
    const void* post_attn_norm_ptr,
    const void* gate_ptr,
    const void* shared_expert_gate_ptr,
    const void* shared_gate_proj_ptr,
    const void* shared_gate_scale_ptr,
    const void* shared_gate_zero_ptr,
    const void* shared_up_proj_ptr,
    const void* shared_up_scale_ptr,
    const void* shared_up_zero_ptr,
    const void* shared_down_proj_ptr,
    const void* shared_down_scale_ptr,
    const void* shared_down_zero_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* workspace_ptr,
    void* output_idx_ptr,
    void* output_ptr,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || num_experts == 0 || moe_intermediate == 0 ||
            shared_intermediate == 0 || top_k == 0 || group_size == 0 ||
            input_hidden_ptr == nullptr || post_attn_norm_ptr == nullptr ||
            gate_ptr == nullptr || shared_expert_gate_ptr == nullptr ||
            shared_gate_proj_ptr == nullptr || shared_gate_scale_ptr == nullptr ||
            shared_gate_zero_ptr == nullptr || shared_up_proj_ptr == nullptr ||
            shared_up_scale_ptr == nullptr || shared_up_zero_ptr == nullptr ||
            shared_down_proj_ptr == nullptr || shared_down_scale_ptr == nullptr ||
            shared_down_zero_ptr == nullptr || gate_up_proj_ptr == nullptr ||
            gate_up_scale_ptr == nullptr || gate_up_zero_ptr == nullptr ||
            down_proj_ptr == nullptr || down_scale_ptr == nullptr ||
            down_zero_ptr == nullptr || workspace_ptr == nullptr ||
            output_idx_ptr == nullptr || output_ptr == nullptr) {
            return 1390;
        }
        if (hidden > UINT32_MAX || num_experts > UINT32_MAX ||
            moe_intermediate > UINT32_MAX || shared_intermediate > UINT32_MAX ||
            top_k > UINT32_MAX || group_size > UINT32_MAX) {
            return 1391;
        }
        if ((hidden % 2) != 0 || (moe_intermediate % 2) != 0 ||
            (shared_intermediate % 2) != 0 || num_experts > 256 || top_k > 16 ||
            (hidden % group_size) != 0 || (moe_intermediate % group_size) != 0 ||
            ((2 * moe_intermediate) % group_size) != 0) {
            return 1392;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.router_stage5 == nil || pipelines.shared_gate_up == nil ||
            pipelines.shared_scalar == nil || pipelines.shared_down == nil ||
            pipelines.expert_gate_up_tiled == nil ||
            pipelines.expert_down_finalize == nil) {
            return 1393;
        }

        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> post_attn_norm = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> shared_expert_gate = nil;
        id<MTLBuffer> shared_gate_proj = nil;
        id<MTLBuffer> shared_gate_scale = nil;
        id<MTLBuffer> shared_gate_zero = nil;
        id<MTLBuffer> shared_up_proj = nil;
        id<MTLBuffer> shared_up_scale = nil;
        id<MTLBuffer> shared_up_zero = nil;
        id<MTLBuffer> shared_down_proj = nil;
        id<MTLBuffer> shared_down_scale = nil;
        id<MTLBuffer> shared_down_zero = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        id<MTLBuffer> down_proj = nil;
        id<MTLBuffer> down_scale = nil;
        id<MTLBuffer> down_zero = nil;
        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> output_idx = nil;
        id<MTLBuffer> output = nil;
        size_t input_hidden_offset = 0;
        size_t post_attn_norm_offset = 0;
        size_t gate_offset = 0;
        size_t shared_expert_gate_offset = 0;
        size_t shared_gate_proj_offset = 0;
        size_t shared_gate_scale_offset = 0;
        size_t shared_gate_zero_offset = 0;
        size_t shared_up_proj_offset = 0;
        size_t shared_up_scale_offset = 0;
        size_t shared_up_zero_offset = 0;
        size_t shared_down_proj_offset = 0;
        size_t shared_down_scale_offset = 0;
        size_t shared_down_zero_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        size_t down_proj_offset = 0;
        size_t down_scale_offset = 0;
        size_t down_zero_offset = 0;
        size_t workspace_offset = 0;
        size_t output_idx_offset = 0;
        size_t output_offset = 0;

        auto lookup_required = [](const void* ptr, id<MTLBuffer>* buffer, size_t* offset, int code) -> int {
            return lookup_buffer(ptr, buffer, offset) == 0 ? 0 : code;
        };
        int status = 0;
        if ((status = lookup_required(input_hidden_ptr, &input_hidden, &input_hidden_offset, 1394)) != 0) return status;
        if ((status = lookup_required(post_attn_norm_ptr, &post_attn_norm, &post_attn_norm_offset, 1395)) != 0) return status;
        if ((status = lookup_required(gate_ptr, &gate, &gate_offset, 1396)) != 0) return status;
        if ((status = lookup_required(shared_expert_gate_ptr, &shared_expert_gate, &shared_expert_gate_offset, 1397)) != 0) return status;
        if ((status = lookup_required(shared_gate_proj_ptr, &shared_gate_proj, &shared_gate_proj_offset, 1398)) != 0) return status;
        if ((status = lookup_required(shared_gate_scale_ptr, &shared_gate_scale, &shared_gate_scale_offset, 1399)) != 0) return status;
        if ((status = lookup_required(shared_gate_zero_ptr, &shared_gate_zero, &shared_gate_zero_offset, 1400)) != 0) return status;
        if ((status = lookup_required(shared_up_proj_ptr, &shared_up_proj, &shared_up_proj_offset, 1401)) != 0) return status;
        if ((status = lookup_required(shared_up_scale_ptr, &shared_up_scale, &shared_up_scale_offset, 1402)) != 0) return status;
        if ((status = lookup_required(shared_up_zero_ptr, &shared_up_zero, &shared_up_zero_offset, 1403)) != 0) return status;
        if ((status = lookup_required(shared_down_proj_ptr, &shared_down_proj, &shared_down_proj_offset, 1404)) != 0) return status;
        if ((status = lookup_required(shared_down_scale_ptr, &shared_down_scale, &shared_down_scale_offset, 1405)) != 0) return status;
        if ((status = lookup_required(shared_down_zero_ptr, &shared_down_zero, &shared_down_zero_offset, 1406)) != 0) return status;
        if ((status = lookup_required(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset, 1407)) != 0) return status;
        if ((status = lookup_required(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset, 1408)) != 0) return status;
        if ((status = lookup_required(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset, 1409)) != 0) return status;
        if ((status = lookup_required(down_proj_ptr, &down_proj, &down_proj_offset, 1410)) != 0) return status;
        if ((status = lookup_required(down_scale_ptr, &down_scale, &down_scale_offset, 1411)) != 0) return status;
        if ((status = lookup_required(down_zero_ptr, &down_zero, &down_zero_offset, 1412)) != 0) return status;
        if ((status = lookup_required(workspace_ptr, &workspace, &workspace_offset, 1413)) != 0) return status;
        if ((status = lookup_required(output_idx_ptr, &output_idx, &output_idx_offset, 1414)) != 0) return status;
        if ((status = lookup_required(output_ptr, &output, &output_offset, 1415)) != 0) return status;

        uint32_t off_h_norm = 0u;
        uint32_t off_topk_val = static_cast<uint32_t>(hidden + 2 * num_experts);
        uint32_t off_topk_idx = static_cast<uint32_t>(hidden + 2 * num_experts + top_k);
        uint32_t off_sg_scalar = static_cast<uint32_t>(hidden + 2 * num_experts + 2 * top_k);
        uint32_t off_sgp = off_sg_scalar + 1u;
        uint32_t off_sup = off_sgp + static_cast<uint32_t>(shared_intermediate);
        uint32_t off_shared_mid = off_sup + static_cast<uint32_t>(shared_intermediate);
        uint32_t off_shared_out = off_shared_mid + static_cast<uint32_t>(shared_intermediate);
        uint32_t off_expert_gu = off_shared_out + static_cast<uint32_t>(hidden);
        uint32_t off_expert_mid = off_expert_gu + static_cast<uint32_t>(top_k * 2 * moe_intermediate);
        uint32_t off_expert_stack = off_expert_mid + static_cast<uint32_t>(top_k * moe_intermediate);
        uint32_t off_moe_out = off_expert_stack + static_cast<uint32_t>(top_k * hidden);

        Qwen36FfnInt4Params params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(num_experts),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(shared_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            off_h_norm,
            off_topk_val,
            off_topk_idx,
            off_sg_scalar,
            off_shared_mid,
            off_shared_out,
            off_expert_mid,
            off_moe_out,
        };
        Qwen36FfnExpertGateUpTiledParams expert_gate_params = {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(moe_intermediate),
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            off_h_norm,
            off_topk_idx,
            off_expert_mid,
        };
        auto encode_router = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.router_stage5];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:0];
            [encoder setBuffer:post_attn_norm offset:post_attn_norm_offset atIndex:1];
            [encoder setBuffer:gate offset:gate_offset atIndex:2];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:3];
            [encoder setBuffer:output_idx offset:output_idx_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            [encoder setBytes:&rms_norm_eps length:sizeof(rms_norm_eps) atIndex:6];
            [encoder setThreadgroupMemoryLength:(256 + 16) * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto encode_shared_gate_up = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.shared_gate_up];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:shared_gate_proj offset:shared_gate_proj_offset atIndex:1];
            [encoder setBuffer:shared_gate_scale offset:shared_gate_scale_offset atIndex:2];
            [encoder setBuffer:shared_gate_zero offset:shared_gate_zero_offset atIndex:3];
            [encoder setBuffer:shared_up_proj offset:shared_up_proj_offset atIndex:4];
            [encoder setBuffer:shared_up_scale offset:shared_up_scale_offset atIndex:5];
            [encoder setBuffer:shared_up_zero offset:shared_up_zero_offset atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(shared_intermediate, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_shared_scalar = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.shared_scalar];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:shared_expert_gate offset:shared_expert_gate_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_shared_down = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.shared_down];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:shared_down_proj offset:shared_down_proj_offset atIndex:1];
            [encoder setBuffer:shared_down_scale offset:shared_down_scale_offset atIndex:2];
            [encoder setBuffer:shared_down_zero offset:shared_down_zero_offset atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto encode_expert_gate_up = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_gate_up_tiled];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:1];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:2];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:3];
            [encoder setBytes:&expert_gate_params length:sizeof(expert_gate_params) atIndex:4];
            [encoder setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto encode_expert_down_finalize = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_down_finalize];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
            [encoder setBuffer:down_proj offset:down_proj_offset atIndex:2];
            [encoder setBuffer:down_scale offset:down_scale_offset atIndex:3];
            [encoder setBuffer:down_zero offset:down_zero_offset atIndex:4];
            [encoder setBuffer:output offset:output_offset atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };

        bool split_profile = qwen36_ffn_phase_profile_enabled();
        if (split_profile) {
            auto submit_profile_phase = [&](auto encode_fn, const std::string& label,
                                            int queue_error, int command_buffer_error,
                                            int encoder_error, int completion_error) -> int {
                int phase_status = encode_or_submit_labeled(
                    encode_fn,
                    label,
                    queue_error,
                    command_buffer_error,
                    encoder_error,
                    completion_error
                );
                if (phase_status != 0) {
                    return phase_status;
                }
                return flush_metal_batch_after_qwen36_ffn_profile_phase();
            };
            if ((status = submit_profile_phase(encode_router, "qwen36_ffn_int4_router_topk_stage5", 1416, 1417, 1418, 1419)) != 0) return status;
            if ((status = submit_profile_phase(encode_shared_gate_up, "qwen36_ffn_int4_shared_gate_up", 1420, 1421, 1422, 1423)) != 0) return status;
            if ((status = submit_profile_phase(encode_shared_scalar, "qwen36_ffn_int4_shared_gate_scalar", 1424, 1425, 1426, 1427)) != 0) return status;
            if ((status = submit_profile_phase(encode_shared_down, "qwen36_ffn_int4_shared_down", 1428, 1429, 1430, 1431)) != 0) return status;
            if ((status = submit_profile_phase(encode_expert_gate_up, "qwen36_ffn_int4_expert_gate_up_tiled_stage5", 1432, 1433, 1434, 1435)) != 0) return status;
            return submit_profile_phase(
                encode_expert_down_finalize,
                "qwen36_ffn_int4_expert_down_finalize",
                1436,
                1437,
                1438,
                1439
            );
        }

        auto encode_stage = [&](id<MTLComputeCommandEncoder> encoder) {
            encode_router(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_shared_gate_up(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_shared_scalar(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_shared_down(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_expert_gate_up(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_expert_down_finalize(encoder);
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode_stage,
                "qwen36_ffn_int4_stage5_with_router",
                1440,
                1441,
                1442,
                1443
            );
        }
        return encode_or_submit_labeled_async(
            encode_stage,
            "qwen36_ffn_int4_stage5_with_router",
            1440,
            1441,
            1442,
            1443
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_mps_transcode_int4_f16(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    size_t group_size,
    void* workspace_ptr,
    const void* gate_up_proj_ptr,
    const void* gate_up_scale_ptr,
    const void* gate_up_zero_ptr,
    const void* down_proj_ptr,
    const void* down_scale_ptr,
    const void* down_zero_ptr,
    void* h_norm_f16_ptr,
    void* gate_up_rhs_f16_ptr,
    void* down_rhs_f16_ptr,
    size_t off_h_norm,
    size_t off_topk_idx,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || group_size == 0 ||
            workspace_ptr == nullptr || gate_up_proj_ptr == nullptr ||
            gate_up_scale_ptr == nullptr || gate_up_zero_ptr == nullptr ||
            down_proj_ptr == nullptr || down_scale_ptr == nullptr ||
            down_zero_ptr == nullptr || h_norm_f16_ptr == nullptr ||
            gate_up_rhs_f16_ptr == nullptr || down_rhs_f16_ptr == nullptr) {
            return 1240;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX || top_k > UINT32_MAX ||
            group_size > UINT32_MAX || off_h_norm > UINT32_MAX || off_topk_idx > UINT32_MAX) {
            return 1241;
        }
        if ((group_size % 2) != 0 || group_size < 2 || group_size > 128 ||
            (hidden % 2) != 0 || (moe_intermediate % 2) != 0) {
            return 1242;
        }

        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_mps_transcode_hnorm == nil ||
            pipelines.expert_mps_transcode_gate_up == nil ||
            pipelines.expert_mps_transcode_down == nil) {
            return 1243;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> gate_up_proj = nil;
        id<MTLBuffer> gate_up_scale = nil;
        id<MTLBuffer> gate_up_zero = nil;
        id<MTLBuffer> down_proj = nil;
        id<MTLBuffer> down_scale = nil;
        id<MTLBuffer> down_zero = nil;
        id<MTLBuffer> h_norm = nil;
        id<MTLBuffer> gate_up_rhs = nil;
        id<MTLBuffer> down_rhs = nil;
        size_t workspace_offset = 0;
        size_t gate_up_proj_offset = 0;
        size_t gate_up_scale_offset = 0;
        size_t gate_up_zero_offset = 0;
        size_t down_proj_offset = 0;
        size_t down_scale_offset = 0;
        size_t down_zero_offset = 0;
        size_t h_norm_offset = 0;
        size_t gate_up_rhs_offset = 0;
        size_t down_rhs_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1244;
        if (lookup_buffer(gate_up_proj_ptr, &gate_up_proj, &gate_up_proj_offset) != 0) return 1245;
        if (lookup_buffer(gate_up_scale_ptr, &gate_up_scale, &gate_up_scale_offset) != 0) return 1246;
        if (lookup_buffer(gate_up_zero_ptr, &gate_up_zero, &gate_up_zero_offset) != 0) return 1247;
        if (lookup_buffer(down_proj_ptr, &down_proj, &down_proj_offset) != 0) return 1248;
        if (lookup_buffer(down_scale_ptr, &down_scale, &down_scale_offset) != 0) return 1249;
        if (lookup_buffer(down_zero_ptr, &down_zero, &down_zero_offset) != 0) return 1250;
        if (lookup_buffer(h_norm_f16_ptr, &h_norm, &h_norm_offset) != 0) return 1251;
        if (lookup_buffer(gate_up_rhs_f16_ptr, &gate_up_rhs, &gate_up_rhs_offset) != 0) return 1252;
        if (lookup_buffer(down_rhs_f16_ptr, &down_rhs, &down_rhs_offset) != 0) return 1253;

        Qwen36FfnInt4Params params = {
            static_cast<uint32_t>(hidden),
            0u,
            static_cast<uint32_t>(moe_intermediate),
            0u,
            static_cast<uint32_t>(top_k),
            static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(off_h_norm),
            0u,
            static_cast<uint32_t>(off_topk_idx),
            0u,
            0u,
            0u,
            0u,
            0u,
        };

        auto encode = [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipelines.expert_mps_transcode_hnorm];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:h_norm offset:h_norm_offset atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(hidden, top_k, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_mps_transcode_gate_up];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:gate_up_proj offset:gate_up_proj_offset atIndex:1];
            [encoder setBuffer:gate_up_scale offset:gate_up_scale_offset atIndex:2];
            [encoder setBuffer:gate_up_zero offset:gate_up_zero_offset atIndex:3];
            [encoder setBuffer:gate_up_rhs offset:gate_up_rhs_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            [encoder setThreadgroupMemoryLength:16 * sizeof(uint16_t) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(2 * moe_intermediate, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(group_size / 2, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [encoder setComputePipelineState:pipelines.expert_mps_transcode_down];
            [encoder setBuffer:workspace offset:workspace_offset atIndex:0];
            [encoder setBuffer:down_proj offset:down_proj_offset atIndex:1];
            [encoder setBuffer:down_scale offset:down_scale_offset atIndex:2];
            [encoder setBuffer:down_zero offset:down_zero_offset atIndex:3];
            [encoder setBuffer:down_rhs offset:down_rhs_offset atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            [encoder setThreadgroupMemoryLength:16 * sizeof(uint16_t) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(hidden, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(group_size / 2, 1, 1)];
        };
        if (wait_for_completion != 0) {
            return encode_or_submit_labeled(
                encode,
                "qwen36_ffn_int4_expert_mps_transcode_int4_f16",
                1254,
                1255,
                1256,
                1257
            );
        }
        return encode_or_submit_labeled_async(
            encode,
            "qwen36_ffn_int4_expert_mps_transcode_int4_f16",
            1254,
            1255,
            1256,
            1257
        );
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_mps_bridge_f16(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    void* workspace_ptr,
    const void* input_hidden_ptr,
    const void* h_norm_f16_ptr,
    const void* gate_up_rhs_f16_ptr,
    void* gate_up_out_f16_ptr,
    void* down_lhs_f16_ptr,
    const void* down_rhs_f16_ptr,
    void* down_out_f16_ptr,
    void* output_ptr,
    size_t off_topk_val,
    size_t off_shared_out,
    size_t off_moe_out,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 ||
            workspace_ptr == nullptr || input_hidden_ptr == nullptr ||
            h_norm_f16_ptr == nullptr || gate_up_rhs_f16_ptr == nullptr ||
            gate_up_out_f16_ptr == nullptr || down_lhs_f16_ptr == nullptr ||
            down_rhs_f16_ptr == nullptr || down_out_f16_ptr == nullptr ||
            output_ptr == nullptr) {
            return 1220;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX || top_k > UINT32_MAX ||
            off_topk_val > UINT32_MAX || off_shared_out > UINT32_MAX ||
            off_moe_out > UINT32_MAX) {
            return 1221;
        }

        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil) {
            return 1222;
        }
        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_mps_silu == nil || pipelines.expert_mps_finalize == nil) {
            return 1223;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> h_norm = nil;
        id<MTLBuffer> gate_up_rhs = nil;
        id<MTLBuffer> gate_up_out = nil;
        id<MTLBuffer> down_lhs = nil;
        id<MTLBuffer> down_rhs = nil;
        id<MTLBuffer> down_out = nil;
        id<MTLBuffer> output = nil;
        size_t workspace_offset = 0;
        size_t input_hidden_offset = 0;
        size_t h_norm_offset = 0;
        size_t gate_up_rhs_offset = 0;
        size_t gate_up_out_offset = 0;
        size_t down_lhs_offset = 0;
        size_t down_rhs_offset = 0;
        size_t down_out_offset = 0;
        size_t output_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1224;
        if (lookup_buffer(input_hidden_ptr, &input_hidden, &input_hidden_offset) != 0) return 1225;
        if (lookup_buffer(h_norm_f16_ptr, &h_norm, &h_norm_offset) != 0) return 1226;
        if (lookup_buffer(gate_up_rhs_f16_ptr, &gate_up_rhs, &gate_up_rhs_offset) != 0) return 1227;
        if (lookup_buffer(gate_up_out_f16_ptr, &gate_up_out, &gate_up_out_offset) != 0) return 1228;
        if (lookup_buffer(down_lhs_f16_ptr, &down_lhs, &down_lhs_offset) != 0) return 1229;
        if (lookup_buffer(down_rhs_f16_ptr, &down_rhs, &down_rhs_offset) != 0) return 1230;
        if (lookup_buffer(down_out_f16_ptr, &down_out, &down_out_offset) != 0) return 1231;
        if (lookup_buffer(output_ptr, &output, &output_offset) != 0) return 1232;

        const NSUInteger h = static_cast<NSUInteger>(hidden);
        const NSUInteger i = static_cast<NSUInteger>(moe_intermediate);
        const NSUInteger k = static_cast<NSUInteger>(top_k);
        const NSUInteger gate_up_cols = 2 * i;
        const NSUInteger f16_size = sizeof(uint16_t);

        auto matrix_at = [](id<MTLBuffer> buffer,
                            NSUInteger offset,
                            NSUInteger rows,
                            NSUInteger cols) -> MPSMatrix* {
            MPSMatrixDescriptor* desc =
                [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                      columns:cols
                                                     rowBytes:cols * sizeof(uint16_t)
                                                     dataType:MPSDataTypeFloat16];
            return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:desc];
        };

        MPSMatrixMultiplication* gate_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                resultRows:1
                                             resultColumns:gate_up_cols
                                           interiorColumns:h
                                                    alpha:1.0
                                                     beta:0.0];
        MPSMatrixMultiplication* down_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                resultRows:1
                                             resultColumns:h
                                           interiorColumns:i
                                                    alpha:1.0
                                                     beta:0.0];
        if (gate_gemm == nil || down_gemm == nil) {
            return 1233;
        }

        Qwen36FfnInt4Params params = {
            static_cast<uint32_t>(hidden),
            0u,
            static_cast<uint32_t>(moe_intermediate),
            0u,
            static_cast<uint32_t>(top_k),
            0u,
            0u,
            static_cast<uint32_t>(off_topk_val),
            0u,
            0u,
            0u,
            static_cast<uint32_t>(off_shared_out),
            0u,
            static_cast<uint32_t>(off_moe_out),
        };

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 1234;
        }
        for (NSUInteger group = 0; group < k; ++group) {
            MPSMatrix* lhs = matrix_at(h_norm, h_norm_offset + group * h * f16_size, 1, h);
            MPSMatrix* rhs =
                matrix_at(gate_up_rhs, gate_up_rhs_offset + group * h * gate_up_cols * f16_size, h, gate_up_cols);
            MPSMatrix* out =
                matrix_at(gate_up_out, gate_up_out_offset + group * gate_up_cols * f16_size, 1, gate_up_cols);
            if (lhs == nil || rhs == nil || out == nil) {
                return 1235;
            }
            [gate_gemm encodeToCommandBuffer:command_buffer leftMatrix:lhs rightMatrix:rhs resultMatrix:out];
        }

        id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
        if (silu_encoder == nil) {
            return 1236;
        }
        [silu_encoder setComputePipelineState:pipelines.expert_mps_silu];
        [silu_encoder setBuffer:gate_up_out offset:gate_up_out_offset atIndex:0];
        [silu_encoder setBuffer:down_lhs offset:down_lhs_offset atIndex:1];
        [silu_encoder setBytes:&params length:sizeof(params) atIndex:2];
        [silu_encoder dispatchThreads:MTLSizeMake(i, k, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [silu_encoder endEncoding];

        for (NSUInteger group = 0; group < k; ++group) {
            MPSMatrix* lhs = matrix_at(down_lhs, down_lhs_offset + group * i * f16_size, 1, i);
            MPSMatrix* rhs =
                matrix_at(down_rhs, down_rhs_offset + group * i * h * f16_size, i, h);
            MPSMatrix* out =
                matrix_at(down_out, down_out_offset + group * h * f16_size, 1, h);
            if (lhs == nil || rhs == nil || out == nil) {
                return 1237;
            }
            [down_gemm encodeToCommandBuffer:command_buffer leftMatrix:lhs rightMatrix:rhs resultMatrix:out];
        }

        id<MTLComputeCommandEncoder> finalize_encoder = [command_buffer computeCommandEncoder];
        if (finalize_encoder == nil) {
            return 1238;
        }
        [finalize_encoder setComputePipelineState:pipelines.expert_mps_finalize];
        [finalize_encoder setBuffer:workspace offset:workspace_offset atIndex:0];
        [finalize_encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
        [finalize_encoder setBuffer:down_out offset:down_out_offset atIndex:2];
        [finalize_encoder setBuffer:output offset:output_offset atIndex:3];
        [finalize_encoder setBytes:&params length:sizeof(params) atIndex:4];
        [finalize_encoder dispatchThreads:MTLSizeMake(h, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [finalize_encoder endEncoding];

        auto start = MetalClock::now();
        if (wait_for_completion != 0) {
            const double elapsed_ms = wait_command_buffer_ms(command_buffer, start);
            if (elapsed_ms <= 0.0 || !std::isfinite(elapsed_ms)) {
                return 1239;
            }
            record_command_buffer_gpu_profile(command_buffer, "qwen36_ffn_int4_expert_mps_bridge_f16");
            return 0;
        }
        [command_buffer commit];
        return 0;
    }
}

extern "C" int supersonic_metal_qwen36_ffn_expert_mps_bridge_indexed_f16(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    size_t rhs_slots,
    void* workspace_ptr,
    const void* input_hidden_ptr,
    const void* h_norm_f16_ptr,
    const void* gate_up_rhs_f16_ptr,
    void* gate_up_out_f16_ptr,
    void* down_lhs_f16_ptr,
    const void* down_rhs_f16_ptr,
    void* down_out_f16_ptr,
    void* output_ptr,
    size_t off_topk_val,
    size_t off_topk_idx,
    size_t off_shared_out,
    size_t off_moe_out,
    int wait_for_completion
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || rhs_slots == 0 ||
            workspace_ptr == nullptr || input_hidden_ptr == nullptr ||
            h_norm_f16_ptr == nullptr || gate_up_rhs_f16_ptr == nullptr ||
            gate_up_out_f16_ptr == nullptr || down_lhs_f16_ptr == nullptr ||
            down_rhs_f16_ptr == nullptr || down_out_f16_ptr == nullptr ||
            output_ptr == nullptr) {
            return 1260;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX || top_k > UINT32_MAX ||
            rhs_slots > UINT32_MAX || off_topk_val > UINT32_MAX ||
            off_topk_idx > UINT32_MAX || off_shared_out > UINT32_MAX ||
            off_moe_out > UINT32_MAX) {
            return 1261;
        }

        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil) {
            return 1262;
        }
        NSError* pipeline_error = nil;
        Qwen36FfnInt4Pipelines pipelines = qwen36_ffn_int4_pipelines(&pipeline_error);
        if (pipelines.expert_mps_silu == nil || pipelines.expert_mps_finalize == nil) {
            return 1263;
        }

        id<MTLBuffer> workspace = nil;
        id<MTLBuffer> input_hidden = nil;
        id<MTLBuffer> h_norm = nil;
        id<MTLBuffer> gate_up_rhs = nil;
        id<MTLBuffer> gate_up_out = nil;
        id<MTLBuffer> down_lhs = nil;
        id<MTLBuffer> down_rhs = nil;
        id<MTLBuffer> down_out = nil;
        id<MTLBuffer> output = nil;
        size_t workspace_offset = 0;
        size_t input_hidden_offset = 0;
        size_t h_norm_offset = 0;
        size_t gate_up_rhs_offset = 0;
        size_t gate_up_out_offset = 0;
        size_t down_lhs_offset = 0;
        size_t down_rhs_offset = 0;
        size_t down_out_offset = 0;
        size_t output_offset = 0;
        if (lookup_buffer(workspace_ptr, &workspace, &workspace_offset) != 0) return 1264;
        if (lookup_buffer(input_hidden_ptr, &input_hidden, &input_hidden_offset) != 0) return 1265;
        if (lookup_buffer(h_norm_f16_ptr, &h_norm, &h_norm_offset) != 0) return 1266;
        if (lookup_buffer(gate_up_rhs_f16_ptr, &gate_up_rhs, &gate_up_rhs_offset) != 0) return 1267;
        if (lookup_buffer(gate_up_out_f16_ptr, &gate_up_out, &gate_up_out_offset) != 0) return 1268;
        if (lookup_buffer(down_lhs_f16_ptr, &down_lhs, &down_lhs_offset) != 0) return 1269;
        if (lookup_buffer(down_rhs_f16_ptr, &down_rhs, &down_rhs_offset) != 0) return 1270;
        if (lookup_buffer(down_out_f16_ptr, &down_out, &down_out_offset) != 0) return 1271;
        if (lookup_buffer(output_ptr, &output, &output_offset) != 0) return 1272;

        const NSUInteger h = static_cast<NSUInteger>(hidden);
        const NSUInteger i = static_cast<NSUInteger>(moe_intermediate);
        const NSUInteger k = static_cast<NSUInteger>(top_k);
        const NSUInteger gate_up_cols = 2 * i;
        const NSUInteger f16_size = sizeof(uint16_t);
        const float* workspace_host = static_cast<const float*>(workspace_ptr);

        auto slot_at = [&](NSUInteger group, uint32_t* slot_out) -> bool {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(float), "float bits must fit uint32_t");
            std::memcpy(&bits, workspace_host + off_topk_idx + group, sizeof(bits));
            if (bits >= rhs_slots) {
                return false;
            }
            *slot_out = bits;
            return true;
        };

        auto matrix_at = [](id<MTLBuffer> buffer,
                            NSUInteger offset,
                            NSUInteger rows,
                            NSUInteger cols) -> MPSMatrix* {
            MPSMatrixDescriptor* desc =
                [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                      columns:cols
                                                     rowBytes:cols * sizeof(uint16_t)
                                                     dataType:MPSDataTypeFloat16];
            return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:desc];
        };

        MPSMatrixMultiplication* gate_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                resultRows:1
                                             resultColumns:gate_up_cols
                                           interiorColumns:h
                                                    alpha:1.0
                                                     beta:0.0];
        MPSMatrixMultiplication* down_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                resultRows:1
                                             resultColumns:h
                                           interiorColumns:i
                                                    alpha:1.0
                                                     beta:0.0];
        if (gate_gemm == nil || down_gemm == nil) {
            return 1273;
        }

        Qwen36FfnInt4Params params = {
            static_cast<uint32_t>(hidden),
            0u,
            static_cast<uint32_t>(moe_intermediate),
            0u,
            static_cast<uint32_t>(top_k),
            0u,
            0u,
            static_cast<uint32_t>(off_topk_val),
            0u,
            0u,
            0u,
            static_cast<uint32_t>(off_shared_out),
            0u,
            static_cast<uint32_t>(off_moe_out),
        };

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return 1274;
        }
        for (NSUInteger group = 0; group < k; ++group) {
            uint32_t slot = 0;
            if (!slot_at(group, &slot)) {
                return 1275;
            }
            MPSMatrix* lhs = matrix_at(h_norm, h_norm_offset + group * h * f16_size, 1, h);
            MPSMatrix* rhs =
                matrix_at(gate_up_rhs, gate_up_rhs_offset + static_cast<NSUInteger>(slot) * h * gate_up_cols * f16_size, h, gate_up_cols);
            MPSMatrix* out =
                matrix_at(gate_up_out, gate_up_out_offset + group * gate_up_cols * f16_size, 1, gate_up_cols);
            if (lhs == nil || rhs == nil || out == nil) {
                return 1276;
            }
            [gate_gemm encodeToCommandBuffer:command_buffer leftMatrix:lhs rightMatrix:rhs resultMatrix:out];
        }

        id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
        if (silu_encoder == nil) {
            return 1277;
        }
        [silu_encoder setComputePipelineState:pipelines.expert_mps_silu];
        [silu_encoder setBuffer:gate_up_out offset:gate_up_out_offset atIndex:0];
        [silu_encoder setBuffer:down_lhs offset:down_lhs_offset atIndex:1];
        [silu_encoder setBytes:&params length:sizeof(params) atIndex:2];
        [silu_encoder dispatchThreads:MTLSizeMake(i, k, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [silu_encoder endEncoding];

        for (NSUInteger group = 0; group < k; ++group) {
            uint32_t slot = 0;
            if (!slot_at(group, &slot)) {
                return 1278;
            }
            MPSMatrix* lhs = matrix_at(down_lhs, down_lhs_offset + group * i * f16_size, 1, i);
            MPSMatrix* rhs =
                matrix_at(down_rhs, down_rhs_offset + static_cast<NSUInteger>(slot) * i * h * f16_size, i, h);
            MPSMatrix* out =
                matrix_at(down_out, down_out_offset + group * h * f16_size, 1, h);
            if (lhs == nil || rhs == nil || out == nil) {
                return 1279;
            }
            [down_gemm encodeToCommandBuffer:command_buffer leftMatrix:lhs rightMatrix:rhs resultMatrix:out];
        }

        id<MTLComputeCommandEncoder> finalize_encoder = [command_buffer computeCommandEncoder];
        if (finalize_encoder == nil) {
            return 1280;
        }
        [finalize_encoder setComputePipelineState:pipelines.expert_mps_finalize];
        [finalize_encoder setBuffer:workspace offset:workspace_offset atIndex:0];
        [finalize_encoder setBuffer:input_hidden offset:input_hidden_offset atIndex:1];
        [finalize_encoder setBuffer:down_out offset:down_out_offset atIndex:2];
        [finalize_encoder setBuffer:output offset:output_offset atIndex:3];
        [finalize_encoder setBytes:&params length:sizeof(params) atIndex:4];
        [finalize_encoder dispatchThreads:MTLSizeMake(h, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [finalize_encoder endEncoding];

        auto start = MetalClock::now();
        if (wait_for_completion != 0) {
            const double elapsed_ms = wait_command_buffer_ms(command_buffer, start);
            if (elapsed_ms <= 0.0 || !std::isfinite(elapsed_ms)) {
                return 1281;
            }
            record_command_buffer_gpu_profile(command_buffer, "qwen36_ffn_int4_expert_mps_static_topn_partial_f16");
            return 0;
        }
        [command_buffer commit];
        return 0;
    }
}

extern "C" int supersonic_metal_qwen36_mps_expert_f16_probe(
    size_t hidden,
    size_t moe_intermediate,
    size_t top_k,
    uint32_t iterations,
    double* gate_up_ms_out,
    double* down_ms_out,
    double* gate_up_tflops_out,
    double* down_tflops_out
) {
    @autoreleasepool {
        if (hidden == 0 || moe_intermediate == 0 || top_k == 0 || iterations == 0 ||
            gate_up_ms_out == nullptr || down_ms_out == nullptr ||
            gate_up_tflops_out == nullptr || down_tflops_out == nullptr) {
            return 1210;
        }
        if (hidden > UINT32_MAX || moe_intermediate > UINT32_MAX || top_k > UINT32_MAX) {
            return 1211;
        }
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil) {
            return 1212;
        }

        const NSUInteger m = static_cast<NSUInteger>(top_k);
        const NSUInteger hidden_cols = static_cast<NSUInteger>(hidden);
        const NSUInteger gate_up_cols = static_cast<NSUInteger>(2 * moe_intermediate);
        const NSUInteger down_k = static_cast<NSUInteger>(moe_intermediate);
        const NSUInteger down_cols = static_cast<NSUInteger>(hidden);

        auto make_buffer = [&](NSUInteger elements) -> id<MTLBuffer> {
            return [device newBufferWithLength:elements * sizeof(uint16_t)
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> gate_lhs_buf = make_buffer(m * hidden_cols);
        id<MTLBuffer> gate_rhs_buf = make_buffer(hidden_cols * gate_up_cols);
        id<MTLBuffer> gate_out_buf = make_buffer(m * gate_up_cols);
        id<MTLBuffer> down_lhs_buf = make_buffer(m * down_k);
        id<MTLBuffer> down_rhs_buf = make_buffer(down_k * down_cols);
        id<MTLBuffer> down_out_buf = make_buffer(m * down_cols);
        if (gate_lhs_buf == nil || gate_rhs_buf == nil || gate_out_buf == nil ||
            down_lhs_buf == nil || down_rhs_buf == nil || down_out_buf == nil) {
            return 1213;
        }

        auto fill_half = [](id<MTLBuffer> buffer, NSUInteger elements, uint16_t base) {
            auto* ptr = static_cast<uint16_t*>(buffer.contents);
            for (NSUInteger i = 0; i < elements; ++i) {
                ptr[i] = static_cast<uint16_t>(base + (i & 1u));
            }
        };
        fill_half(gate_lhs_buf, m * hidden_cols, 0x3c00u);
        fill_half(gate_rhs_buf, hidden_cols * gate_up_cols, 0x3800u);
        memset(gate_out_buf.contents, 0, m * gate_up_cols * sizeof(uint16_t));
        fill_half(down_lhs_buf, m * down_k, 0x3c00u);
        fill_half(down_rhs_buf, down_k * down_cols, 0x3800u);
        memset(down_out_buf.contents, 0, m * down_cols * sizeof(uint16_t));

        auto matrix = [](id<MTLBuffer> buffer, NSUInteger rows, NSUInteger cols) -> MPSMatrix* {
            MPSMatrixDescriptor* desc =
                [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                      columns:cols
                                                     rowBytes:cols * sizeof(uint16_t)
                                                     dataType:MPSDataTypeFloat16];
            return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
        };

        MPSMatrix* gate_lhs = matrix(gate_lhs_buf, m, hidden_cols);
        MPSMatrix* gate_rhs = matrix(gate_rhs_buf, hidden_cols, gate_up_cols);
        MPSMatrix* gate_out = matrix(gate_out_buf, m, gate_up_cols);
        MPSMatrix* down_lhs = matrix(down_lhs_buf, m, down_k);
        MPSMatrix* down_rhs = matrix(down_rhs_buf, down_k, down_cols);
        MPSMatrix* down_out = matrix(down_out_buf, m, down_cols);
        MPSMatrixMultiplication* gate_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                resultRows:m
                                             resultColumns:gate_up_cols
                                           interiorColumns:hidden_cols
                                                    alpha:1.0
                                                     beta:0.0];
        MPSMatrixMultiplication* down_gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:false
                                             transposeRight:false
                                                resultRows:m
                                             resultColumns:down_cols
                                           interiorColumns:down_k
                                                    alpha:1.0
                                                     beta:0.0];
        if (gate_lhs == nil || gate_rhs == nil || gate_out == nil ||
            down_lhs == nil || down_rhs == nil || down_out == nil ||
            gate_gemm == nil || down_gemm == nil) {
            return 1214;
        }

        auto run_mps = [&](MPSMatrixMultiplication* gemm,
                           MPSMatrix* lhs,
                           MPSMatrix* rhs,
                           MPSMatrix* out,
                           const char* op) -> double {
            id<MTLCommandBuffer> warm = [queue commandBuffer];
            if (warm == nil) {
                return 0.0;
            }
            [gemm encodeToCommandBuffer:warm leftMatrix:lhs rightMatrix:rhs resultMatrix:out];
            [warm commit];
            [warm waitUntilCompleted];
            if (warm.status != MTLCommandBufferStatusCompleted) {
                return 0.0;
            }

            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            if (command_buffer == nil) {
                return 0.0;
            }
            for (uint32_t i = 0; i < iterations; ++i) {
                [gemm encodeToCommandBuffer:command_buffer leftMatrix:lhs rightMatrix:rhs resultMatrix:out];
            }
            auto start = MetalClock::now();
            const double elapsed_ms = wait_command_buffer_ms(command_buffer, start);
            record_profile_elapsed(op, "native", elapsed_ms);
            return elapsed_ms;
        };

        const double gate_up_ms =
            run_mps(gate_gemm, gate_lhs, gate_rhs, gate_out, "qwen36_mps_expert_gate_up_f16_probe");
        const double down_ms =
            run_mps(down_gemm, down_lhs, down_rhs, down_out, "qwen36_mps_expert_down_f16_probe");
        if (gate_up_ms <= 0.0 || down_ms <= 0.0 ||
            !std::isfinite(gate_up_ms) || !std::isfinite(down_ms)) {
            return 1215;
        }

        volatile uint16_t guard =
            static_cast<uint16_t*>(gate_out_buf.contents)[0] ^
            static_cast<uint16_t*>(down_out_buf.contents)[0];
        (void)guard;

        const double gate_up_flops = static_cast<double>(iterations) * 2.0 *
            static_cast<double>(top_k) * static_cast<double>(2 * moe_intermediate) *
            static_cast<double>(hidden);
        const double down_flops = static_cast<double>(iterations) * 2.0 *
            static_cast<double>(top_k) * static_cast<double>(hidden) *
            static_cast<double>(moe_intermediate);
        *gate_up_ms_out = gate_up_ms;
        *down_ms_out = down_ms;
        *gate_up_tflops_out = gate_up_flops / (gate_up_ms / 1000.0) / 1.0e12;
        *down_tflops_out = down_flops / (down_ms / 1000.0) / 1.0e12;
        return 0;
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

extern "C" double supersonic_metal_mpp_tile_gemm_f16_tflops(uint32_t size, uint32_t iterations) {
#if SUPERSONIC_HAVE_MTL4_MPP
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        if (device == nil || size == 0 || iterations == 0 || (size % 64u) != 0u) {
            return 0.0;
        }
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        if (queue == nil) {
            return 0.0;
        }

        NSString* source =
             @"#include <metal_stdlib>\n"
             "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
             "using namespace metal;\n"
             "using namespace mpp::tensor_ops;\n"
             "kernel void supersonic_mpp_gemm_tile(tensor<device half, dextents<int32_t, 2>> A [[buffer(0)]],\n"
             "                                      tensor<device half, dextents<int32_t, 2>> B [[buffer(1)]],\n"
             "                                      tensor<device float, dextents<int32_t, 2>> C [[buffer(2)]]) {\n"
             "  constexpr auto desc = matmul2d_descriptor(64, 32, 64, false, false, false);\n"
             "  matmul2d<desc, execution_simdgroups<4>> op;\n"
             "  auto tA = A.slice(0, 0);\n"
             "  auto tB = B.slice(0, 0);\n"
             "  auto tC = C.slice(0, 0);\n"
             "  op.run(tA, tB, tC);\n"
             "}\n";
        id<MTLComputePipelineState> pipeline =
            mpp_mtl4_pipeline_from_source(source, @"supersonic_mpp_gemm_tile", 128);
        if (pipeline == nil) {
            return 0.0;
        }

        const NSUInteger a_count = 64u * 64u;
        const NSUInteger b_count = 32u * 64u;
        id<MTLBuffer> a_buf = [device newBufferWithLength:a_count * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [device newBufferWithLength:b_count * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        if (a_buf == nil || b_buf == nil) {
            return 0.0;
        }
        auto* a = static_cast<uint16_t*>(a_buf.contents);
        auto* b = static_cast<uint16_t*>(b_buf.contents);
        for (NSUInteger i = 0; i < a_count; ++i) {
            a[i] = static_cast<uint16_t>(0x3c00u + (i & 1u));
        }
        for (NSUInteger i = 0; i < b_count; ++i) {
            b[i] = static_cast<uint16_t>(0x3800u + (i & 1u));
        }

        id<MTLTensor> a_tensor = mpp_tensor_from_device(
            device,
            mpp_device_tensor_descriptor(64, 64, MTLTensorDataTypeFloat16)
        );
        id<MTLTensor> b_tensor = mpp_tensor_from_device(
            device,
            mpp_device_tensor_descriptor(32, 64, MTLTensorDataTypeFloat16)
        );
        id<MTLTensor> c_tensor = mpp_tensor_from_device(
            device,
            mpp_device_tensor_descriptor(32, 64, MTLTensorDataTypeFloat32)
        );
        if (a_tensor == nil || b_tensor == nil || c_tensor == nil) {
            return 0.0;
        }
        mpp_tensor_replace_all_f16(a_tensor, a, 64, 64);
        mpp_tensor_replace_all_f16(b_tensor, b, 32, 64);

        id<MTL4ArgumentTable> args = mpp_mtl4_argument_table(device, 3);
        id<MTLSharedEvent> event = [device newSharedEvent];
        if (args == nil || event == nil) {
            return 0.0;
        }
        [args setResource:a_tensor.gpuResourceID atBufferIndex:0];
        [args setResource:b_tensor.gpuResourceID atBufferIndex:1];
        [args setResource:c_tensor.gpuResourceID atBufferIndex:2];

        const uint32_t tile_m = size / 64u;
        const uint32_t tile_n = size / 32u;
        const uint32_t tile_k = size / 64u;
        const uint32_t tg_x = tile_n;
        const uint32_t tg_y = tile_m * tile_k;
        const NSUInteger threads_per_threadgroup = pipeline.threadExecutionWidth * 4u;

        if (!mpp_encode_gemm_mtl4(
                device, queue, pipeline, args, tg_x, tg_y, threads_per_threadgroup, event, 1)) {
            return 0.0;
        }

        auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < iterations; ++i) {
            const uint64_t signal_value = static_cast<uint64_t>(i) + 2u;
            if (!mpp_encode_gemm_mtl4(
                    device, queue, pipeline, args, tg_x, tg_y, threads_per_threadgroup, event, signal_value)) {
                return 0.0;
            }
        }

        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        volatile float guard = mpp_tensor_first_f32(c_tensor);
        if (guard == 0.0f || !std::isfinite(static_cast<double>(guard))) {
            return 0.0;
        }
        (void)guard;

        const double flops = static_cast<double>(iterations)
            * 2.0
            * static_cast<double>(tile_m)
            * static_cast<double>(tile_n)
            * static_cast<double>(tile_k)
            * 64.0
            * 32.0
            * 64.0;
        return flops / seconds / 1.0e12;
    }
#else
    (void)size;
    (void)iterations;
    return 0.0;
#endif
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

extern "C" int supersonic_metal_full_attention_prefill_tmajor_bf16_f32(
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
            query_ptr == nullptr || key_ptr == nullptr || value_ptr == nullptr || out_ptr == nullptr ||
            (q_heads % kv_heads) != 0) {
            return 211;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = full_attention_tmajor_pipeline_bf16_f32(&pipeline_error);
        if (pipeline == nil) {
            return 212;
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
            return 213;
        }
        if (lookup_buffer(key_ptr, &key, &key_offset) != 0) {
            return 214;
        }
        if (lookup_buffer(value_ptr, &value, &value_offset) != 0) {
            return 215;
        }
        if (lookup_buffer(out_ptr, &out, &out_offset) != 0) {
            return 216;
        }

        FullAttentionParams params = {
            static_cast<uint32_t>(q_heads),
            static_cast<uint32_t>(kv_heads),
            static_cast<uint32_t>(q_len),
            static_cast<uint32_t>(kv_len),
            static_cast<uint32_t>(kv_len),
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
            MTLSize threads_per_grid = MTLSizeMake(head_dim, q_heads, q_len);
            [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
        }, 217, 218, 219, 220);
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
