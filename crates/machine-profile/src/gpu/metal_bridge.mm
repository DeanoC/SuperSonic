#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdint.h>
#include <string>

namespace {

struct MpMetalDeviceInfo {
    uint64_t total_vram_bytes;
    uint64_t recommended_working_set_bytes;
    uint32_t core_count;
    uint32_t wave_size;
    uint64_t max_threadgroup_memory_bytes;
    uint64_t max_threads_per_threadgroup;
};

struct MpMetalMppProbeInfo {
    int32_t status;
    float tensor_write_value;
    float matmul_value;
};

id<MTLDevice> metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> metal_queue() {
    static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
    return queue;
}

std::string normalized_arch_name(id<MTLDevice> device) {
    NSString* name = device ? device.name : nil;
    if (name == nil) {
        return "apple-gpu";
    }
    NSString* lowered = [[name lowercaseString] stringByReplacingOccurrencesOfString:@" " withString:@"-"];
    return std::string([lowered UTF8String]);
}

void copy_cstr(const std::string& src, char* out, size_t out_len) {
    if (out == nullptr || out_len == 0) {
        return;
    }
    const size_t n = std::min(out_len - 1, src.size());
    memcpy(out, src.data(), n);
    out[n] = '\0';
}

NSDictionary* system_profiler_gpu_entry() {
    @autoreleasepool {
        NSTask* task = [[NSTask alloc] init];
        task.executableURL = [NSURL fileURLWithPath:@"/usr/sbin/system_profiler"];
        task.arguments = @[ @"SPDisplaysDataType", @"-json" ];
        NSPipe* pipe = [NSPipe pipe];
        task.standardOutput = pipe;
        task.standardError = [NSPipe pipe];
        NSError* launch_error = nil;
        if (![task launchAndReturnError:&launch_error]) {
            return nil;
        }
        [task waitUntilExit];
        if (task.terminationStatus != 0) {
            return nil;
        }
        NSData* data = [[pipe fileHandleForReading] readDataToEndOfFile];
        if (data.length == 0) {
            return nil;
        }
        NSError* json_error = nil;
        NSDictionary* root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&json_error];
        if (![root isKindOfClass:[NSDictionary class]]) {
            return nil;
        }
        NSArray* displays = root[@"SPDisplaysDataType"];
        if (![displays isKindOfClass:[NSArray class]]) {
            return nil;
        }
        for (NSDictionary* entry in displays) {
            NSString* type = entry[@"sppci_device_type"];
            if ([type isKindOfClass:[NSString class]] && [type containsString:@"gpu"]) {
                return entry;
            }
        }
        return nil;
    }
}

uint32_t system_profiler_gpu_cores(NSDictionary* entry) {
    NSString* cores = entry[@"sppci_cores"];
    if ([cores isKindOfClass:[NSString class]]) {
        return static_cast<uint32_t>(cores.intValue);
    }
    NSNumber* n = entry[@"sppci_cores"];
    if ([n isKindOfClass:[NSNumber class]]) {
        return static_cast<uint32_t>(n.unsignedIntValue);
    }
    return 0;
}

std::string system_profiler_metal_support(NSDictionary* entry) {
    NSString* support = entry[@"spdisplays_mtlgpufamilysupport"];
    if (![support isKindOfClass:[NSString class]]) {
        return "Apple GPU Family";
    }
    if ([support isEqualToString:@"spdisplays_metal4"]) {
        return "Metal 4";
    }
    return std::string([support UTF8String]);
}

id<MTLComputePipelineState> pipeline_from_source(NSString* source, NSString* name) {
    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if SUPERSONIC_HAVE_MTL4_MPP
    options.languageVersion = MTLLanguageVersion4_0;
#endif
    id<MTLLibrary> library = [metal_device() newLibraryWithSource:source options:options error:&error];
    if (library == nil || error != nil) {
        if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
            NSLog(@"machine-profile Metal compile failed for %@: %@", name, error);
        }
        return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
            NSLog(@"machine-profile Metal function %@ not found", name);
        }
        return nil;
    }
    id<MTLComputePipelineState> pipeline = [metal_device() newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"machine-profile Metal pipeline failed for %@: %@", name, error);
    }
    return pipeline;
}

#if SUPERSONIC_HAVE_MTL4_MPP
id<MTLComputePipelineState> mtl4_pipeline_from_source(
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
            NSLog(@"machine-profile Metal4 compile failed for %@: %@", name, error);
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
            NSLog(@"machine-profile Metal4 compiler failed for %@: %@", name, error);
        }
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [compiler newComputePipelineStateWithDescriptor:pipeline_desc compilerTaskOptions:nil error:&error];
    if (pipeline == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"machine-profile Metal4 pipeline failed for %@: %@", name, error);
    }
    return pipeline;
}
#endif  // SUPERSONIC_HAVE_MTL4_MPP

void copy_probe_status(const char* src, char* out, size_t out_len) {
    if (out == nullptr || out_len == 0) {
        return;
    }
    const size_t n = std::min(out_len - 1, strlen(src));
    memcpy(out, src, n);
    out[n] = '\0';
}

double command_elapsed_seconds(id<MTLCommandBuffer> command_buffer, std::chrono::steady_clock::time_point start) {
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return 0.0;
    }
    const CFTimeInterval gpu_start = command_buffer.GPUStartTime;
    const CFTimeInterval gpu_end = command_buffer.GPUEndTime;
    if (gpu_end > gpu_start && gpu_start > 0.0) {
        return static_cast<double>(gpu_end - gpu_start);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double run_unary_bandwidth_kernel(NSString* function_name, NSString* source, uint64_t bytes, bool read_kernel) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || bytes < 4096) {
            return 0.0;
        }
        id<MTLComputePipelineState> pipeline = pipeline_from_source(source, function_name);
        if (pipeline == nil) {
            return 0.0;
        }
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sink = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (buf == nil || sink == nil) {
            return 0.0;
        }
        memset(buf.contents, 1, bytes);
        memset(sink.contents, 0, sizeof(uint32_t));
        const uint32_t n_words = static_cast<uint32_t>(bytes / sizeof(uint32_t));
        id<MTLBuffer> params = [device newBufferWithBytes:&n_words length:sizeof(n_words) options:MTLResourceStorageModeShared];
        const NSUInteger threads = 65536;
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf offset:0 atIndex:0];
        [encoder setBuffer:sink offset:0 atIndex:1];
        [encoder setBuffer:params offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        const double traffic = read_kernel ? static_cast<double>(bytes) : static_cast<double>(bytes);
        return traffic / seconds / 1.0e9;
    }
}

#if SUPERSONIC_HAVE_MTL4_MPP
MTLTensorDescriptor* tensor_descriptor(NSUInteger dim0, NSUInteger dim1, MTLTensorDataType data_type) {
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

MTLTensorDescriptor* device_tensor_descriptor(NSUInteger dim0, NSUInteger dim1, MTLTensorDataType data_type) {
    MTLTensorDescriptor* desc = tensor_descriptor(dim0, dim1, data_type);
    desc.strides = nil;
    return desc;
}

id<MTLTensor> tensor_from_device(id<MTLDevice> device, MTLTensorDescriptor* desc) {
    NSError* error = nil;
    id<MTLTensor> tensor = [device newTensorWithDescriptor:desc error:&error];
    if (tensor == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"machine-profile Metal device tensor creation failed: %@", error);
    }
    return tensor;
}

void tensor_replace_all_f16(id<MTLTensor> tensor, const uint16_t* values, NSUInteger dim0, NSUInteger dim1) {
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

float tensor_first_f32(id<MTLTensor> tensor) {
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

id<MTL4ArgumentTable> mtl4_argument_table(id<MTLDevice> device, NSUInteger buffer_bind_count) {
    MTL4ArgumentTableDescriptor* desc = [[MTL4ArgumentTableDescriptor alloc] init];
    desc.maxBufferBindCount = buffer_bind_count;
    desc.initializeBindings = YES;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:desc error:&error];
    if (table == nil && NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
        NSLog(@"machine-profile Metal4 argument table failed: %@", error);
    }
    return table;
}

bool wait_for_mtl4_queue(id<MTL4CommandQueue> queue, id<MTLSharedEvent> event, uint64_t value) {
    [queue signalEvent:event value:value];
    return [event waitUntilSignaledValue:value timeoutMS:30000];
}

bool encode_mpp_gemm_mtl4(
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
    return wait_for_mtl4_queue(queue, event, signal_value);
}
#endif  // SUPERSONIC_HAVE_MTL4_MPP

}  // namespace

extern "C" int mp_metal_query_device_info(
    char* arch_name_out,
    size_t arch_name_len,
    char* device_name_out,
    size_t device_name_len,
    char* family_out,
    size_t family_len,
    MpMetalDeviceInfo* info_out
) {
    @autoreleasepool {
        if (arch_name_out == nullptr || arch_name_len == 0 || device_name_out == nullptr ||
            device_name_len == 0 || family_out == nullptr || family_len == 0 || info_out == nullptr) {
            return 1;
        }
        id<MTLDevice> device = metal_device();
        if (device == nil) {
            return 2;
        }

        const std::string arch = normalized_arch_name(device);
        NSString* name = device.name ?: @"Apple GPU";
        copy_cstr(arch, arch_name_out, arch_name_len);
        copy_cstr(std::string([name UTF8String]), device_name_out, device_name_len);
        NSDictionary* profiler_entry = system_profiler_gpu_entry();
        copy_cstr(system_profiler_metal_support(profiler_entry), family_out, family_len);

        uint64_t working_set = 0;
        if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            working_set = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        }
        if (working_set == 0) {
            working_set = static_cast<uint64_t>(NSProcessInfo.processInfo.physicalMemory);
        }

        MTLSize max_threads = device.maxThreadsPerThreadgroup;
        info_out->total_vram_bytes = working_set;
        info_out->recommended_working_set_bytes = working_set;
        info_out->core_count = system_profiler_gpu_cores(profiler_entry);
        info_out->wave_size = 32;
        info_out->max_threadgroup_memory_bytes = static_cast<uint64_t>(device.maxThreadgroupMemoryLength);
        info_out->max_threads_per_threadgroup = static_cast<uint64_t>(max_threads.width);
        return 0;
    }
}

extern "C" double mp_metal_unified_read_gb_s(uint64_t bytes) {
    NSString* source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void mp_read(device const uint* buf [[buffer(0)]],\n"
         "                    device atomic_uint* sink [[buffer(1)]],\n"
         "                    constant uint& n [[buffer(2)]],\n"
         "                    uint tid [[thread_position_in_grid]],\n"
         "                    uint grid [[threads_per_grid]]) {\n"
         "  uint acc = 0;\n"
         "  for (uint i = tid; i < n; i += grid) acc += buf[i];\n"
         "  atomic_fetch_add_explicit(sink, acc, memory_order_relaxed);\n"
         "}\n";
    return run_unary_bandwidth_kernel(@"mp_read", source, bytes, true);
}

extern "C" double mp_metal_unified_write_gb_s(uint64_t bytes) {
    NSString* source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void mp_write(device uint* buf [[buffer(0)]],\n"
         "                     device atomic_uint* sink [[buffer(1)]],\n"
         "                     constant uint& n [[buffer(2)]],\n"
         "                     uint tid [[thread_position_in_grid]],\n"
         "                     uint grid [[threads_per_grid]]) {\n"
         "  for (uint i = tid; i < n; i += grid) buf[i] = i;\n"
         "  if (tid == 0) atomic_store_explicit(sink, 1, memory_order_relaxed);\n"
         "}\n";
    return run_unary_bandwidth_kernel(@"mp_write", source, bytes, false);
}

extern "C" double mp_metal_unified_copy_gb_s(uint64_t bytes) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || bytes < 4096) {
            return 0.0;
        }
        id<MTLBuffer> src = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dst = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (src == nil || dst == nil) {
            return 0.0;
        }
        memset(src.contents, 3, bytes);
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        [encoder copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
        [encoder endEncoding];
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        return static_cast<double>(bytes) / seconds / 1.0e9;
    }
}

extern "C" double mp_metal_threadgroup_gb_s(uint32_t core_count, uint32_t iterations) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || core_count == 0 || iterations == 0) {
            return 0.0;
        }
        NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void mp_tg(device uint* out [[buffer(0)]],\n"
             "                  constant uint& iters [[buffer(1)]],\n"
             "                  uint tid [[thread_position_in_threadgroup]],\n"
             "                  uint gid [[threadgroup_position_in_grid]]) {\n"
             "  threadgroup uint scratch[256];\n"
             "  uint v = tid + gid;\n"
             "  for (uint i = 0; i < iters; ++i) {\n"
             "    scratch[tid] = v + i;\n"
             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
             "    v += scratch[(tid + 17) & 255];\n"
             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
             "  }\n"
             "  if (tid == 0) out[gid] = v;\n"
             "}\n";
        id<MTLComputePipelineState> pipeline = pipeline_from_source(source, @"mp_tg");
        if (pipeline == nil) {
            return 0.0;
        }
        id<MTLBuffer> out = [device newBufferWithLength:core_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> params = [device newBufferWithBytes:&iterations length:sizeof(iterations) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:out offset:0 atIndex:0];
        [encoder setBuffer:params offset:0 atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(core_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        const double bytes = static_cast<double>(core_count) * 256.0 * static_cast<double>(iterations) * 2.0 * sizeof(uint32_t);
        return bytes / seconds / 1.0e9;
    }
}

extern "C" int mp_metal_simdgroup_matrix_probe() {
    @autoreleasepool {
        NSString* source =
             @"#include <metal_stdlib>\n"
             "#include <metal_simdgroup_matrix>\n"
             "using namespace metal;\n"
             "kernel void mp_probe(device half* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {\n"
             "  simdgroup_half8x8 a;\n"
             "  simdgroup_half8x8 b;\n"
             "  simdgroup_float8x8 c;\n"
             "  simdgroup_multiply_accumulate(c, a, b, c);\n"
             "  out[tid] = half(1.0);\n"
             "}\n";
        id<MTLComputePipelineState> pipeline = pipeline_from_source(source, @"mp_probe");
        return pipeline == nil ? 1 : 0;
    }
}

extern "C" int mp_metal_mpp_tensor_matmul_probe_detail(
    MpMetalMppProbeInfo* info_out,
    char* status_out,
    size_t status_len
) {
#if SUPERSONIC_HAVE_MTL4_MPP
    @autoreleasepool {
        if (info_out == nullptr) {
            copy_probe_status("invalid-output", status_out, status_len);
            return 99;
        }
        info_out->status = 0;
        info_out->tensor_write_value = 0.0f;
        info_out->matmul_value = 0.0f;

        id<MTLDevice> device = metal_device();
        if (device == nil) {
            info_out->status = 2;
            copy_probe_status("no-device", status_out, status_len);
            return info_out->status;
        }

        NSString* tensor_write_source =
             @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void mp_tensor_write(tensor<device float, dextents<int32_t, 2>> C [[buffer(0)]]) {\n"
             "  C[0, 0] = 123.0f;\n"
             "}\n";
        id<MTLComputePipelineState> tensor_write_pipeline =
            mtl4_pipeline_from_source(tensor_write_source, @"mp_tensor_write", 32);
        if (tensor_write_pipeline == nil) {
            info_out->status = 10;
            copy_probe_status("tensor-write-pipeline", status_out, status_len);
            return info_out->status;
        }
        id<MTL4CommandQueue> tensor_write_queue = [device newMTL4CommandQueue];
        id<MTLSharedEvent> tensor_write_event = [device newSharedEvent];
        id<MTL4ArgumentTable> tensor_write_args = mtl4_argument_table(device, 1);
        id<MTLTensor> tensor_write_tensor =
            tensor_from_device(device, device_tensor_descriptor(1, 1, MTLTensorDataTypeFloat32));
        if (tensor_write_queue == nil || tensor_write_event == nil ||
            tensor_write_args == nil || tensor_write_tensor == nil) {
            info_out->status = 11;
            copy_probe_status("tensor-write-setup", status_out, status_len);
            return info_out->status;
        }
        [tensor_write_args setResource:tensor_write_tensor.gpuResourceID atBufferIndex:0];
        if (!encode_mpp_gemm_mtl4(
                device, tensor_write_queue, tensor_write_pipeline, tensor_write_args,
                1, 1, 32, tensor_write_event, 1)) {
            info_out->status = 12;
            copy_probe_status("tensor-write-dispatch", status_out, status_len);
            return info_out->status;
        }
        info_out->tensor_write_value = tensor_first_f32(tensor_write_tensor);
        if (info_out->tensor_write_value != 123.0f) {
            info_out->status = 13;
            copy_probe_status("tensor-write-readback", status_out, status_len);
            return info_out->status;
        }

        NSString* source =
             @"#include <metal_stdlib>\n"
             "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
             "using namespace metal;\n"
             "using namespace mpp::tensor_ops;\n"
             "kernel void mp_mpp_probe(tensor<device half, dextents<int32_t, 2>> A [[buffer(0)]],\n"
             "                         tensor<device half, dextents<int32_t, 2>> B [[buffer(1)]],\n"
             "                         tensor<device float, dextents<int32_t, 2>> C [[buffer(2)]],\n"
             "                         uint2 tgid [[threadgroup_position_in_grid]]) {\n"
             "  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);\n"
             "  matmul2d<desc, execution_simdgroups<4>> op;\n"
             "  auto tA = A.slice(0, tgid.y * 64);\n"
             "  auto tB = B.slice(tgid.x * 32, 0);\n"
             "  auto tC = C.slice(tgid.x * 32, tgid.y * 64);\n"
             "  op.run(tA, tB, tC);\n"
             "}\n";
        id<MTLComputePipelineState> pipeline = mtl4_pipeline_from_source(source, @"mp_mpp_probe", 128);
        if (pipeline == nil) {
            info_out->status = 20;
            copy_probe_status("mpp-pipeline", status_out, status_len);
            return info_out->status;
        }
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        id<MTLSharedEvent> event = [device newSharedEvent];
        id<MTL4ArgumentTable> args = mtl4_argument_table(device, 3);
        if (queue == nil || event == nil || args == nil) {
            info_out->status = 21;
            copy_probe_status("mpp-command-setup", status_out, status_len);
            return info_out->status;
        }
        const NSUInteger a_bytes = 64u * 64u * sizeof(uint16_t);
        const NSUInteger b_bytes = 32u * 64u * sizeof(uint16_t);
        id<MTLBuffer> a_buf = [device newBufferWithLength:a_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [device newBufferWithLength:b_bytes options:MTLResourceStorageModeShared];
        if (a_buf == nil || b_buf == nil) {
            info_out->status = 22;
            copy_probe_status("mpp-input-buffer", status_out, status_len);
            return info_out->status;
        }
        auto* a = static_cast<uint16_t*>(a_buf.contents);
        auto* b = static_cast<uint16_t*>(b_buf.contents);
        for (NSUInteger i = 0; i < 64u * 64u; ++i) {
            a[i] = 0x3c00u;
        }
        for (NSUInteger i = 0; i < 32u * 64u; ++i) {
            b[i] = 0x3800u;
        }
        id<MTLTensor> a_tensor = tensor_from_device(device, device_tensor_descriptor(64, 64, MTLTensorDataTypeFloat16));
        id<MTLTensor> b_tensor = tensor_from_device(device, device_tensor_descriptor(32, 64, MTLTensorDataTypeFloat16));
        id<MTLTensor> c_tensor = tensor_from_device(device, device_tensor_descriptor(32, 64, MTLTensorDataTypeFloat32));
        if (a_tensor == nil || b_tensor == nil || c_tensor == nil) {
            info_out->status = 23;
            copy_probe_status("mpp-tensor-create", status_out, status_len);
            return info_out->status;
        }
        tensor_replace_all_f16(a_tensor, a, 64, 64);
        tensor_replace_all_f16(b_tensor, b, 32, 64);
        [args setResource:a_tensor.gpuResourceID atBufferIndex:0];
        [args setResource:b_tensor.gpuResourceID atBufferIndex:1];
        [args setResource:c_tensor.gpuResourceID atBufferIndex:2];
        const NSUInteger threads_per_threadgroup = pipeline.threadExecutionWidth * 4u;
        const bool ok = encode_mpp_gemm_mtl4(
            device, queue, pipeline, args, 1, 1, threads_per_threadgroup, event, 1
        );
        if (!ok) {
            info_out->status = 24;
            copy_probe_status("mpp-dispatch", status_out, status_len);
            return info_out->status;
        }
        info_out->matmul_value = tensor_first_f32(c_tensor);
        if (info_out->matmul_value == 0.0f || !std::isfinite(static_cast<double>(info_out->matmul_value))) {
            info_out->status = 25;
            copy_probe_status("mpp-readback", status_out, status_len);
            return info_out->status;
        }
        copy_probe_status("ok", status_out, status_len);
        return 0;
    }
#else
    if (info_out != nullptr) {
        info_out->status = 1;
        info_out->tensor_write_value = 0.0f;
        info_out->matmul_value = 0.0f;
    }
    copy_probe_status("sdk-unavailable", status_out, status_len);
    return 1;
#endif
}

extern "C" int mp_metal_mpp_tensor_matmul_probe() {
    MpMetalMppProbeInfo info;
    char status[64];
    return mp_metal_mpp_tensor_matmul_probe_detail(&info, status, sizeof(status));
}

extern "C" double mp_metal_mpp_tensor_gemm_f16_tflops(uint32_t size, uint32_t iterations) {
#if SUPERSONIC_HAVE_MTL4_MPP
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        if (device == nil || queue == nil || size == 0 || iterations == 0 || (size % 64u) != 0u) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM setup rejected: size=%u iterations=%u", size, iterations);
            }
            return 0.0;
        }
        NSString* source =
             @"#include <metal_stdlib>\n"
             "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
             "using namespace metal;\n"
             "using namespace mpp::tensor_ops;\n"
             "kernel void mp_mpp_gemm_tile(tensor<device half, dextents<int32_t, 2>> A [[buffer(0)]],\n"
             "                             tensor<device half, dextents<int32_t, 2>> B [[buffer(1)]],\n"
             "                             tensor<device float, dextents<int32_t, 2>> C [[buffer(2)]]) {\n"
             "  constexpr auto desc = matmul2d_descriptor(64, 32, 64, false, false, false);\n"
             "  matmul2d<desc, execution_simdgroups<4>> op;\n"
             "  auto tA = A.slice(0, 0);\n"
             "  auto tB = B.slice(0, 0);\n"
             "  auto tC = C.slice(0, 0);\n"
             "  op.run(tA, tB, tC);\n"
             "}\n";
        id<MTLComputePipelineState> pipeline = mtl4_pipeline_from_source(source, @"mp_mpp_gemm_tile", 128);
        if (pipeline == nil) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM pipeline failed");
            }
            return 0.0;
        }

        const NSUInteger a_count = 64u * 64u;
        const NSUInteger b_count = 32u * 64u;
        id<MTLBuffer> a_buf = [device newBufferWithLength:a_count * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [device newBufferWithLength:b_count * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        if (a_buf == nil || b_buf == nil) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM input buffers failed");
            }
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
        id<MTLTensor> a_tensor = tensor_from_device(device, device_tensor_descriptor(64, 64, MTLTensorDataTypeFloat16));
        id<MTLTensor> b_tensor = tensor_from_device(device, device_tensor_descriptor(32, 64, MTLTensorDataTypeFloat16));
        id<MTLTensor> c_tensor = tensor_from_device(device, device_tensor_descriptor(32, 64, MTLTensorDataTypeFloat32));
        if (a_tensor == nil || b_tensor == nil || c_tensor == nil) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM tensor creation failed");
            }
            return 0.0;
        }
        tensor_replace_all_f16(a_tensor, a, 64, 64);
        tensor_replace_all_f16(b_tensor, b, 32, 64);

        id<MTL4ArgumentTable> args = mtl4_argument_table(device, 3);
        id<MTLSharedEvent> event = [device newSharedEvent];
        if (args == nil || event == nil) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM argument table/event failed");
            }
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

        if (!encode_mpp_gemm_mtl4(
                device, queue, pipeline, args, tg_x, tg_y, threads_per_threadgroup, event, 1)) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM warmup dispatch failed");
            }
            return 0.0;
        }

        auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < iterations; ++i) {
            const uint64_t signal_value = static_cast<uint64_t>(i) + 2u;
            if (!encode_mpp_gemm_mtl4(
                    device, queue, pipeline, args, tg_x, tg_y, threads_per_threadgroup, event, signal_value)) {
                if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                    NSLog(@"machine-profile MPP GEMM timed dispatch failed at iteration %u", i);
                }
                return 0.0;
            }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        volatile float guard = tensor_first_f32(c_tensor);
        if (guard == 0.0f || !std::isfinite(static_cast<double>(guard))) {
            if (NSProcessInfo.processInfo.environment[@"SUPERSONIC_METAL_PROFILE_DEBUG"] != nil) {
                NSLog(@"machine-profile MPP GEMM guard failed: %f", static_cast<double>(guard));
            }
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

static double run_simdgroup_mma_f16(
    uint32_t core_count,
    uint32_t threadgroups_per_core,
    uint32_t simdgroups_per_threadgroup,
    uint32_t accumulators,
    uint32_t iterations,
    bool f16_accum
) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || core_count == 0 || threadgroups_per_core == 0 ||
            simdgroups_per_threadgroup == 0 || accumulators == 0 || iterations == 0) {
            return 0.0;
        }
        const uint32_t threads_per_threadgroup = simdgroups_per_threadgroup * 32u;
        if (threads_per_threadgroup > device.maxThreadsPerThreadgroup.width) {
            return 0.0;
        }
        accumulators = std::min<uint32_t>(accumulators, 32u);

        NSString* acc_type = f16_accum ? @"half" : @"float";
        NSString* matrix_type = f16_accum ? @"simdgroup_half8x8" : @"simdgroup_float8x8";
        NSString* out_type = f16_accum ? @"half" : @"float";
        NSString* zero = f16_accum ? @"0.0h" : @"0.0f";
        NSMutableString* source = [NSMutableString string];
        [source appendString:
            @"#include <metal_stdlib>\n"
             "#include <metal_simdgroup_matrix>\n"
             "using namespace metal;\n"
             "struct Params { uint iterations; uint tile_count; };\n"];
        [source appendFormat:
            @"kernel void mp_mma_sweep(device const half* a_buf [[buffer(0)]],\n"
             "                         device const half* b_buf [[buffer(1)]],\n"
             "                         device %@* out [[buffer(2)]],\n"
             "                         constant Params& p [[buffer(3)]],\n"
             "                         uint tg [[threadgroup_position_in_grid]],\n"
             "                         uint sg [[simdgroup_index_in_threadgroup]],\n"
             "                         uint simdgroups_per_tg [[simdgroups_per_threadgroup]]) {\n"
             "  const uint logical = tg * simdgroups_per_tg + sg;\n"
             "  const uint tile = logical %% p.tile_count;\n"
             "  simdgroup_half8x8 a;\n"
             "  simdgroup_half8x8 b;\n", out_type];
        for (uint32_t i = 0; i < accumulators; ++i) {
            [source appendFormat:@"  %@ c%u = make_filled_simdgroup_matrix<%@, 8>(%@);\n",
                                 matrix_type, i, acc_type, zero];
        }
        [source appendString:
             @"  simdgroup_load(a, a_buf + tile * 64, 8);\n"
             "  simdgroup_load(b, b_buf + tile * 64, 8);\n"
             "  for (uint i = 0; i < p.iterations; ++i) {\n"];
        for (uint32_t i = 0; i < accumulators; ++i) {
            [source appendFormat:@"    simdgroup_multiply_accumulate(c%u, a, b, c%u);\n", i, i];
        }
        [source appendString:@"  }\n"];
        [source appendFormat:@"  const uint base = logical * %u;\n", accumulators * 64u];
        for (uint32_t i = 0; i < accumulators; ++i) {
            [source appendFormat:@"  simdgroup_store(c%u, out + base + %u, 8);\n", i, i * 64u];
        }
        [source appendString:@"}\n"];

        id<MTLComputePipelineState> pipeline = pipeline_from_source(source, @"mp_mma_sweep");
        if (pipeline == nil) {
            return 0.0;
        }
        const uint32_t threadgroups = core_count * threadgroups_per_core;
        const uint32_t logical_simdgroups = threadgroups * simdgroups_per_threadgroup;
        const uint32_t tile_count = 1024;
        const uint64_t tile_elems = static_cast<uint64_t>(tile_count) * 64u;
        const uint64_t out_elems = static_cast<uint64_t>(logical_simdgroups) * static_cast<uint64_t>(accumulators) * 64u;
        const size_t out_elem_size = f16_accum ? sizeof(uint16_t) : sizeof(float);
        id<MTLBuffer> a = [device newBufferWithLength:tile_elems * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [device newBufferWithLength:tile_elems * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:out_elems * out_elem_size options:MTLResourceStorageModeShared];
        if (a == nil || b == nil || out == nil) {
            return 0.0;
        }
        auto* a_words = static_cast<uint16_t*>(a.contents);
        auto* b_words = static_cast<uint16_t*>(b.contents);
        for (uint64_t i = 0; i < tile_elems; ++i) {
            a_words[i] = static_cast<uint16_t>(0x3c00u + (i & 3u));
            b_words[i] = static_cast<uint16_t>(0x3800u + (i & 1u));
        }
        memset(out.contents, 0, out_elems * out_elem_size);
        struct Params {
            uint32_t iterations;
            uint32_t tile_count;
        } params = { iterations, tile_count };
        id<MTLBuffer> params_buf = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBuffer:params_buf offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
        [encoder endEncoding];
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        volatile uint16_t guard = static_cast<uint16_t*>(out.contents)[0];
        (void)guard;
        const double flops = static_cast<double>(logical_simdgroups)
            * static_cast<double>(iterations)
            * static_cast<double>(accumulators)
            * 2.0
            * 8.0
            * 8.0
            * 8.0;
        return flops / seconds / 1.0e12;
    }
}

extern "C" double mp_metal_simdgroup_mma_f16_sweep_tflops(
    uint32_t core_count,
    uint32_t threadgroups_per_core,
    uint32_t simdgroups_per_threadgroup,
    uint32_t accumulators,
    uint32_t iterations,
    uint32_t f16_accum
) {
    return run_simdgroup_mma_f16(
        core_count,
        threadgroups_per_core,
        simdgroups_per_threadgroup,
        accumulators,
        iterations,
        f16_accum != 0
    );
}

extern "C" double mp_metal_simdgroup_mma_f16_tflops(
    uint32_t core_count,
    uint32_t threadgroups_per_core,
    uint32_t iterations
) {
    return run_simdgroup_mma_f16(core_count, threadgroups_per_core, 32, 16, iterations, false);
}

extern "C" double mp_metal_simdgroup_mma_f16_accum_f16_tflops(
    uint32_t core_count,
    uint32_t threadgroups_per_core,
    uint32_t iterations
) {
    return run_simdgroup_mma_f16(core_count, threadgroups_per_core, 32, 8, iterations, true);
}

extern "C" double mp_metal_simdgroup_gemm_f16_tflops(
    uint32_t size,
    uint32_t iterations,
    uint32_t simdgroups_per_threadgroup
) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || size == 0 || iterations == 0 || simdgroups_per_threadgroup == 0) {
            return 0.0;
        }
        if ((size % 8u) != 0u) {
            return 0.0;
        }
        const uint32_t threads_per_threadgroup = simdgroups_per_threadgroup * 32u;
        if (threads_per_threadgroup > device.maxThreadsPerThreadgroup.width) {
            return 0.0;
        }
        NSString* source =
            @"#include <metal_stdlib>\n"
             "#include <metal_simdgroup_matrix>\n"
             "using namespace metal;\n"
             "struct Params { uint n; uint tiles_n; uint tiles_total; };\n"
             "kernel void mp_sg_gemm(device const half* a_buf [[buffer(0)]],\n"
             "                       device const half* b_buf [[buffer(1)]],\n"
             "                       device float* c_buf [[buffer(2)]],\n"
             "                       constant Params& p [[buffer(3)]],\n"
             "                       uint tg [[threadgroup_position_in_grid]],\n"
             "                       uint sg [[simdgroup_index_in_threadgroup]],\n"
             "                       uint simdgroups_per_tg [[simdgroups_per_threadgroup]]) {\n"
             "  const uint logical = tg * simdgroups_per_tg + sg;\n"
             "  if (logical >= p.tiles_total) return;\n"
             "  const uint tile_m = logical / p.tiles_n;\n"
             "  const uint tile_n = logical - tile_m * p.tiles_n;\n"
             "  simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
             "  for (uint tile_k = 0; tile_k < p.tiles_n; ++tile_k) {\n"
             "    simdgroup_half8x8 a;\n"
             "    simdgroup_half8x8 b;\n"
             "    simdgroup_load(a, a_buf + (tile_m * 8) * p.n + tile_k * 8, p.n);\n"
             "    simdgroup_load(b, b_buf + (tile_k * 8) * p.n + tile_n * 8, p.n);\n"
             "    simdgroup_multiply_accumulate(c, a, b, c);\n"
             "  }\n"
             "  simdgroup_store(c, c_buf + (tile_m * 8) * p.n + tile_n * 8, p.n);\n"
             "}\n";
        id<MTLComputePipelineState> pipeline = pipeline_from_source(source, @"mp_sg_gemm");
        if (pipeline == nil) {
            return 0.0;
        }
        const NSUInteger n = static_cast<NSUInteger>(size);
        const NSUInteger half_bytes = n * n * sizeof(uint16_t);
        const NSUInteger out_bytes = n * n * sizeof(float);
        id<MTLBuffer> a_buf = [device newBufferWithLength:half_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [device newBufferWithLength:half_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> c_buf = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (a_buf == nil || b_buf == nil || c_buf == nil) {
            return 0.0;
        }
        auto* a = static_cast<uint16_t*>(a_buf.contents);
        auto* b = static_cast<uint16_t*>(b_buf.contents);
        for (NSUInteger i = 0; i < n * n; ++i) {
            a[i] = static_cast<uint16_t>(0x3c00u + (i & 1u));
            b[i] = static_cast<uint16_t>(0x3800u + (i & 1u));
        }
        memset(c_buf.contents, 0, out_bytes);
        struct Params {
            uint32_t n;
            uint32_t tiles_n;
            uint32_t tiles_total;
        } params = { size, size / 8u, (size / 8u) * (size / 8u) };
        id<MTLBuffer> params_buf = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        const uint32_t threadgroups = (params.tiles_total + simdgroups_per_threadgroup - 1u) / simdgroups_per_threadgroup;

        {
            id<MTLCommandBuffer> warm = [queue commandBuffer];
            id<MTLComputeCommandEncoder> warm_encoder = [warm computeCommandEncoder];
            [warm_encoder setComputePipelineState:pipeline];
            [warm_encoder setBuffer:a_buf offset:0 atIndex:0];
            [warm_encoder setBuffer:b_buf offset:0 atIndex:1];
            [warm_encoder setBuffer:c_buf offset:0 atIndex:2];
            [warm_encoder setBuffer:params_buf offset:0 atIndex:3];
            [warm_encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
            [warm_encoder endEncoding];
            [warm commit];
            [warm waitUntilCompleted];
            if (warm.status != MTLCommandBufferStatusCompleted) {
                return 0.0;
            }
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        for (uint32_t i = 0; i < iterations; ++i) {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:a_buf offset:0 atIndex:0];
            [encoder setBuffer:b_buf offset:0 atIndex:1];
            [encoder setBuffer:c_buf offset:0 atIndex:2];
            [encoder setBuffer:params_buf offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
            [encoder endEncoding];
        }
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        volatile float guard = static_cast<float*>(c_buf.contents)[0];
        (void)guard;
        const double flops = static_cast<double>(iterations)
            * 2.0
            * static_cast<double>(size)
            * static_cast<double>(size)
            * static_cast<double>(size);
        return flops / seconds / 1.0e12;
    }
}

extern "C" double mp_metal_mps_gemm_f16_tflops(uint32_t size, uint32_t iterations) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || size == 0 || iterations == 0) {
            return 0.0;
        }
        const NSUInteger n = static_cast<NSUInteger>(size);
        const NSUInteger row_bytes = n * sizeof(uint16_t);
        const NSUInteger bytes = row_bytes * n;
        id<MTLBuffer> a_buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> c_buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (a_buf == nil || b_buf == nil || c_buf == nil) {
            return 0.0;
        }
        auto* a = static_cast<uint16_t*>(a_buf.contents);
        auto* b = static_cast<uint16_t*>(b_buf.contents);
        for (NSUInteger i = 0; i < static_cast<NSUInteger>(size) * static_cast<NSUInteger>(size); ++i) {
            a[i] = static_cast<uint16_t>(0x3c00u + (i & 1u));
            b[i] = static_cast<uint16_t>(0x3800u + (i & 1u));
        }
        memset(c_buf.contents, 0, bytes);

        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                           columns:n
                                                                          rowBytes:row_bytes
                                                                          dataType:MPSDataTypeFloat16];
        MPSMatrix* a_mat = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:desc];
        MPSMatrix* b_mat = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:desc];
        MPSMatrix* c_mat = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:desc];
        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                          transposeLeft:false
                                                                         transposeRight:false
                                                                            resultRows:n
                                                                         resultColumns:n
                                                                       interiorColumns:n
                                                                                alpha:1.0
                                                                                 beta:0.0];
        if (a_mat == nil || b_mat == nil || c_mat == nil || gemm == nil) {
            return 0.0;
        }

        // Warm the MPS pipeline and allocation path before the measured pass.
        {
            id<MTLCommandBuffer> warm = [queue commandBuffer];
            [gemm encodeToCommandBuffer:warm leftMatrix:a_mat rightMatrix:b_mat resultMatrix:c_mat];
            [warm commit];
            [warm waitUntilCompleted];
            if (warm.status != MTLCommandBufferStatusCompleted) {
                return 0.0;
            }
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        for (uint32_t i = 0; i < iterations; ++i) {
            [gemm encodeToCommandBuffer:command_buffer leftMatrix:a_mat rightMatrix:b_mat resultMatrix:c_mat];
        }
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        volatile uint16_t guard = static_cast<uint16_t*>(c_buf.contents)[0];
        (void)guard;
        const double flops = static_cast<double>(iterations)
            * 2.0
            * static_cast<double>(size)
            * static_cast<double>(size)
            * static_cast<double>(size);
        return flops / seconds / 1.0e12;
    }
}

extern "C" double mp_metal_int4_gemv_gb_s(uint32_t in_dim, uint32_t out_dim, uint32_t group_size, uint32_t iterations) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device();
        id<MTLCommandQueue> queue = metal_queue();
        if (device == nil || queue == nil || in_dim == 0 || out_dim == 0 || group_size == 0 || iterations == 0) {
            return 0.0;
        }
        NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "struct Params { uint in_dim; uint out_dim; uint group_size; uint iterations; };\n"
             "kernel void mp_int4(device const half* lhs [[buffer(0)]],\n"
             "                    device const uchar* rhs [[buffer(1)]],\n"
             "                    device const half* scales [[buffer(2)]],\n"
             "                    device const half* zeros [[buffer(3)]],\n"
             "                    device half* out [[buffer(4)]],\n"
             "                    constant Params& p [[buffer(5)]],\n"
             "                    uint row [[thread_position_in_grid]]) {\n"
             "  if (row >= p.out_dim) return;\n"
             "  const uint groups = (p.in_dim + p.group_size - 1) / p.group_size;\n"
             "  float acc = 0.0f;\n"
             "  for (uint rep = 0; rep < p.iterations; ++rep) {\n"
             "    for (uint k = 0; k < p.in_dim; ++k) {\n"
             "      uint packed = rhs[row * ((p.in_dim + 1) / 2) + (k >> 1)];\n"
             "      uint nib = (k & 1) ? (packed >> 4) : (packed & 15);\n"
             "      uint g = k / p.group_size;\n"
             "      float w = (float(nib) - float(zeros[row * groups + g])) * float(scales[row * groups + g]);\n"
             "      acc += float(lhs[k]) * w;\n"
             "    }\n"
             "  }\n"
             "  out[row] = half(acc);\n"
             "}\n";
        id<MTLComputePipelineState> pipeline = pipeline_from_source(source, @"mp_int4");
        if (pipeline == nil) {
            return 0.0;
        }
        const uint64_t rhs_bytes = static_cast<uint64_t>(out_dim) * ((in_dim + 1) / 2);
        const uint64_t groups = (in_dim + group_size - 1) / group_size;
        id<MTLBuffer> lhs = [device newBufferWithLength:in_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> rhs = [device newBufferWithLength:rhs_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales = [device newBufferWithLength:out_dim * groups * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> zeros = [device newBufferWithLength:out_dim * groups * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:out_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        if (lhs == nil || rhs == nil || scales == nil || zeros == nil || out == nil) {
            return 0.0;
        }
        memset(lhs.contents, 0x3c, in_dim * sizeof(uint16_t));
        memset(rhs.contents, 0x11, rhs_bytes);
        memset(scales.contents, 0x3c, out_dim * groups * sizeof(uint16_t));
        memset(zeros.contents, 0, out_dim * groups * sizeof(uint16_t));
        struct Params { uint32_t in_dim; uint32_t out_dim; uint32_t group_size; uint32_t iterations; } params = {
            in_dim, out_dim, group_size, iterations
        };
        id<MTLBuffer> params_buf = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lhs offset:0 atIndex:0];
        [encoder setBuffer:rhs offset:0 atIndex:1];
        [encoder setBuffer:scales offset:0 atIndex:2];
        [encoder setBuffer:zeros offset:0 atIndex:3];
        [encoder setBuffer:out offset:0 atIndex:4];
        [encoder setBuffer:params_buf offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        auto start = std::chrono::steady_clock::now();
        [command_buffer commit];
        const double seconds = command_elapsed_seconds(command_buffer, start);
        if (seconds <= 0.0 || !std::isfinite(seconds)) {
            return 0.0;
        }
        const double bytes = static_cast<double>(iterations) *
            (static_cast<double>(rhs_bytes) + static_cast<double>(out_dim * groups * sizeof(uint16_t) * 2) +
             static_cast<double>(out_dim) * static_cast<double>(in_dim) * sizeof(uint16_t));
        return bytes / seconds / 1.0e9;
    }
}
