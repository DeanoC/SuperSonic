// Metal decode megakernel bridge (incremental port scaffold).
//
// HIP production decode uses supersonic_qwen35_4b_hip_persistent_decode in
// full_attention_4b.hip (64 fused layers, grid barriers, GQH/INT4/FP8 paths).
// Metal currently routes greedy decode through the component prefill-op replay
// in runtime/prefill_engine.rs (mtp_decode_step_greedy).
//
// Port order:
// 1. Batched command-buffer dispatch + GPU blit infrastructure (metal_bridge.mm)
// 2. Fused tail: final RMSNorm + lm_head + argmax (single-token greedy)
// 3. Per-layer fusion using existing prefill.metal kernels
// 4. Full persistent decode kernel with Metal-specific sync (threadgroup barriers)

extern "C" int supersonic_qwen35_hip_persistent_decode(...) {
    return 260;
}

extern "C" int supersonic_qwen35_4b_hip_persistent_decode(...) {
    return 401;
}
