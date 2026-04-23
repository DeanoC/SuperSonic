pub mod dflash;
mod ffi;
pub mod gemma4;
mod layer_desc;
mod metal_host;
mod metal_native;
pub mod phi4;
pub mod prefill_ffi;

pub use ffi::{
    cuda_argmax_bf16, cuda_lm_head_argmax_bf16, matmul_rhs_transposed_4b, metal_argmax_bf16_into,
    metal_lm_head_argmax_bf16, metal_lm_head_argmax_bf16_into, persistent_decode,
    persistent_decode_4b,
    persistent_decode_4b_qwen35_sm86_specialized, persistent_decode_qwen08_sm86_specialized,
    query_gpu_info, query_hip_device_clock_khz, qwen_rms_norm_standalone_matvec_host_f32, rms_norm,
    rms_norm_4b, rms_norm_4b_multirow, set_qwen35_4b_launch_preset, standalone_matvec,
    standalone_matvec_4b, standalone_matvec_host_f32,
};
pub use layer_desc::{
    BatchSeqDesc, DecodeLayerDesc, FP8ScaleDesc, INT4ScaleDesc, KVCacheFp8Desc, MAX_BATCH_SIZE,
};
