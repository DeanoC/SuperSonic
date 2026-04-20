mod ffi;
mod layer_desc;
pub mod prefill_ffi;
pub mod gemma4;

pub use ffi::{
    cuda_argmax_bf16, cuda_lm_head_argmax_bf16, matmul_rhs_transposed_4b,
    persistent_decode, persistent_decode_4b, persistent_decode_qwen08_sm86_specialized, query_gpu_info, rms_norm, rms_norm_4b,
    rms_norm_4b_multirow, standalone_matvec, standalone_matvec_4b,
};
pub use layer_desc::{DecodeLayerDesc, FP8ScaleDesc, INT4ScaleDesc, KVCacheFp8Desc, BatchSeqDesc, MAX_BATCH_SIZE};
