mod ffi;
mod layer_desc;
pub mod prefill_ffi;

pub use ffi::{persistent_decode, persistent_decode_4b, query_gpu_info, rms_norm, rms_norm_4b, standalone_matvec, standalone_matvec_4b};
pub use layer_desc::DecodeLayerDesc;
