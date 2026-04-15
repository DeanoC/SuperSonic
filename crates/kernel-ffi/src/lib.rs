mod ffi;
mod layer_desc;

pub use ffi::{persistent_decode, query_gpu_info, rms_norm, standalone_matvec};
pub use layer_desc::DecodeLayerDesc;
