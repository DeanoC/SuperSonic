mod ffi;
mod layer_desc;

pub use ffi::{persistent_decode, rms_norm, standalone_matvec};
pub use layer_desc::DecodeLayerDesc;
