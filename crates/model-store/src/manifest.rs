use serde::{Deserialize, Serialize};

pub const FORMAT_VERSION: u32 = 1;
pub const CONVERTER_VERSION: u32 = 2;

/// Describes the layout transformation applied to a tensor at bake time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutTag {
    /// Verbatim copy from safetensors.
    Raw,
    /// Conv1d weight with shape [C_out, 1, K] squeezed to [C_out, K].
    DepthwiseConvSqueezed,
    /// Bias reshaped from [H] to [1, 1, H].
    HeadBiasReshaped,
    /// A_log: F32 exp() converted to BF16, reshaped from [H] to [1, 1, H].
    HeadExpReshaped,
    /// FP8 E4M3 weight dequantized to BF16 using block-wise scale_inv at bake time.
    Fp8Dequantized,
}

/// Metadata for a single tensor in the baked package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    /// Dtype as string: "bf16", "f32", "f16", "u8", "u32", "i64".
    pub dtype: String,
    pub layout: LayoutTag,
    /// Byte offset in weights.bin (4096-aligned).
    pub offset: u64,
    /// Byte length of tensor data.
    pub byte_len: u64,
}

/// Top-level manifest for a baked package.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub converter_version: u32,
    pub model_family: String,
    pub tensors: Vec<TensorMeta>,
}
