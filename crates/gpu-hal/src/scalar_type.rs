use std::ffi::c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    BF16,
    F32,
    U8,
    U32,
    I64,
}

impl ScalarType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::U32 => 4,
            Self::U8 => 1,
            Self::I64 => 8,
        }
    }

    /// Dtype code for the persistent decode kernel (0=F16, 1=F32, 2=BF16).
    pub fn kernel_dtype_code(self) -> c_int {
        match self {
            Self::F16 => 0,
            Self::F32 => 1,
            Self::BF16 => 2,
            _ => panic!("unsupported kernel dtype: {self:?}"),
        }
    }

    /// Convert from a string name (used by baked manifest).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" => Some(Self::F32),
            "u8" => Some(Self::U8),
            "u32" => Some(Self::U32),
            "i64" => Some(Self::I64),
            _ => None,
        }
    }

    /// Convert from safetensors Dtype.
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Option<Self> {
        match dtype {
            safetensors::Dtype::F16 => Some(Self::F16),
            safetensors::Dtype::BF16 => Some(Self::BF16),
            safetensors::Dtype::F32 => Some(Self::F32),
            safetensors::Dtype::U8 => Some(Self::U8),
            safetensors::Dtype::U32 => Some(Self::U32),
            safetensors::Dtype::I64 => Some(Self::I64),
            _ => None,
        }
    }
}
