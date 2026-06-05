use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub const FORMAT_VERSION: u32 = 2;
pub const CONVERTER_VERSION: u32 = 1;
// Phase 6.1 (this PR) extends the baker to capture `mtp.*` tensors for
// Qwen3.6-MoE self-speculative decode. The change is purely additive —
// existing bakes don't have MTP tensors but the production decode path
// doesn't read them, so the bake is still valid. Bump CONVERTER_VERSION
// when the runtime starts CONSUMING the MTP weights (Phase 6.2+).

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
    /// FP8 E4M3 weight stored natively (not dequantized). Companion _scale_inv tensor
    /// is stored separately. Used for runtime FP8 dequant on GPU.
    Fp8Native,
    /// INT8 quantized weight. Stored as raw two's-complement bytes with a
    /// companion per-row `.SCB` scale tensor in F32, matching the
    /// BitsAndBytes `Linear8bitLt` state_dict representation.
    Int8Quantized,
    /// INT4 quantized weight. Packed as 2 nibbles per byte.
    /// Companion _int4_scale and _int4_zero tensors stored separately.
    Int4Quantized,
    /// HIGGS 4-bit grid-coded weights. Runtime-backed profile, not compatible
    /// with native INT4 scale/zero sidecars.
    HiggsGridQuantized,
    /// QuIP# E8P codebook blocks.
    QuipE8Quantized,
    /// QTIP trellis-coded blocks.
    QtipTrellisQuantized,
    /// Verbatim GGML K-block quantized tensors from GGUF. Shape is
    /// `[rows, row_bytes]`, dtype is `u8`, and logical input columns are
    /// supplied by the model descriptor at runtime.
    GgmlQ4K,
    GgmlQ5K,
    GgmlQ6K,
}

/// User-facing weight quantization profile. These names are stable CLI,
/// manifest, bake-directory, and release-asset identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantProfile {
    Bf16,
    Fp8Native,
    Int4Gptq,
    Int4Awq,
    Int4Autoround,
    Int4Hqq,
    Higgs4,
    QuipE8,
    QtipTrellis2,
    Q4Km,
    Q4KmGptq,
}

impl QuantProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8Native => "fp8-native",
            Self::Int4Gptq => "int4-gptq",
            Self::Int4Awq => "int4-awq",
            Self::Int4Autoround => "int4-autoround",
            Self::Int4Hqq => "int4-hqq",
            Self::Higgs4 => "higgs4",
            Self::QuipE8 => "quip-e8",
            Self::QtipTrellis2 => "qtip-trellis2",
            Self::Q4Km => "q4km",
            Self::Q4KmGptq => "q4km-gptq",
        }
    }

    pub fn is_native_int4_runtime(self) -> bool {
        matches!(
            self,
            Self::Int4Gptq | Self::Int4Awq | Self::Int4Autoround | Self::Int4Hqq | Self::Q4KmGptq
        )
    }

    pub fn is_runtime_backed_lowbit(self) -> bool {
        matches!(self, Self::Higgs4 | Self::QuipE8 | Self::QtipTrellis2)
    }
}

impl std::fmt::Display for QuantProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for QuantProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8-native" | "fp8" => Ok(Self::Fp8Native),
            "int4-gptq" | "gptq" => Ok(Self::Int4Gptq),
            "int4-awq" | "awq" => Ok(Self::Int4Awq),
            "int4-autoround" | "autoround" | "signround" | "int4-signround" => {
                Ok(Self::Int4Autoround)
            }
            "int4-hqq" | "hqq" => Ok(Self::Int4Hqq),
            "higgs4" | "higgs-4" => Ok(Self::Higgs4),
            "quip-e8" | "quipe8" => Ok(Self::QuipE8),
            "qtip-trellis2" | "qtip" => Ok(Self::QtipTrellis2),
            "q4km" => Ok(Self::Q4Km),
            "q4km-gptq" => Ok(Self::Q4KmGptq),
            other => Err(format!(
                "unknown quant profile {other:?}; expected one of: bf16, fp8-native, int4-gptq, int4-awq, int4-autoround, int4-hqq, higgs4, quip-e8, qtip-trellis2"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantMethodMeta {
    pub profile: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_bits_per_weight: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_weight_bits: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_precision_assignment: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_corpus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_samples: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_seqlen: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_version: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_method: Option<QuantMethodMeta>,
    pub tensors: Vec<TensorMeta>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_profile_parse_aliases_and_canonical_names() {
        assert_eq!(
            "int4-gptq".parse::<QuantProfile>().unwrap(),
            QuantProfile::Int4Gptq
        );
        assert_eq!(
            "awq".parse::<QuantProfile>().unwrap(),
            QuantProfile::Int4Awq
        );
        assert_eq!(
            "int4-signround".parse::<QuantProfile>().unwrap(),
            QuantProfile::Int4Autoround
        );
        assert_eq!(QuantProfile::QuipE8.as_str(), "quip-e8");
    }

    #[test]
    fn quant_profile_runtime_classes() {
        assert!(QuantProfile::Int4Hqq.is_native_int4_runtime());
        assert!(QuantProfile::Higgs4.is_runtime_backed_lowbit());
        assert!(!QuantProfile::Bf16.is_native_int4_runtime());
    }

    #[test]
    fn quant_method_meta_round_trips_lower_precision_artifact_fields() {
        let meta = QuantMethodMeta {
            profile: "autoround-int2-4-mixed".to_string(),
            parameters: Some(serde_json::json!({"enable_alg_ext": true})),
            average_bits_per_weight: Some(3.0),
            candidate_weight_bits: vec!["int2".to_string(), "int4".to_string()],
            mixed_precision_assignment: Some(serde_json::json!({
                "layers.0.mlp.down_proj": "int4",
                "layers.1.mlp.down_proj": "int2"
            })),
            calibration_corpus: Some("wikitext2".to_string()),
            calibration_samples: Some(128),
            calibration_seqlen: Some(2048),
            calibration_seed: Some(20260504),
            source_dtype: Some("bf16".to_string()),
            producer_version: Some("auto-round enable_alg_ext".to_string()),
        };

        let s = serde_json::to_string(&meta).unwrap();
        assert!(s.contains("\"average_bits_per_weight\":3.0"));
        assert!(s.contains("\"candidate_weight_bits\":[\"int2\",\"int4\"]"));
        let parsed: QuantMethodMeta = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.profile, "autoround-int2-4-mixed");
        assert_eq!(parsed.average_bits_per_weight, Some(3.0));
        assert_eq!(parsed.candidate_weight_bits, vec!["int2", "int4"]);
        assert!(parsed.mixed_precision_assignment.is_some());
    }
}
