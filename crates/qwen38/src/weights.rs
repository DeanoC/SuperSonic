use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use model_store::gqh::GqhHeader;

use crate::config::TextConfig;

pub const LOWBIT_NATIVE_INT4: i32 = 4;
pub const LOWBIT_HIGGS4: i32 = 20;
pub const LOWBIT_QUIP_E8: i32 = 21;
pub const LOWBIT_QTIP_TRELLIS2: i32 = 22;
pub const LOWBIT_GGML_Q8_0: i32 = 8;
pub const LOWBIT_GGML_Q2_K: i32 = 10;
pub const LOWBIT_GGML_Q3_K: i32 = 11;
pub const LOWBIT_GGML_Q4_K: i32 = 12;
pub const LOWBIT_GGML_Q5_K: i32 = 13;
pub const LOWBIT_GGML_Q6_K: i32 = 14;
pub const LOWBIT_ROCMFP3_MIX: i32 = 105;
pub const LOWBIT_ROCMFP2_MIX: i32 = 106;
pub const LOWBIT_GQH3: i32 = 108;
pub const LOWBIT_GQH2_H: i32 = 109;
pub const LOWBIT_GQH2_C: i32 = 110;
pub const LOWBIT_GQH4: i32 = 111;

pub fn ggml_k_row_bytes(qtype: i32, cols: usize) -> Option<usize> {
    if qtype == LOWBIT_GGML_Q8_0 {
        return (cols % 32 == 0).then_some((cols / 32) * 34);
    }
    if cols % 256 != 0 {
        return None;
    }
    let blocks = cols / 256;
    match qtype {
        LOWBIT_GGML_Q2_K => Some(blocks * 84),
        LOWBIT_GGML_Q4_K => Some(blocks * 144),
        LOWBIT_GGML_Q5_K => Some(blocks * 176),
        LOWBIT_GGML_Q6_K => Some(blocks * 210),
        LOWBIT_ROCMFP3_MIX => {
            model_store::dmix2::row_bytes(model_store::dmix2::GGML_TYPE_Q3_1_ROCMFP3_MIX, cols).ok()
        }
        LOWBIT_ROCMFP2_MIX => {
            model_store::dmix2::row_bytes(model_store::dmix2::GGML_TYPE_Q2_1_ROCMFP2_MIX, cols).ok()
        }
        LOWBIT_GQH3 => model_store::gqh::device_row_bytes(model_store::gqh::GqhRung::Gqh3, cols),
        LOWBIT_GQH2_H => model_store::gqh::device_row_bytes(model_store::gqh::GqhRung::Gqh2H, cols),
        LOWBIT_GQH2_C => model_store::gqh::device_row_bytes(model_store::gqh::GqhRung::Gqh2C, cols),
        LOWBIT_GQH4 => model_store::gqh::device_row_bytes(model_store::gqh::GqhRung::Gqh4, cols),
        _ => None,
    }
}

pub fn infer_lowbit_type(weight: &GpuBuffer, logical_cols: usize, native_int4: bool) -> i32 {
    if weight.dtype() != ScalarType::U8 || weight.shape().len() < 2 {
        return 0;
    }
    if native_int4 {
        return LOWBIT_NATIVE_INT4;
    }
    let row_bytes = weight.shape()[1];
    for qtype in [
        LOWBIT_GGML_Q8_0,
        LOWBIT_GGML_Q2_K,
        LOWBIT_GGML_Q4_K,
        LOWBIT_GGML_Q5_K,
        LOWBIT_GGML_Q6_K,
        LOWBIT_ROCMFP3_MIX,
        LOWBIT_ROCMFP2_MIX,
        LOWBIT_GQH3,
        LOWBIT_GQH2_H,
        LOWBIT_GQH2_C,
        LOWBIT_GQH4,
    ] {
        if ggml_k_row_bytes(qtype, logical_cols) == Some(row_bytes) {
            return qtype;
        }
    }
    0
}

pub fn is_gqh_qtype(qtype: i32) -> bool {
    kernel_ffi::gqh::rung_from_ggml_type(qtype as u32).is_some()
}

pub fn is_mix_qtype(qtype: i32) -> bool {
    qtype == LOWBIT_ROCMFP3_MIX || qtype == LOWBIT_ROCMFP2_MIX
}

/// GQH fused dequant-matmul using the header registered against `weight`.
/// `lhs` is `[>=m, k]`, `out` is `[>=m, n]`; only the first `m` rows are used.
pub fn matmul_gqh(
    ordinal: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    qtype: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let rung = kernel_ffi::gqh::rung_from_ggml_type(qtype as u32)
        .ok_or_else(|| GpuError::InvalidArg(format!("not a GQH qtype: {qtype}")))?;
    let header = kernel_ffi::gqh::lookup_header(ordinal, weight.as_ptr());
    if header.is_none() && qtype != LOWBIT_GQH2_C {
        return Err(GpuError::InvalidArg(
            "GQH header not registered for weight buffer".into(),
        ));
    }
    let (tensor_scale, grid_code) = header
        .map(|h| (h.tensor_scale, h.grid_code))
        .unwrap_or((0.0, 0));
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_gqh(
        ordinal,
        m,
        n,
        k,
        lhs,
        weight,
        tensor_scale,
        grid_code,
        rung,
        out,
    )
}

/// Mix (105/106) fused dequant-matmul. Same f32 activation convention as GQH.
pub fn matmul_mix(
    ordinal: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    qtype: i32,
    out: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let mix = kernel_ffi::gqh::lookup_mix(ordinal, weight.as_ptr()).ok_or_else(|| {
        GpuError::InvalidArg("mix sidecar not registered for weight buffer".into())
    })?;
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_mix(
        ordinal, m, n, k, lhs, weight, mix.mode, &mix.lut, qtype, out,
    )
}

/// All immutable model weights on GPU.
pub struct Qwen38Weights {
    pub config: TextConfig,
    pub weight_prefix: String,
    pub(crate) embed_tokens: Arc<GpuBuffer>,
    pub(crate) lm_head: Arc<GpuBuffer>,
    pub lm_head_scale: Option<GpuBuffer>,
    /// INT4 GPTQ scale tile for the lm_head when present in the baked package.
    /// Shape `[vocab/group_size, hidden/group_size]`, BF16. Implies `lm_head`
    /// holds packed-u8 nibbles instead of BF16 weights.
    pub lm_head_int4_scale: Option<GpuBuffer>,
    /// INT4 GPTQ zero tile, parallel to `lm_head_int4_scale`. BF16.
    pub lm_head_int4_zero: Option<GpuBuffer>,
    /// Optional AWQ input-channel inverse scale sidecar. BF16 `[hidden]`.
    pub lm_head_awq_inv_scale: Option<GpuBuffer>,
    pub norm_weight: GpuBuffer,
    pub layers: Vec<LayerWeights>,
    /// True if weights are FP8 with runtime dequant (native FP8 bake).
    pub is_fp8: bool,
    /// FP8 quantization block size (typically 128). Only valid when is_fp8.
    pub fp8_block_size: usize,
    /// True if weights are INT4-quantized with runtime dequant.
    pub is_int4: bool,
    /// INT4 quantization group size (typically 128). Only valid when is_int4.
    pub int4_group_size: usize,
    /// True if weights use BitsAndBytes-style INT8 runtime matmul.
    pub is_int8: bool,
    /// Outlier threshold used by the mixed INT8 path.
    pub int8_outlier_threshold: f32,
    /// Per-tensor GQH headers from `geoquant.gqh.headers`, keyed by role name.
    pub gqh_headers: BTreeMap<String, GqhHeader>,
    /// Device 8-byte `{tensor_scale, grid_code}` sidecars for the megakernel.
    /// Kept alive so `INT4ScaleDesc` pointers stay valid.
    pub gqh_sidecars: BTreeMap<String, GpuBuffer>,
    /// Optional NextN/MTP block (`blk.64` on Qwen3.8 GGUF).
    pub mtp: Option<MtpWeights>,
    /// RAII guards for every packed tensor registered with the GQH bridge.
    /// Kept separate from the buffers so cleanup can run before `hipFree`.
    pub(crate) gqh_registrations: kernel_ffi::gqh::RegistrationBatch,
}

/// DeepSeek-style NextN head: enorm/hnorm + eh_proj + one full-attn decoder
/// block. Shares `embed_tokens` / `lm_head` when dedicated tensors are absent.
pub struct MtpWeights {
    pub enorm_w: GpuBuffer,
    pub hnorm_w: GpuBuffer,
    pub eh_proj_w: GpuBuffer,
    pub shared_head_norm_w: GpuBuffer,
    pub layer: LayerWeights,
}

impl Qwen38Weights {
    /// Borrow the shared embedding table without extending its allocation
    /// lifetime past the model-owned GQH registrations.
    pub fn embed_tokens(&self) -> &GpuBuffer {
        self.embed_tokens.as_ref()
    }

    /// Borrow the shared output head without allowing an `Arc` clone to outlive
    /// the model-owned GQH registration guard.
    pub fn lm_head(&self) -> &GpuBuffer {
        self.lm_head.as_ref()
    }

    pub fn lm_head_lowbit_params(
        &self,
        logical_cols: usize,
    ) -> Option<(i32, &GpuBuffer, &GpuBuffer)> {
        let qtype = infer_lowbit_type(
            self.lm_head.as_ref(),
            logical_cols,
            self.lm_head_int4_scale.is_some(),
        );
        if qtype == 0 {
            return None;
        }
        if qtype == LOWBIT_NATIVE_INT4 {
            let scale = self.lm_head_int4_scale.as_ref()?;
            let zero = self.lm_head_int4_zero.as_ref()?;
            Some((qtype, scale, zero))
        } else {
            let weight = self.lm_head.as_ref();
            Some((qtype, weight, weight))
        }
    }

    pub fn gqh_header(&self, role: &str) -> Option<&GqhHeader> {
        self.gqh_headers.get(role)
    }

    pub fn load_gguf(
        gguf_path: &Path,
        config: &TextConfig,
        ordinal: usize,
    ) -> Result<Self, model_store::Error> {
        let file = model_store::gguf::GgufFile::open(gguf_path)?;
        crate::gguf_ingest::load_weights(&file, config, ordinal)
    }
}

#[derive(Clone, Copy)]
pub enum LayerKind {
    Linear,
    Full,
}

pub struct LayerWeights {
    pub kind: LayerKind,
    // Common (all layers)
    pub input_norm_w: GpuBuffer,
    pub post_attn_norm_w: GpuBuffer,
    pub gate_proj_w: GpuBuffer,
    pub up_proj_w: GpuBuffer,
    pub down_proj_w: GpuBuffer,
    // FP8 scale_inv for common weights (None when BF16)
    pub gate_proj_scale: Option<GpuBuffer>,
    pub up_proj_scale: Option<GpuBuffer>,
    pub down_proj_scale: Option<GpuBuffer>,
    // INT8 per-row scales (BitsAndBytes `.SCB`)
    pub gate_proj_int8_scale: Option<GpuBuffer>,
    pub up_proj_int8_scale: Option<GpuBuffer>,
    pub down_proj_int8_scale: Option<GpuBuffer>,
    // INT4 scale/zero for common weights (None when not INT4)
    pub gate_proj_int4_scale: Option<GpuBuffer>,
    pub gate_proj_int4_zero: Option<GpuBuffer>,
    pub gate_proj_awq_inv_scale: Option<GpuBuffer>,
    pub up_proj_int4_scale: Option<GpuBuffer>,
    pub up_proj_int4_zero: Option<GpuBuffer>,
    pub up_proj_awq_inv_scale: Option<GpuBuffer>,
    pub down_proj_int4_scale: Option<GpuBuffer>,
    pub down_proj_int4_zero: Option<GpuBuffer>,
    pub down_proj_awq_inv_scale: Option<GpuBuffer>,
    // Linear attention only
    pub linear: Option<LinearWeights>,
    // Full attention only
    pub full: Option<FullWeights>,
}

pub struct LinearWeights {
    pub qkv_proj_w: GpuBuffer, // [6144, hidden]
    pub z_proj_w: GpuBuffer,   // [2048, hidden]
    pub qkvz_proj_w: Option<GpuBuffer>,
    pub b_proj_w: GpuBuffer, // [16, hidden]
    pub a_proj_w: GpuBuffer, // [16, hidden]
    pub ba_proj_w: Option<GpuBuffer>,
    pub conv1d_w: GpuBuffer,    // [6144, 1, 4]
    pub out_proj_w: GpuBuffer,  // [hidden, 2048]
    pub dt_bias: GpuBuffer,     // [16]
    pub a_log_exp: GpuBuffer,   // [16] — exp(-A_log) precomputed on CPU
    pub norm_w: GpuBuffer,      // [128] — F32
    pub norm_w_bf16: GpuBuffer, // [128] — BF16 for gated RMSNorm component path
    // FP8 scale_inv (None when BF16)
    pub qkv_proj_scale: Option<GpuBuffer>,
    pub z_proj_scale: Option<GpuBuffer>,
    pub b_proj_scale: Option<GpuBuffer>,
    pub a_proj_scale: Option<GpuBuffer>,
    pub out_proj_scale: Option<GpuBuffer>,
    // INT8 per-row scales (None when not INT8)
    pub qkv_proj_int8_scale: Option<GpuBuffer>,
    pub z_proj_int8_scale: Option<GpuBuffer>,
    pub b_proj_int8_scale: Option<GpuBuffer>,
    pub a_proj_int8_scale: Option<GpuBuffer>,
    pub out_proj_int8_scale: Option<GpuBuffer>,
    // INT4 scale/zero (None when not INT4)
    pub qkv_proj_int4_scale: Option<GpuBuffer>,
    pub qkv_proj_int4_zero: Option<GpuBuffer>,
    pub qkv_proj_awq_inv_scale: Option<GpuBuffer>,
    pub z_proj_int4_scale: Option<GpuBuffer>,
    pub z_proj_int4_zero: Option<GpuBuffer>,
    pub z_proj_awq_inv_scale: Option<GpuBuffer>,
    pub out_proj_int4_scale: Option<GpuBuffer>,
    pub out_proj_int4_zero: Option<GpuBuffer>,
    pub out_proj_awq_inv_scale: Option<GpuBuffer>,
}

pub struct FullWeights {
    pub q_proj_w: GpuBuffer,         // [4096, hidden]
    pub k_proj_w: GpuBuffer,         // [512, hidden]
    pub v_proj_w: GpuBuffer,         // [512, hidden]
    pub o_proj_w: GpuBuffer,         // [hidden, 2048]
    pub q_norm_w: Option<GpuBuffer>, // [head_dim] when present
    pub k_norm_w: Option<GpuBuffer>, // [head_dim] when present
    // FP8 scale_inv (None when BF16)
    pub q_proj_scale: Option<GpuBuffer>,
    pub k_proj_scale: Option<GpuBuffer>,
    pub v_proj_scale: Option<GpuBuffer>,
    pub o_proj_scale: Option<GpuBuffer>,
    // INT8 per-row scales (None when not INT8)
    pub q_proj_int8_scale: Option<GpuBuffer>,
    pub k_proj_int8_scale: Option<GpuBuffer>,
    pub v_proj_int8_scale: Option<GpuBuffer>,
    pub o_proj_int8_scale: Option<GpuBuffer>,
    // INT4 scale/zero (None when not INT4)
    pub q_proj_int4_scale: Option<GpuBuffer>,
    pub q_proj_int4_zero: Option<GpuBuffer>,
    pub q_proj_awq_inv_scale: Option<GpuBuffer>,
    pub k_proj_int4_scale: Option<GpuBuffer>,
    pub k_proj_int4_zero: Option<GpuBuffer>,
    pub k_proj_awq_inv_scale: Option<GpuBuffer>,
    pub v_proj_int4_scale: Option<GpuBuffer>,
    pub v_proj_int4_zero: Option<GpuBuffer>,
    pub v_proj_awq_inv_scale: Option<GpuBuffer>,
    pub o_proj_int4_scale: Option<GpuBuffer>,
    pub o_proj_int4_zero: Option<GpuBuffer>,
    pub o_proj_awq_inv_scale: Option<GpuBuffer>,
}

impl Drop for Qwen38Weights {
    fn drop(&mut self) {
        // Run before Rust drops the owned GpuBuffers. The bridge caches by raw
        // allocation address, so invalidation after hipFree would be too late
        // when HIP reuses that address for the next model.
        self.gqh_registrations.clear();
    }
}
