use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use model_store::gqh::GqhHeader;
use model_store::manifest::LayoutTag;

use crate::config::TextConfig;
use crate::loader::{LoadError, WeightLoader};

/// Ensure a small 1D tensor is stored as F32 on the GPU.
/// Kernels for `linear_attn.norm.weight` read it as `float*`, but some bakes
/// (e.g. the GPTQ INT4 bake) can store it as BF16. Upcast when needed so
/// every backend sees the same F32 layout.
fn ensure_f32_on_gpu(buf: GpuBuffer, ordinal: usize) -> Result<GpuBuffer, model_store::Error> {
    if buf.dtype() == ScalarType::F32 {
        return Ok(buf);
    }
    if buf.dtype() != ScalarType::BF16 {
        return Err(model_store::Error::Other(format!(
            "norm.weight: unsupported dtype {:?} (expected F32 or BF16)",
            buf.dtype()
        )));
    }
    let elems = buf.elem_count();
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::F32, buf.shape())
        .map_err(|e| model_store::Error::Other(format!("norm_w upcast alloc: {e}")))?;
    kernel_ffi::prefill_ffi::cast(
        ordinal,
        ScalarType::BF16,
        ScalarType::F32,
        elems,
        &buf,
        &mut out,
    )
    .map_err(|e| model_store::Error::Other(format!("norm_w bf16->f32 cast: {e}")))?;
    Ok(out)
}

fn ensure_bf16_on_gpu(buf: GpuBuffer, ordinal: usize) -> Result<GpuBuffer, model_store::Error> {
    if buf.dtype() == ScalarType::BF16 {
        return Ok(buf);
    }
    if buf.dtype() != ScalarType::F32 {
        return Err(model_store::Error::Other(format!(
            "norm.weight: unsupported dtype {:?} (expected F32 or BF16)",
            buf.dtype()
        )));
    }
    let elems = buf.elem_count();
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, buf.shape())
        .map_err(|e| model_store::Error::Other(format!("norm_w bf16 alloc: {e}")))?;
    kernel_ffi::prefill_ffi::cast(
        ordinal,
        ScalarType::F32,
        ScalarType::BF16,
        elems,
        &buf,
        &mut out,
    )
    .map_err(|e| model_store::Error::Other(format!("norm_w f32->bf16 cast: {e}")))?;
    Ok(out)
}

fn fused_ba_projection_from_store(
    store: &model_store::BakedStore,
    b_name: &str,
    a_name: &str,
    ordinal: usize,
) -> Result<Option<GpuBuffer>, model_store::Error> {
    let Some(b_meta) = store.meta(b_name) else {
        return Ok(None);
    };
    let Some(a_meta) = store.meta(a_name) else {
        return Ok(None);
    };
    if b_meta.dtype != "bf16"
        || a_meta.dtype != "bf16"
        || b_meta.layout != LayoutTag::Raw
        || a_meta.layout != LayoutTag::Raw
        || b_meta.shape != a_meta.shape
        || b_meta.shape.len() != 2
    {
        return Ok(None);
    }

    let b_bytes = store
        .raw_bytes(b_name)
        .ok_or_else(|| model_store::Error::NotFound(b_name.to_string()))?;
    let a_bytes = store
        .raw_bytes(a_name)
        .ok_or_else(|| model_store::Error::NotFound(a_name.to_string()))?;
    let mut fused = Vec::with_capacity(b_bytes.len() + a_bytes.len());
    fused.extend_from_slice(b_bytes);
    fused.extend_from_slice(a_bytes);

    let rows = b_meta.shape[0] * 2;
    let cols = b_meta.shape[1];
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[rows, cols], &fused)
        .map(Some)
        .map_err(model_store::Error::Gpu)
}

fn fused_qkvz_projection_from_store(
    store: &model_store::BakedStore,
    qkv_name: &str,
    z_name: &str,
    ordinal: usize,
) -> Result<Option<GpuBuffer>, model_store::Error> {
    let Some(qkv_meta) = store.meta(qkv_name) else {
        return Ok(None);
    };
    let Some(z_meta) = store.meta(z_name) else {
        return Ok(None);
    };
    if qkv_meta.dtype != "u8"
        || z_meta.dtype != "u8"
        || qkv_meta.layout != z_meta.layout
        || !matches!(
            qkv_meta.layout,
            LayoutTag::GgmlQ4K | LayoutTag::GgmlQ5K | LayoutTag::GgmlQ6K
        )
        || qkv_meta.shape.len() != 2
        || z_meta.shape.len() != 2
        || qkv_meta.shape[1] != z_meta.shape[1]
    {
        return Ok(None);
    }

    let qkv_bytes = store
        .raw_bytes(qkv_name)
        .ok_or_else(|| model_store::Error::NotFound(qkv_name.to_string()))?;
    let z_bytes = store
        .raw_bytes(z_name)
        .ok_or_else(|| model_store::Error::NotFound(z_name.to_string()))?;
    let mut fused = Vec::with_capacity(qkv_bytes.len() + z_bytes.len());
    fused.extend_from_slice(qkv_bytes);
    fused.extend_from_slice(z_bytes);

    let rows = qkv_meta.shape[0] + z_meta.shape[0];
    let cols = qkv_meta.shape[1];
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[rows, cols], &fused)
        .map(Some)
        .map_err(model_store::Error::Gpu)
}

fn fused_qkvz_enabled(config: &TextConfig) -> bool {
    if std::env::var_os("SUPERSONIC_DISABLE_FUSED_QKVZ").is_some() {
        return false;
    }
    std::env::var_os("SUPERSONIC_ENABLE_FUSED_QKVZ").is_some()
        || (config.hidden_size == 5120 && config.num_hidden_layers == 64)
}

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
        LOWBIT_ROCMFP3_MIX => model_store::dmix2::row_bytes(
            model_store::dmix2::GGML_TYPE_Q3_1_ROCMFP3_MIX,
            cols,
        )
        .ok(),
        LOWBIT_ROCMFP2_MIX => model_store::dmix2::row_bytes(
            model_store::dmix2::GGML_TYPE_Q2_1_ROCMFP2_MIX,
            cols,
        )
        .ok(),
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
    let rung = kernel_ffi::gqh::rung_from_ggml_type(qtype as u32).ok_or_else(|| {
        GpuError::InvalidArg(format!("not a GQH qtype: {qtype}"))
    })?;
    let header = kernel_ffi::gqh::lookup_header(weight.as_ptr());
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
    let mix = kernel_ffi::gqh::lookup_mix(weight.as_ptr()).ok_or_else(|| {
        GpuError::InvalidArg("mix sidecar not registered for weight buffer".into())
    })?;
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_mix(
        ordinal,
        m,
        n,
        k,
        lhs,
        weight,
        mix.mode,
        &mix.lut,
        qtype,
        out,
    )
}

/// All immutable model weights on GPU.
pub struct Qwen35Weights {
    pub config: TextConfig,
    pub weight_prefix: String,
    pub embed_tokens: Arc<GpuBuffer>,
    pub lm_head: Arc<GpuBuffer>,
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
    /// Optional baked INT8 store kept alive for host-side mixed-path access.
    pub int8_baked_store: Option<Arc<model_store::BakedStore>>,
    /// Outlier threshold used by the mixed INT8 path.
    pub int8_outlier_threshold: f32,
    /// Per-tensor GQH headers from `geoquant.gqh.headers`, keyed by role name.
    pub gqh_headers: BTreeMap<String, GqhHeader>,
    /// Device 8-byte `{tensor_scale, grid_code}` sidecars for the megakernel.
    /// Kept alive so `INT4ScaleDesc` pointers stay valid.
    pub gqh_sidecars: BTreeMap<String, GpuBuffer>,
}

impl Qwen35Weights {
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

impl Qwen35Weights {
    /// Load all weights from a HuggingFace model directory.
    pub fn load(
        model_dir: &Path,
        config: &TextConfig,
        ordinal: usize,
        weight_prefix: &str,
    ) -> Result<Self, LoadError> {
        let loader = WeightLoader::from_dir(model_dir)?;
        let prefix = weight_prefix;

        let embed_tokens =
            Arc::new(loader.load_to_gpu(&format!("{prefix}.embed_tokens.weight"), ordinal)?);

        // lm_head: tied to embed_tokens if not present
        let lm_head = if loader.contains("lm_head.weight") {
            Arc::new(loader.load_to_gpu("lm_head.weight", ordinal)?)
        } else {
            embed_tokens.clone()
        };

        let norm_weight = loader.load_to_gpu(&format!("{prefix}.norm.weight"), ordinal)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{idx}");
            let is_full = config.is_full_attention(idx);

            let input_norm_w =
                loader.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
            let post_attn_norm_w =
                loader.load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)?;
            let gate_proj_w = loader.load_to_gpu(&format!("{lp}.mlp.gate_proj.weight"), ordinal)?;
            let up_proj_w = loader.load_to_gpu(&format!("{lp}.mlp.up_proj.weight"), ordinal)?;
            let down_proj_w = loader.load_to_gpu(&format!("{lp}.mlp.down_proj.weight"), ordinal)?;

            let (linear, full) = if is_full {
                let fa = format!("{lp}.self_attn");
                let full = FullWeights {
                    q_proj_w: loader.load_to_gpu(&format!("{fa}.q_proj.weight"), ordinal)?,
                    k_proj_w: loader.load_to_gpu(&format!("{fa}.k_proj.weight"), ordinal)?,
                    v_proj_w: loader.load_to_gpu(&format!("{fa}.v_proj.weight"), ordinal)?,
                    o_proj_w: loader.load_to_gpu(&format!("{fa}.o_proj.weight"), ordinal)?,
                    q_norm_w: Some(loader.load_to_gpu(&format!("{fa}.q_norm.weight"), ordinal)?),
                    k_norm_w: Some(loader.load_to_gpu(&format!("{fa}.k_norm.weight"), ordinal)?),
                    q_proj_scale: None,
                    k_proj_scale: None,
                    v_proj_scale: None,
                    o_proj_scale: None,
                    q_proj_int8_scale: None,
                    k_proj_int8_scale: None,
                    v_proj_int8_scale: None,
                    o_proj_int8_scale: None,
                    q_proj_int4_scale: None,
                    q_proj_int4_zero: None,
                    q_proj_awq_inv_scale: None,
                    k_proj_int4_scale: None,
                    k_proj_int4_zero: None,
                    k_proj_awq_inv_scale: None,
                    v_proj_int4_scale: None,
                    v_proj_int4_zero: None,
                    v_proj_awq_inv_scale: None,
                    o_proj_int4_scale: None,
                    o_proj_int4_zero: None,
                    o_proj_awq_inv_scale: None,
                };
                (None, Some(full))
            } else {
                let la = format!("{lp}.linear_attn");
                // A_log is stored as F32; the kernel expects exp(A_log) as BF16.
                let a_log_raw = loader.load_to_gpu(&format!("{la}.A_log"), ordinal)?;
                // Precompute exp(A_log) on CPU, upload as BF16
                let a_log_host = a_log_raw.to_host_bytes()?;
                let a_log_f32: Vec<f32> = a_log_host
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let a_log_exp_bf16: Vec<u8> = a_log_f32
                    .iter()
                    .flat_map(|&v| {
                        let exp_val = v.exp();
                        half::bf16::from_f32(exp_val).to_le_bytes()
                    })
                    .collect();
                let num_heads = a_log_f32.len();
                let a_log_exp = GpuBuffer::from_host_bytes(
                    ordinal,
                    gpu_hal::ScalarType::BF16,
                    &[num_heads],
                    &a_log_exp_bf16,
                )?;

                let linear = LinearWeights {
                    qkv_proj_w: loader.load_to_gpu(&format!("{la}.in_proj_qkv.weight"), ordinal)?,
                    z_proj_w: loader.load_to_gpu(&format!("{la}.in_proj_z.weight"), ordinal)?,
                    qkvz_proj_w: None,
                    b_proj_w: loader.load_to_gpu(&format!("{la}.in_proj_b.weight"), ordinal)?,
                    a_proj_w: loader.load_to_gpu(&format!("{la}.in_proj_a.weight"), ordinal)?,
                    ba_proj_w: None,
                    conv1d_w: loader.load_to_gpu(&format!("{la}.conv1d.weight"), ordinal)?,
                    out_proj_w: loader.load_to_gpu(&format!("{la}.out_proj.weight"), ordinal)?,
                    dt_bias: loader.load_to_gpu(&format!("{la}.dt_bias"), ordinal)?,
                    a_log_exp,
                    norm_w: ensure_f32_on_gpu(
                        loader.load_to_gpu(&format!("{la}.norm.weight"), ordinal)?,
                        ordinal,
                    )
                    .map_err(|e| LoadError::UnsupportedDtype(e.to_string()))?,
                    norm_w_bf16: ensure_bf16_on_gpu(
                        loader.load_to_gpu(&format!("{la}.norm.weight"), ordinal)?,
                        ordinal,
                    )
                    .map_err(|e| LoadError::UnsupportedDtype(e.to_string()))?,
                    qkv_proj_scale: None,
                    z_proj_scale: None,
                    b_proj_scale: None,
                    a_proj_scale: None,
                    out_proj_scale: None,
                    qkv_proj_int8_scale: None,
                    z_proj_int8_scale: None,
                    b_proj_int8_scale: None,
                    a_proj_int8_scale: None,
                    out_proj_int8_scale: None,
                    qkv_proj_int4_scale: None,
                    qkv_proj_int4_zero: None,
                    qkv_proj_awq_inv_scale: None,
                    z_proj_int4_scale: None,
                    z_proj_int4_zero: None,
                    z_proj_awq_inv_scale: None,
                    out_proj_int4_scale: None,
                    out_proj_int4_zero: None,
                    out_proj_awq_inv_scale: None,
                };
                (Some(linear), None)
            };

            layers.push(LayerWeights {
                kind: if is_full {
                    LayerKind::Full
                } else {
                    LayerKind::Linear
                },
                input_norm_w,
                post_attn_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                gate_proj_scale: None,
                up_proj_scale: None,
                down_proj_scale: None,
                gate_proj_int8_scale: None,
                up_proj_int8_scale: None,
                down_proj_int8_scale: None,
                gate_proj_int4_scale: None,
                gate_proj_int4_zero: None,
                gate_proj_awq_inv_scale: None,
                up_proj_int4_scale: None,
                up_proj_int4_zero: None,
                up_proj_awq_inv_scale: None,
                down_proj_int4_scale: None,
                down_proj_int4_zero: None,
                down_proj_awq_inv_scale: None,
                linear,
                full,
            });
        }

        Ok(Self {
            config: config.clone(),
            weight_prefix: prefix.to_string(),
            embed_tokens,
            lm_head,
            lm_head_scale: None,
            lm_head_int4_scale: None,
            lm_head_int4_zero: None,
            lm_head_awq_inv_scale: None,
            norm_weight,
            layers,
            is_fp8: false,
            fp8_block_size: 0,
            is_int4: false,
            int4_group_size: 0,
            is_int8: false,
            int8_baked_store: None,
            int8_outlier_threshold: 0.0,
            gqh_headers: BTreeMap::new(),
            gqh_sidecars: BTreeMap::new(),
        })
    }

    /// Load all weights from a baked SuperSonic package.
    /// No CPU transforms needed — everything was pre-processed at bake time.
    /// When the baked package contains FP8-native weights (LayoutTag::Fp8Native),
    /// scale_inv tensors are loaded alongside each weight for runtime dequant.
    pub fn load_baked(
        store: &model_store::BakedStore,
        config: &TextConfig,
        ordinal: usize,
        weight_prefix: &str,
    ) -> Result<Self, model_store::Error> {
        let prefix = weight_prefix;

        // Helper: load a scale tensor if it exists in the store.
        let load_scale = |name: &str| -> Result<Option<GpuBuffer>, model_store::Error> {
            let scale_name = format!("{name}_scale_inv");
            if store.contains(&scale_name) {
                Ok(Some(store.load_to_gpu(&scale_name, ordinal)?))
            } else {
                Ok(None)
            }
        };

        // Helper: load INT4 scale/zero tensor pair if they exist.
        let load_int4 =
            |name: &str| -> Result<(Option<GpuBuffer>, Option<GpuBuffer>), model_store::Error> {
                let scale_name = format!("{name}_int4_scale");
                let zero_name = format!("{name}_int4_zero");
                if store.contains(&scale_name) && store.contains(&zero_name) {
                    Ok((
                        Some(store.load_to_gpu(&scale_name, ordinal)?),
                        Some(store.load_to_gpu(&zero_name, ordinal)?),
                    ))
                } else {
                    Ok((None, None))
                }
            };
        let load_awq_inv_scale = |name: &str| -> Result<Option<GpuBuffer>, model_store::Error> {
            let scale_name = format!("{name}_awq_inv_scale");
            if store.contains(&scale_name) {
                Ok(Some(store.load_to_gpu(&scale_name, ordinal)?))
            } else {
                Ok(None)
            }
        };
        let is_ggml_lowbit = |name: &str| -> bool {
            matches!(
                store.layout(name),
                Some(LayoutTag::GgmlQ4K | LayoutTag::GgmlQ5K | LayoutTag::GgmlQ6K)
            )
        };

        let embed_tokens =
            Arc::new(store.load_to_gpu(&format!("{prefix}.embed_tokens.weight"), ordinal)?);

        let lm_head_name = "lm_head.weight";
        let lm_head = if store.contains(lm_head_name) {
            Arc::new(store.load_to_gpu(lm_head_name, ordinal)?)
        } else {
            embed_tokens.clone()
        };
        let lm_head_scale = load_scale(lm_head_name)?;
        let (lm_head_int4_scale, lm_head_int4_zero) = load_int4(lm_head_name)?;
        let lm_head_awq_inv_scale = load_awq_inv_scale(lm_head_name)?;

        let norm_weight = store.load_to_gpu(&format!("{prefix}.norm.weight"), ordinal)?;

        let mut is_fp8 = false;
        let mut fp8_block_size: usize = 0;
        let mut is_int4 = false;
        let mut int4_group_size: usize = 0;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{idx}");
            let is_full = config.is_full_attention(idx);

            let input_norm_w =
                store.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
            let post_attn_norm_w =
                store.load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)?;

            let gate_name = format!("{lp}.mlp.gate_proj.weight");
            let up_name = format!("{lp}.mlp.up_proj.weight");
            let down_name = format!("{lp}.mlp.down_proj.weight");
            let gate_proj_w = store.load_to_gpu(&gate_name, ordinal)?;
            let up_proj_w = store.load_to_gpu(&up_name, ordinal)?;
            let down_proj_w = store.load_to_gpu(&down_name, ordinal)?;
            let gate_proj_scale = load_scale(&gate_name)?;
            let up_proj_scale = load_scale(&up_name)?;
            let down_proj_scale = load_scale(&down_name)?;
            let (gate_proj_int4_scale, gate_proj_int4_zero) = load_int4(&gate_name)?;
            let (up_proj_int4_scale, up_proj_int4_zero) = load_int4(&up_name)?;
            let (down_proj_int4_scale, down_proj_int4_zero) = load_int4(&down_name)?;
            let gate_proj_awq_inv_scale = load_awq_inv_scale(&gate_name)?;
            let up_proj_awq_inv_scale = load_awq_inv_scale(&up_name)?;
            let down_proj_awq_inv_scale = load_awq_inv_scale(&down_name)?;

            // Detect FP8 and compute block_size from first scale tensor encountered
            if !is_fp8 {
                if let Some(ref scale) = gate_proj_scale {
                    is_fp8 = true;
                    let w_shape = gate_proj_w.shape();
                    let s_shape = scale.shape();
                    if s_shape[0] > 0 {
                        fp8_block_size = w_shape[0] / s_shape[0];
                    } else {
                        fp8_block_size = 128;
                    }
                }
            }

            // Detect INT4 and compute group_size from first INT4 scale tensor
            if !is_int4 {
                if let Some(ref i4_scale) = gate_proj_int4_scale {
                    is_int4 = true;
                    // For INT4: packed weight shape is [rows, cols/2], scale shape is [rows/gsz, cols/gsz]
                    // We stored original cols = packed_cols * 2 in the scale tensor dims.
                    // group_size = original_cols / scale_cols = (packed_cols * 2) / scale_cols
                    let s_shape = i4_scale.shape();
                    let packed_cols = gate_proj_w.shape()[1]; // cols/2 in packed format
                    let original_cols = packed_cols * 2;
                    if s_shape.len() == 2 && s_shape[1] > 0 {
                        int4_group_size = original_cols / s_shape[1];
                    } else {
                        int4_group_size = 128;
                    }
                } else if is_ggml_lowbit(&gate_name) {
                    is_int4 = true;
                    int4_group_size = 128;
                }
            }

            let (linear, full) = if is_full {
                let fa = format!("{lp}.self_attn");
                let q_name = format!("{fa}.q_proj.weight");
                let k_name = format!("{fa}.k_proj.weight");
                let v_name = format!("{fa}.v_proj.weight");
                let o_name = format!("{fa}.o_proj.weight");
                let (q_i4s, q_i4z) = load_int4(&q_name)?;
                let (k_i4s, k_i4z) = load_int4(&k_name)?;
                let (v_i4s, v_i4z) = load_int4(&v_name)?;
                let (o_i4s, o_i4z) = load_int4(&o_name)?;
                let q_awq_inv = load_awq_inv_scale(&q_name)?;
                let k_awq_inv = load_awq_inv_scale(&k_name)?;
                let v_awq_inv = load_awq_inv_scale(&v_name)?;
                let o_awq_inv = load_awq_inv_scale(&o_name)?;
                let full = FullWeights {
                    q_proj_w: store.load_to_gpu(&q_name, ordinal)?,
                    k_proj_w: store.load_to_gpu(&k_name, ordinal)?,
                    v_proj_w: store.load_to_gpu(&v_name, ordinal)?,
                    o_proj_w: store.load_to_gpu(&o_name, ordinal)?,
                    q_norm_w: Some(store.load_to_gpu(&format!("{fa}.q_norm.weight"), ordinal)?),
                    k_norm_w: Some(store.load_to_gpu(&format!("{fa}.k_norm.weight"), ordinal)?),
                    q_proj_scale: load_scale(&q_name)?,
                    k_proj_scale: load_scale(&k_name)?,
                    v_proj_scale: load_scale(&v_name)?,
                    o_proj_scale: load_scale(&o_name)?,
                    q_proj_int8_scale: None,
                    k_proj_int8_scale: None,
                    v_proj_int8_scale: None,
                    o_proj_int8_scale: None,
                    q_proj_int4_scale: q_i4s,
                    q_proj_int4_zero: q_i4z,
                    q_proj_awq_inv_scale: q_awq_inv,
                    k_proj_int4_scale: k_i4s,
                    k_proj_int4_zero: k_i4z,
                    k_proj_awq_inv_scale: k_awq_inv,
                    v_proj_int4_scale: v_i4s,
                    v_proj_int4_zero: v_i4z,
                    v_proj_awq_inv_scale: v_awq_inv,
                    o_proj_int4_scale: o_i4s,
                    o_proj_int4_zero: o_i4z,
                    o_proj_awq_inv_scale: o_awq_inv,
                };
                (None, Some(full))
            } else {
                let la = format!("{lp}.linear_attn");
                let qkv_name = format!("{la}.in_proj_qkv.weight");
                let z_name = format!("{la}.in_proj_z.weight");
                let b_name = format!("{la}.in_proj_b.weight");
                let a_name = format!("{la}.in_proj_a.weight");
                let out_name = format!("{la}.out_proj.weight");
                let (qkv_i4s, qkv_i4z) = load_int4(&qkv_name)?;
                let (z_i4s, z_i4z) = load_int4(&z_name)?;
                let (out_i4s, out_i4z) = load_int4(&out_name)?;
                let qkv_awq_inv = load_awq_inv_scale(&qkv_name)?;
                let z_awq_inv = load_awq_inv_scale(&z_name)?;
                let out_awq_inv = load_awq_inv_scale(&out_name)?;
                let ba_proj_w = fused_ba_projection_from_store(store, &b_name, &a_name, ordinal)?;
                let qkvz_proj_w = if fused_qkvz_enabled(config) {
                    fused_qkvz_projection_from_store(store, &qkv_name, &z_name, ordinal)?
                } else {
                    None
                };
                let linear = LinearWeights {
                    qkv_proj_w: store.load_to_gpu(&qkv_name, ordinal)?,
                    z_proj_w: store.load_to_gpu(&z_name, ordinal)?,
                    qkvz_proj_w,
                    b_proj_w: store.load_to_gpu(&b_name, ordinal)?,
                    a_proj_w: store.load_to_gpu(&a_name, ordinal)?,
                    ba_proj_w,
                    conv1d_w: store.load_to_gpu(&format!("{la}.conv1d.weight"), ordinal)?,
                    out_proj_w: store.load_to_gpu(&out_name, ordinal)?,
                    dt_bias: store.load_to_gpu(&format!("{la}.dt_bias"), ordinal)?,
                    a_log_exp: store.load_to_gpu(&format!("{la}.A_log"), ordinal)?,
                    norm_w: ensure_f32_on_gpu(
                        store.load_to_gpu(&format!("{la}.norm.weight"), ordinal)?,
                        ordinal,
                    )?,
                    norm_w_bf16: ensure_bf16_on_gpu(
                        store.load_to_gpu(&format!("{la}.norm.weight"), ordinal)?,
                        ordinal,
                    )?,
                    qkv_proj_scale: load_scale(&qkv_name)?,
                    z_proj_scale: load_scale(&z_name)?,
                    b_proj_scale: load_scale(&b_name)?,
                    a_proj_scale: load_scale(&a_name)?,
                    out_proj_scale: load_scale(&out_name)?,
                    qkv_proj_int8_scale: None,
                    z_proj_int8_scale: None,
                    b_proj_int8_scale: None,
                    a_proj_int8_scale: None,
                    out_proj_int8_scale: None,
                    qkv_proj_int4_scale: qkv_i4s,
                    qkv_proj_int4_zero: qkv_i4z,
                    qkv_proj_awq_inv_scale: qkv_awq_inv,
                    z_proj_int4_scale: z_i4s,
                    z_proj_int4_zero: z_i4z,
                    z_proj_awq_inv_scale: z_awq_inv,
                    out_proj_int4_scale: out_i4s,
                    out_proj_int4_zero: out_i4z,
                    out_proj_awq_inv_scale: out_awq_inv,
                };
                (Some(linear), None)
            };

            layers.push(LayerWeights {
                kind: if is_full {
                    LayerKind::Full
                } else {
                    LayerKind::Linear
                },
                input_norm_w,
                post_attn_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                gate_proj_scale,
                up_proj_scale,
                down_proj_scale,
                gate_proj_int8_scale: None,
                up_proj_int8_scale: None,
                down_proj_int8_scale: None,
                gate_proj_int4_scale,
                gate_proj_int4_zero,
                gate_proj_awq_inv_scale,
                up_proj_int4_scale,
                up_proj_int4_zero,
                up_proj_awq_inv_scale,
                down_proj_int4_scale,
                down_proj_int4_zero,
                down_proj_awq_inv_scale,
                linear,
                full,
            });
        }

        Ok(Self {
            config: config.clone(),
            weight_prefix: prefix.to_string(),
            embed_tokens,
            lm_head,
            lm_head_scale,
            lm_head_int4_scale,
            lm_head_int4_zero,
            lm_head_awq_inv_scale,
            norm_weight,
            layers,
            is_fp8,
            fp8_block_size,
            is_int4,
            int4_group_size,
            is_int8: false,
            int8_baked_store: None,
            int8_outlier_threshold: 0.0,
            gqh_headers: BTreeMap::new(),
            gqh_sidecars: BTreeMap::new(),
        })
    }
}
