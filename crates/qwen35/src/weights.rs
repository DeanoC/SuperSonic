use std::path::Path;
use std::sync::Arc;

use gpu_hal::{GpuBuffer, ScalarType};

use crate::config::TextConfig;
use crate::loader::{LoadError, WeightLoader};

/// Ensure a small 1D tensor is stored as F32 on the GPU.
/// Kernels for `linear_attn.norm.weight` read it as `float*`, but some bakes
/// (e.g. the GPTQ INT4 bake) can store it as BF16. Upcast when needed so
/// every backend sees the same F32 layout.
fn ensure_f32_on_gpu(
    buf: GpuBuffer,
    ordinal: usize,
) -> Result<GpuBuffer, model_store::Error> {
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
    kernel_ffi::prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, elems, &buf, &mut out)
        .map_err(|e| model_store::Error::Other(format!("norm_w bf16->f32 cast: {e}")))?;
    Ok(out)
}

/// All immutable model weights on GPU.
pub struct Qwen35Weights {
    pub config: TextConfig,
    pub embed_tokens: Arc<GpuBuffer>,
    pub lm_head: Arc<GpuBuffer>,
    pub lm_head_scale: Option<GpuBuffer>,
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
    // INT4 scale/zero for common weights (None when not INT4)
    pub gate_proj_int4_scale: Option<GpuBuffer>,
    pub gate_proj_int4_zero: Option<GpuBuffer>,
    pub up_proj_int4_scale: Option<GpuBuffer>,
    pub up_proj_int4_zero: Option<GpuBuffer>,
    pub down_proj_int4_scale: Option<GpuBuffer>,
    pub down_proj_int4_zero: Option<GpuBuffer>,
    // Linear attention only
    pub linear: Option<LinearWeights>,
    // Full attention only
    pub full: Option<FullWeights>,
}

pub struct LinearWeights {
    pub qkv_proj_w: GpuBuffer,     // [6144, hidden]
    pub z_proj_w: GpuBuffer,       // [2048, hidden]
    pub b_proj_w: GpuBuffer,       // [16, hidden]
    pub a_proj_w: GpuBuffer,       // [16, hidden]
    pub conv1d_w: GpuBuffer,       // [6144, 1, 4]
    pub out_proj_w: GpuBuffer,     // [hidden, 2048]
    pub dt_bias: GpuBuffer,        // [16]
    pub a_log_exp: GpuBuffer,      // [16] — exp(-A_log) precomputed on CPU
    pub norm_w: GpuBuffer,         // [128] — F32
    // FP8 scale_inv (None when BF16)
    pub qkv_proj_scale: Option<GpuBuffer>,
    pub z_proj_scale: Option<GpuBuffer>,
    pub b_proj_scale: Option<GpuBuffer>,
    pub a_proj_scale: Option<GpuBuffer>,
    pub out_proj_scale: Option<GpuBuffer>,
    // INT4 scale/zero (None when not INT4)
    pub qkv_proj_int4_scale: Option<GpuBuffer>,
    pub qkv_proj_int4_zero: Option<GpuBuffer>,
    pub z_proj_int4_scale: Option<GpuBuffer>,
    pub z_proj_int4_zero: Option<GpuBuffer>,
    pub out_proj_int4_scale: Option<GpuBuffer>,
    pub out_proj_int4_zero: Option<GpuBuffer>,
}

pub struct FullWeights {
    pub q_proj_w: GpuBuffer,   // [4096, hidden]
    pub k_proj_w: GpuBuffer,   // [512, hidden]
    pub v_proj_w: GpuBuffer,   // [512, hidden]
    pub o_proj_w: GpuBuffer,   // [hidden, 2048]
    pub q_norm_w: Option<GpuBuffer>,   // [head_dim] when present
    pub k_norm_w: Option<GpuBuffer>,   // [head_dim] when present
    // FP8 scale_inv (None when BF16)
    pub q_proj_scale: Option<GpuBuffer>,
    pub k_proj_scale: Option<GpuBuffer>,
    pub v_proj_scale: Option<GpuBuffer>,
    pub o_proj_scale: Option<GpuBuffer>,
    // INT4 scale/zero (None when not INT4)
    pub q_proj_int4_scale: Option<GpuBuffer>,
    pub q_proj_int4_zero: Option<GpuBuffer>,
    pub k_proj_int4_scale: Option<GpuBuffer>,
    pub k_proj_int4_zero: Option<GpuBuffer>,
    pub v_proj_int4_scale: Option<GpuBuffer>,
    pub v_proj_int4_zero: Option<GpuBuffer>,
    pub o_proj_int4_scale: Option<GpuBuffer>,
    pub o_proj_int4_zero: Option<GpuBuffer>,
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

        let embed_tokens = Arc::new(loader.load_to_gpu(
            &format!("{prefix}.embed_tokens.weight"),
            ordinal,
        )?);

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

            let input_norm_w = loader.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
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
                    q_proj_scale: None, k_proj_scale: None, v_proj_scale: None, o_proj_scale: None,
                    q_proj_int4_scale: None, q_proj_int4_zero: None,
                    k_proj_int4_scale: None, k_proj_int4_zero: None,
                    v_proj_int4_scale: None, v_proj_int4_zero: None,
                    o_proj_int4_scale: None, o_proj_int4_zero: None,
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
                    b_proj_w: loader.load_to_gpu(&format!("{la}.in_proj_b.weight"), ordinal)?,
                    a_proj_w: loader.load_to_gpu(&format!("{la}.in_proj_a.weight"), ordinal)?,
                    conv1d_w: loader.load_to_gpu(&format!("{la}.conv1d.weight"), ordinal)?,
                    out_proj_w: loader.load_to_gpu(&format!("{la}.out_proj.weight"), ordinal)?,
                    dt_bias: loader.load_to_gpu(&format!("{la}.dt_bias"), ordinal)?,
                    a_log_exp,
                    norm_w: ensure_f32_on_gpu(
                        loader.load_to_gpu(&format!("{la}.norm.weight"), ordinal)?,
                        ordinal,
                    ).map_err(|e| LoadError::UnsupportedDtype(e.to_string()))?,
                    qkv_proj_scale: None, z_proj_scale: None, b_proj_scale: None,
                    a_proj_scale: None, out_proj_scale: None,
                    qkv_proj_int4_scale: None, qkv_proj_int4_zero: None,
                    z_proj_int4_scale: None, z_proj_int4_zero: None,
                    out_proj_int4_scale: None, out_proj_int4_zero: None,
                };
                (Some(linear), None)
            };

            layers.push(LayerWeights {
                kind: if is_full { LayerKind::Full } else { LayerKind::Linear },
                input_norm_w,
                post_attn_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                gate_proj_scale: None, up_proj_scale: None, down_proj_scale: None,
                gate_proj_int4_scale: None, gate_proj_int4_zero: None,
                up_proj_int4_scale: None, up_proj_int4_zero: None,
                down_proj_int4_scale: None, down_proj_int4_zero: None,
                linear,
                full,
            });
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            lm_head_scale: None,
            norm_weight,
            layers,
            is_fp8: false,
            fp8_block_size: 0,
            is_int4: false,
            int4_group_size: 0,
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
        let load_int4 = |name: &str| -> Result<(Option<GpuBuffer>, Option<GpuBuffer>), model_store::Error> {
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

        let embed_tokens = Arc::new(
            store.load_to_gpu(&format!("{prefix}.embed_tokens.weight"), ordinal)?,
        );

        let lm_head_name = "lm_head.weight";
        let lm_head = if store.contains(lm_head_name) {
            Arc::new(store.load_to_gpu(lm_head_name, ordinal)?)
        } else {
            embed_tokens.clone()
        };
        let lm_head_scale = load_scale(lm_head_name)?;

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
                    let packed_cols = gate_proj_w.shape()[1];  // cols/2 in packed format
                    let original_cols = packed_cols * 2;
                    if s_shape.len() == 2 && s_shape[1] > 0 {
                        int4_group_size = original_cols / s_shape[1];
                    } else {
                        int4_group_size = 128;
                    }
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
                    q_proj_int4_scale: q_i4s, q_proj_int4_zero: q_i4z,
                    k_proj_int4_scale: k_i4s, k_proj_int4_zero: k_i4z,
                    v_proj_int4_scale: v_i4s, v_proj_int4_zero: v_i4z,
                    o_proj_int4_scale: o_i4s, o_proj_int4_zero: o_i4z,
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
                let linear = LinearWeights {
                    qkv_proj_w: store.load_to_gpu(&qkv_name, ordinal)?,
                    z_proj_w: store.load_to_gpu(&z_name, ordinal)?,
                    b_proj_w: store.load_to_gpu(&b_name, ordinal)?,
                    a_proj_w: store.load_to_gpu(&a_name, ordinal)?,
                    conv1d_w: store.load_to_gpu(&format!("{la}.conv1d.weight"), ordinal)?,
                    out_proj_w: store.load_to_gpu(&out_name, ordinal)?,
                    dt_bias: store.load_to_gpu(&format!("{la}.dt_bias"), ordinal)?,
                    a_log_exp: store.load_to_gpu(&format!("{la}.A_log"), ordinal)?,
                    norm_w: ensure_f32_on_gpu(
                        store.load_to_gpu(&format!("{la}.norm.weight"), ordinal)?,
                        ordinal,
                    )?,
                    qkv_proj_scale: load_scale(&qkv_name)?,
                    z_proj_scale: load_scale(&z_name)?,
                    b_proj_scale: load_scale(&b_name)?,
                    a_proj_scale: load_scale(&a_name)?,
                    out_proj_scale: load_scale(&out_name)?,
                    qkv_proj_int4_scale: qkv_i4s, qkv_proj_int4_zero: qkv_i4z,
                    z_proj_int4_scale: z_i4s, z_proj_int4_zero: z_i4z,
                    out_proj_int4_scale: out_i4s, out_proj_int4_zero: out_i4z,
                };
                (Some(linear), None)
            };

            layers.push(LayerWeights {
                kind: if is_full { LayerKind::Full } else { LayerKind::Linear },
                input_norm_w,
                post_attn_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                gate_proj_scale, up_proj_scale, down_proj_scale,
                gate_proj_int4_scale, gate_proj_int4_zero,
                up_proj_int4_scale, up_proj_int4_zero,
                down_proj_int4_scale, down_proj_int4_zero,
                linear,
                full,
            });
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            lm_head_scale,
            norm_weight,
            layers,
            is_fp8,
            fp8_block_size,
            is_int4,
            int4_group_size,
        })
    }
}
