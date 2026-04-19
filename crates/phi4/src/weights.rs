use std::path::Path;
use std::sync::Arc;

use gpu_hal::GpuBuffer;

use crate::config::Phi4Config;
use crate::loader::{LoadError, WeightLoader};

/// All immutable Phi-4 weights on GPU.
///
/// Phi-4-mini is pure full-attention (32 layers, no sliding / no hybrid linear
/// attention), tied embeddings, and has no QK-norm. Fused `qkv_proj` and
/// `gate_up_proj` tensors are split at load time (raw safetensors path) or
/// pre-split at bake time (baked path).
///
/// INT4 and FP8 slots are held as `Option` placeholders so those backends can
/// be added later without breaking the struct layout.
pub struct Phi4Weights {
    pub config: Phi4Config,
    /// `model.embed_tokens.weight`. Also used as tied lm_head.
    pub embed_tokens: Arc<GpuBuffer>,
    /// `lm_head.weight` if the checkpoint untied embeddings; otherwise aliases
    /// `embed_tokens`.
    pub lm_head: Arc<GpuBuffer>,
    /// Final RMSNorm weight (`model.norm.weight`).
    pub norm_weight: GpuBuffer,
    pub layers: Vec<Phi4LayerWeights>,
    pub is_int4: bool,
    pub int4_group_size: usize,
}

pub struct Phi4LayerWeights {
    pub input_norm_w: GpuBuffer,
    pub post_attn_norm_w: GpuBuffer,
    pub q_proj_w: GpuBuffer,
    pub k_proj_w: GpuBuffer,
    pub v_proj_w: GpuBuffer,
    pub o_proj_w: GpuBuffer,
    pub gate_proj_w: GpuBuffer,
    pub up_proj_w: GpuBuffer,
    pub down_proj_w: GpuBuffer,
    pub q_proj_int4_scale: Option<GpuBuffer>,
    pub q_proj_int4_zero: Option<GpuBuffer>,
    pub k_proj_int4_scale: Option<GpuBuffer>,
    pub k_proj_int4_zero: Option<GpuBuffer>,
    pub v_proj_int4_scale: Option<GpuBuffer>,
    pub v_proj_int4_zero: Option<GpuBuffer>,
    pub o_proj_int4_scale: Option<GpuBuffer>,
    pub o_proj_int4_zero: Option<GpuBuffer>,
    pub gate_proj_int4_scale: Option<GpuBuffer>,
    pub gate_proj_int4_zero: Option<GpuBuffer>,
    pub up_proj_int4_scale: Option<GpuBuffer>,
    pub up_proj_int4_zero: Option<GpuBuffer>,
    pub down_proj_int4_scale: Option<GpuBuffer>,
    pub down_proj_int4_zero: Option<GpuBuffer>,
}

impl Phi4Weights {
    /// Load all weights from a HuggingFace model directory. Splits fused
    /// `qkv_proj` / `gate_up_proj` on the fly.
    pub fn load(
        model_dir: &Path,
        config: &Phi4Config,
        ordinal: usize,
        weight_prefix: &str,
    ) -> Result<Self, LoadError> {
        let loader = WeightLoader::from_dir(model_dir)?;
        let prefix = weight_prefix;

        let embed_tokens = Arc::new(
            loader.load_to_gpu(&format!("{prefix}.embed_tokens.weight"), ordinal)?,
        );

        let lm_head = if loader.contains("lm_head.weight") {
            Arc::new(loader.load_to_gpu("lm_head.weight", ordinal)?)
        } else {
            // tie_word_embeddings=true: reuse embed_tokens.
            embed_tokens.clone()
        };

        let norm_weight = loader.load_to_gpu(&format!("{prefix}.norm.weight"), ordinal)?;

        let head_dim = config.head_dim();
        let q_rows = config.num_attention_heads * head_dim;
        let k_rows = config.num_key_value_heads * head_dim;
        let v_rows = k_rows;
        let intermediate = config.intermediate_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{idx}");
            let input_norm_w =
                loader.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
            let post_attn_norm_w =
                loader.load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)?;

            // Split fused qkv_proj → q / k / v along dim 0.
            let qkv_name = format!("{lp}.self_attn.qkv_proj.weight");
            let q_proj_w = loader.load_row_slice_to_gpu(&qkv_name, 0, q_rows, ordinal)?;
            let k_proj_w = loader.load_row_slice_to_gpu(&qkv_name, q_rows, k_rows, ordinal)?;
            let v_proj_w = loader.load_row_slice_to_gpu(
                &qkv_name,
                q_rows + k_rows,
                v_rows,
                ordinal,
            )?;

            let o_proj_w = loader.load_to_gpu(&format!("{lp}.self_attn.o_proj.weight"), ordinal)?;

            // Split fused gate_up_proj → gate / up along dim 0.
            let gate_up_name = format!("{lp}.mlp.gate_up_proj.weight");
            let gate_proj_w = loader.load_row_slice_to_gpu(&gate_up_name, 0, intermediate, ordinal)?;
            let up_proj_w = loader.load_row_slice_to_gpu(
                &gate_up_name,
                intermediate,
                intermediate,
                ordinal,
            )?;
            let down_proj_w = loader.load_to_gpu(&format!("{lp}.mlp.down_proj.weight"), ordinal)?;

            layers.push(Phi4LayerWeights {
                input_norm_w,
                post_attn_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                o_proj_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                q_proj_int4_scale: None, q_proj_int4_zero: None,
                k_proj_int4_scale: None, k_proj_int4_zero: None,
                v_proj_int4_scale: None, v_proj_int4_zero: None,
                o_proj_int4_scale: None, o_proj_int4_zero: None,
                gate_proj_int4_scale: None, gate_proj_int4_zero: None,
                up_proj_int4_scale: None, up_proj_int4_zero: None,
                down_proj_int4_scale: None, down_proj_int4_zero: None,
            });
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            norm_weight,
            layers,
            is_int4: false,
            int4_group_size: 0,
        })
    }

    /// Load from a baked SuperSonic package. Relies on the Phi-4 baker having
    /// already split fused tensors into q_proj/k_proj/v_proj/gate_proj/up_proj.
    pub fn load_baked(
        store: &model_store::BakedStore,
        config: &Phi4Config,
        ordinal: usize,
        weight_prefix: &str,
    ) -> Result<Self, model_store::Error> {
        let prefix = weight_prefix;

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
        let lm_head = if store.contains("lm_head.weight") {
            Arc::new(store.load_to_gpu("lm_head.weight", ordinal)?)
        } else {
            embed_tokens.clone()
        };
        let norm_weight = store.load_to_gpu(&format!("{prefix}.norm.weight"), ordinal)?;

        let mut is_int4 = false;
        let mut int4_group_size: usize = 0;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{idx}");
            let input_norm_w =
                store.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
            let post_attn_norm_w =
                store.load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)?;

            let q_name = format!("{lp}.self_attn.q_proj.weight");
            let k_name = format!("{lp}.self_attn.k_proj.weight");
            let v_name = format!("{lp}.self_attn.v_proj.weight");
            let o_name = format!("{lp}.self_attn.o_proj.weight");
            let gate_name = format!("{lp}.mlp.gate_proj.weight");
            let up_name = format!("{lp}.mlp.up_proj.weight");
            let down_name = format!("{lp}.mlp.down_proj.weight");

            let q_proj_w = store.load_to_gpu(&q_name, ordinal)?;
            let k_proj_w = store.load_to_gpu(&k_name, ordinal)?;
            let v_proj_w = store.load_to_gpu(&v_name, ordinal)?;
            let o_proj_w = store.load_to_gpu(&o_name, ordinal)?;
            let gate_proj_w = store.load_to_gpu(&gate_name, ordinal)?;
            let up_proj_w = store.load_to_gpu(&up_name, ordinal)?;
            let down_proj_w = store.load_to_gpu(&down_name, ordinal)?;

            let (q_i4s, q_i4z) = load_int4(&q_name)?;
            let (k_i4s, k_i4z) = load_int4(&k_name)?;
            let (v_i4s, v_i4z) = load_int4(&v_name)?;
            let (o_i4s, o_i4z) = load_int4(&o_name)?;
            let (gate_i4s, gate_i4z) = load_int4(&gate_name)?;
            let (up_i4s, up_i4z) = load_int4(&up_name)?;
            let (down_i4s, down_i4z) = load_int4(&down_name)?;

            if !is_int4 {
                if let Some(ref i4_scale) = q_i4s {
                    is_int4 = true;
                    let s_shape = i4_scale.shape();
                    let packed_cols = q_proj_w.shape()[1];
                    let original_cols = packed_cols * 2;
                    int4_group_size = if s_shape.len() == 2 && s_shape[1] > 0 {
                        original_cols / s_shape[1]
                    } else {
                        128
                    };
                }
            }

            layers.push(Phi4LayerWeights {
                input_norm_w,
                post_attn_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                o_proj_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                q_proj_int4_scale: q_i4s, q_proj_int4_zero: q_i4z,
                k_proj_int4_scale: k_i4s, k_proj_int4_zero: k_i4z,
                v_proj_int4_scale: v_i4s, v_proj_int4_zero: v_i4z,
                o_proj_int4_scale: o_i4s, o_proj_int4_zero: o_i4z,
                gate_proj_int4_scale: gate_i4s, gate_proj_int4_zero: gate_i4z,
                up_proj_int4_scale: up_i4s, up_proj_int4_zero: up_i4z,
                down_proj_int4_scale: down_i4s, down_proj_int4_zero: down_i4z,
            });
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            norm_weight,
            layers,
            is_int4,
            int4_group_size,
        })
    }
}
