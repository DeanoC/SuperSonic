//! GPU-resident weights for the DFlash draft.
//!
//! The draft's 58 safetensors tensors map 1:1 to the structs below. What the
//! draft does NOT own (and must NOT try to load locally):
//!   - `embed_tokens` — borrowed from the target via `Arc::clone`
//!   - `lm_head` — borrowed from the target via `Arc::clone`
//!
//! See `docs/dflash.md` §7 for the rationale and the canonical proof that
//! these tensors don't exist in the draft checkpoint.
//!
//! Injection formula note: there are NO `k_inject` / `v_inject` projections.
//! The per-layer `k_proj` / `v_proj` are applied to BOTH the draft hidden
//! states AND the fused target-tap vector, then concatenated along the
//! sequence axis inside attention. See `docs/dflash.md` §4.

use std::path::Path;
use std::sync::Arc;

use gpu_hal::{GpuBuffer, ScalarType};

use crate::config::DFlashConfig;
use crate::loader::{GgufWeightLoader, LoadError, WeightLoader};

pub struct LinearWeight {
    pub weight: GpuBuffer,
    pub quant_type: i32,
    pub logical_rows: usize,
    pub logical_cols: usize,
}

impl LinearWeight {
    pub fn from_buffer(weight: GpuBuffer) -> Result<Self, LoadError> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(LoadError::UnexpectedTensor(format!(
                "draft linear weight must be rank-2, got shape {shape:?}"
            )));
        }
        let logical_rows = shape[0];
        let logical_cols = shape[1];
        Ok(Self {
            weight,
            quant_type: 0,
            logical_rows,
            logical_cols,
        })
    }

    pub fn from_parts(
        weight: GpuBuffer,
        quant_type: i32,
        logical_rows: usize,
        logical_cols: usize,
    ) -> Self {
        Self {
            weight,
            quant_type,
            logical_rows,
            logical_cols,
        }
    }

    pub fn is_lowbit(&self) -> bool {
        self.quant_type != 0
    }
}

pub struct DFlashLayerWeights {
    // RMSNorm weights (Qwen3: no add_unit_offset; weight is used as-is).
    pub input_norm_w: GpuBuffer,
    pub post_attn_norm_w: GpuBuffer,

    // Attention projections. Safetensors are BF16; GGUF may use GGML low-bit
    // row blocks while keeping the same logical shapes.
    //   q_proj: [q_out_dim=4096, hidden=4096]
    //   k_proj: [kv_out_dim=1024, hidden=4096]
    //   v_proj: [kv_out_dim=1024, hidden=4096]
    //   o_proj: [hidden=4096, q_out_dim=4096]
    pub q_proj_w: LinearWeight,
    pub k_proj_w: LinearWeight,
    pub v_proj_w: LinearWeight,
    pub kv_proj_w: LinearWeight,
    pub o_proj_w: LinearWeight,

    // Per-head RMSNorm over head_dim (NOT hidden_size). Shape [head_dim].
    pub q_norm_w: GpuBuffer,
    pub k_norm_w: GpuBuffer,

    // SwiGLU MLP (BF16). Shapes:
    //   gate_proj: [intermediate=12288, hidden=4096]
    //   up_proj:   [intermediate=12288, hidden=4096]
    //   down_proj: [hidden=4096, intermediate=12288]
    pub gate_proj_w: LinearWeight,
    pub up_proj_w: LinearWeight,
    pub down_proj_w: LinearWeight,
}

pub struct DFlashWeights {
    pub config: DFlashConfig,

    // Arc-shared with target — NOT owned by the draft checkpoint.
    pub embed_tokens: Arc<GpuBuffer>,
    pub lm_head: Arc<GpuBuffer>,

    // Tap fuser (runs once per decode round):
    //   fc: [hidden=4096, num_taps*hidden=20480]  (no bias)
    //   hidden_norm: [hidden=4096]  (RMSNorm weight)
    pub fc_w: LinearWeight,
    pub hidden_norm_w: GpuBuffer,

    // Final RMSNorm (applied to last-layer output before lm_head).
    pub norm_w: GpuBuffer,

    pub layers: Vec<DFlashLayerWeights>,

    pub dummy_lowbit_scale: GpuBuffer,
}

impl DFlashWeights {
    /// Load the DFlash draft from a HuggingFace-style directory containing a
    /// single `model.safetensors`. The target's `embed_tokens` / `lm_head`
    /// must be supplied via `Arc::clone` — they are NOT in the draft file.
    pub fn load(
        model_dir: &Path,
        config: &DFlashConfig,
        ordinal: usize,
        embed_tokens: Arc<GpuBuffer>,
        lm_head: Arc<GpuBuffer>,
    ) -> Result<Self, LoadError> {
        if let Some(path) = std::env::var_os("SUPERSONIC_DFLASH_DRAFT_GGUF") {
            return Self::load_gguf(Path::new(&path), config, ordinal, embed_tokens, lm_head);
        }

        let loader = WeightLoader::from_dir(model_dir)?;

        if loader.contains("embed_tokens.weight") || loader.contains("model.embed_tokens.weight") {
            return Err(LoadError::UnexpectedTensor(
                "DFlash draft checkpoint unexpectedly contains embed_tokens — \
                 this crate shares embed_tokens with the target via Arc. \
                 Refusing to load a duplicate copy."
                    .into(),
            ));
        }
        if loader.contains("lm_head.weight") {
            return Err(LoadError::UnexpectedTensor(
                "DFlash draft checkpoint unexpectedly contains lm_head — \
                 this crate shares lm_head with the target via Arc. \
                 Refusing to load a duplicate copy."
                    .into(),
            ));
        }

        let fc_w = LinearWeight::from_buffer(loader.load_to_gpu("fc.weight", ordinal)?)?;
        let hidden_norm_w = loader.load_to_gpu("hidden_norm.weight", ordinal)?;
        let norm_w = loader.load_to_gpu("norm.weight", ordinal)?;
        let dummy_lowbit_scale =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[1, 1], &[0, 0])?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let lp = format!("layers.{idx}");
            let input_norm_w =
                loader.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
            let post_attn_norm_w =
                loader.load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)?;

            let q_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.self_attn.q_proj.weight"), ordinal)?,
            )?;
            let k_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.self_attn.k_proj.weight"), ordinal)?,
            )?;
            let v_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.self_attn.v_proj.weight"), ordinal)?,
            )?;
            let kv_proj_w = LinearWeight::from_buffer(loader.load_concat_dim0_to_gpu(
                &format!("{lp}.self_attn.k_proj.weight"),
                &format!("{lp}.self_attn.v_proj.weight"),
                ordinal,
            )?)?;
            let o_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.self_attn.o_proj.weight"), ordinal)?,
            )?;

            let q_norm_w = loader.load_to_gpu(&format!("{lp}.self_attn.q_norm.weight"), ordinal)?;
            let k_norm_w = loader.load_to_gpu(&format!("{lp}.self_attn.k_norm.weight"), ordinal)?;

            let gate_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.mlp.gate_proj.weight"), ordinal)?,
            )?;
            let up_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.mlp.up_proj.weight"), ordinal)?,
            )?;
            let down_proj_w = LinearWeight::from_buffer(
                loader.load_to_gpu(&format!("{lp}.mlp.down_proj.weight"), ordinal)?,
            )?;

            layers.push(DFlashLayerWeights {
                input_norm_w,
                post_attn_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                kv_proj_w,
                o_proj_w,
                q_norm_w,
                k_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
            });
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            fc_w,
            hidden_norm_w,
            norm_w,
            layers,
            dummy_lowbit_scale,
        })
    }

    fn load_gguf(
        path: &Path,
        config: &DFlashConfig,
        ordinal: usize,
        embed_tokens: Arc<GpuBuffer>,
        lm_head: Arc<GpuBuffer>,
    ) -> Result<Self, LoadError> {
        let loader = GgufWeightLoader::from_file(path)?;
        let fc_w = if loader.contains("dflash.fc.weight") {
            Self::linear_from_gguf(loader.load_linear_to_gpu("dflash.fc.weight", ordinal)?)
        } else {
            Self::linear_from_gguf(loader.load_linear_to_gpu("dflash_fc.weight", ordinal)?)
        };
        let hidden_norm_w = if loader.contains("dflash.hidden_norm.weight") {
            loader.load_norm_bf16_to_gpu("dflash.hidden_norm.weight", ordinal)?
        } else {
            loader.load_norm_bf16_to_gpu("dflash_hidden_norm.weight", ordinal)?
        };
        let norm_w = loader.load_norm_bf16_to_gpu("output_norm.weight", ordinal)?;
        let dummy_lowbit_scale =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[1, 1], &[0, 0])?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let input_norm_w =
                loader.load_norm_bf16_to_gpu(&format!("blk.{idx}.attn_norm.weight"), ordinal)?;
            let ffn_norm = format!("blk.{idx}.ffn_norm.weight");
            let post_attention_norm = format!("blk.{idx}.post_attention_norm.weight");
            let post_attn_norm_w = if loader.contains(&ffn_norm) {
                loader.load_norm_bf16_to_gpu(&ffn_norm, ordinal)?
            } else {
                loader.load_norm_bf16_to_gpu(&post_attention_norm, ordinal)?
            };

            let q_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.attn_q.weight"), ordinal)?,
            );
            let k_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.attn_k.weight"), ordinal)?,
            );
            let v_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.attn_v.weight"), ordinal)?,
            );
            let kv_proj_w = Self::linear_from_gguf(loader.load_concat_dim0_linear_to_gpu(
                &format!("blk.{idx}.attn_k.weight"),
                &format!("blk.{idx}.attn_v.weight"),
                ordinal,
            )?);
            let o_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.attn_output.weight"), ordinal)?,
            );

            let q_norm_w =
                loader.load_norm_bf16_to_gpu(&format!("blk.{idx}.attn_q_norm.weight"), ordinal)?;
            let k_norm_w =
                loader.load_norm_bf16_to_gpu(&format!("blk.{idx}.attn_k_norm.weight"), ordinal)?;

            let gate_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.ffn_gate.weight"), ordinal)?,
            );
            let up_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.ffn_up.weight"), ordinal)?,
            );
            let down_proj_w = Self::linear_from_gguf(
                loader.load_linear_to_gpu(&format!("blk.{idx}.ffn_down.weight"), ordinal)?,
            );

            layers.push(DFlashLayerWeights {
                input_norm_w,
                post_attn_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                kv_proj_w,
                o_proj_w,
                q_norm_w,
                k_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
            });
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            fc_w,
            hidden_norm_w,
            norm_w,
            layers,
            dummy_lowbit_scale,
        })
    }

    fn linear_from_gguf(parts: (GpuBuffer, i32, usize, usize)) -> LinearWeight {
        let (weight, quant_type, logical_rows, logical_cols) = parts;
        LinearWeight::from_parts(weight, quant_type, logical_rows, logical_cols)
    }
}
