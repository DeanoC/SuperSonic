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

use gpu_hal::GpuBuffer;

use crate::config::DFlashConfig;
use crate::loader::{LoadError, WeightLoader};

pub struct DFlashLayerWeights {
    // RMSNorm weights (Qwen3: no add_unit_offset; weight is used as-is).
    pub input_norm_w: GpuBuffer,
    pub post_attn_norm_w: GpuBuffer,

    // Attention projections (BF16). Shapes per Qwen3.5-9B draft:
    //   q_proj: [q_out_dim=4096, hidden=4096]
    //   k_proj: [kv_out_dim=1024, hidden=4096]
    //   v_proj: [kv_out_dim=1024, hidden=4096]
    //   o_proj: [hidden=4096, q_out_dim=4096]
    pub q_proj_w: GpuBuffer,
    pub k_proj_w: GpuBuffer,
    pub v_proj_w: GpuBuffer,
    pub o_proj_w: GpuBuffer,

    // Per-head RMSNorm over head_dim (NOT hidden_size). Shape [head_dim].
    pub q_norm_w: GpuBuffer,
    pub k_norm_w: GpuBuffer,

    // SwiGLU MLP (BF16). Shapes:
    //   gate_proj: [intermediate=12288, hidden=4096]
    //   up_proj:   [intermediate=12288, hidden=4096]
    //   down_proj: [hidden=4096, intermediate=12288]
    pub gate_proj_w: GpuBuffer,
    pub up_proj_w: GpuBuffer,
    pub down_proj_w: GpuBuffer,
}

pub struct DFlashWeights {
    pub config: DFlashConfig,

    // Arc-shared with target — NOT owned by the draft checkpoint.
    pub embed_tokens: Arc<GpuBuffer>,
    pub lm_head: Arc<GpuBuffer>,

    // Tap fuser (runs once per decode round):
    //   fc: [hidden=4096, num_taps*hidden=20480]  (no bias)
    //   hidden_norm: [hidden=4096]  (RMSNorm weight)
    pub fc_w: GpuBuffer,
    pub hidden_norm_w: GpuBuffer,

    // Final RMSNorm (applied to last-layer output before lm_head).
    pub norm_w: GpuBuffer,

    pub layers: Vec<DFlashLayerWeights>,
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

        let fc_w = loader.load_to_gpu("fc.weight", ordinal)?;
        let hidden_norm_w = loader.load_to_gpu("hidden_norm.weight", ordinal)?;
        let norm_w = loader.load_to_gpu("norm.weight", ordinal)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let lp = format!("layers.{idx}");
            let input_norm_w =
                loader.load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)?;
            let post_attn_norm_w = loader
                .load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)?;

            let q_proj_w = loader.load_to_gpu(&format!("{lp}.self_attn.q_proj.weight"), ordinal)?;
            let k_proj_w = loader.load_to_gpu(&format!("{lp}.self_attn.k_proj.weight"), ordinal)?;
            let v_proj_w = loader.load_to_gpu(&format!("{lp}.self_attn.v_proj.weight"), ordinal)?;
            let o_proj_w = loader.load_to_gpu(&format!("{lp}.self_attn.o_proj.weight"), ordinal)?;

            let q_norm_w = loader.load_to_gpu(&format!("{lp}.self_attn.q_norm.weight"), ordinal)?;
            let k_norm_w = loader.load_to_gpu(&format!("{lp}.self_attn.k_norm.weight"), ordinal)?;

            let gate_proj_w =
                loader.load_to_gpu(&format!("{lp}.mlp.gate_proj.weight"), ordinal)?;
            let up_proj_w = loader.load_to_gpu(&format!("{lp}.mlp.up_proj.weight"), ordinal)?;
            let down_proj_w =
                loader.load_to_gpu(&format!("{lp}.mlp.down_proj.weight"), ordinal)?;

            layers.push(DFlashLayerWeights {
                input_norm_w,
                post_attn_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
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
        })
    }
}
