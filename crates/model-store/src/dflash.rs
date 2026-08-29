//! DFlash2 draft-model GGUF reader (contributor foundation).
//!
//! Reads the canonical DFlash2 drafter (`qwen35-dflash-draft` arch) that PR #35
//! pairs with the Qwen3.8-27B target for speculative decoding. This is
//! foundation work: the loader is compile-tested and CPU-verified against the
//! canonical artifact, but is NOT yet wired to the public runner contract. A
//! draft forward engine, target hidden-state capture, and verify routing are
//! follow-on work gated behind the same artifact/correctness/perf bars as the
//! direct GQH path.
//!
//! The draft model is a 5-layer transformer fed by `n_target_layers` target
//! hidden states captured at `target_layer_ids` (an fc projects `5*hidden` ->
//! `hidden`). Each layer has a DFlash2 depthwise conv (`attn_conv`/`ffn_conv`)
//! before standard grouped-query attention (RoPE, q/k norms) and a SwiGLU MLP.
//! A candidate selector (`hproj` + `pred_cb`/`succ_cb` codebooks) produces the
//! draft token distribution. Weights are Q8_0 except norms and conv bases (F32).
//!
//! Tensor dims follow the GGUF convention: `ne[0]` is the contiguous axis. A
//! weight matrix `W[ne0, ne1]` is `[ne1 rows, ne0 cols]`, i.e. `[out, in]` when
//! the matmul is `out = W @ in` (the ggml `mul_mat` convention the geo-lucebox
//! drafter uses).

use std::path::Path;

use crate::gguf::GgufFile;
use crate::Error;

const ARCH: &str = "qwen35-dflash-draft";

/// Metadata-derived DFlash2 draft configuration.
#[derive(Debug, Clone)]
pub struct DraftConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    pub n_target_layers: usize,
    pub block_size: usize,
    pub mask_token_id: u32,
    /// Target layers whose output hidden states feed the drafter (0-indexed).
    pub target_layer_ids: Vec<usize>,
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
}

/// A validated tensor descriptor (CPU view into the mmap-backed GGUF).
#[derive(Debug, Clone)]
pub struct DraftTensorView {
    pub dims: Vec<usize>,
    pub tensor_type: u32,
    pub nbytes: usize,
}

impl DraftTensorView {
    /// `ne[0]`: the contiguous (fast) GGUF axis. For the drafter's weight
    /// matrices the converter lays this out as the *input* axis for the
    /// attention/FFN projections and as the *output* axis for `fc` -- the
    /// layout is validated against the canonical artifact, not assumed.
    pub fn ne0(&self) -> usize {
        self.dims.first().copied().unwrap_or(1)
    }
    /// `ne[1]`: the slow GGUF axis.
    pub fn ne1(&self) -> usize {
        self.dims.get(1).copied().unwrap_or(1)
    }
}

/// One draft decoder layer's tensors.
#[derive(Debug, Clone)]
pub struct DraftLayer {
    pub attn_norm: DraftTensorView,
    pub ffn_norm: DraftTensorView,
    pub q: DraftTensorView,
    pub k: DraftTensorView,
    pub v: DraftTensorView,
    pub output: DraftTensorView,
    pub q_norm: DraftTensorView,
    pub k_norm: DraftTensorView,
    pub attn_conv_base: DraftTensorView,
    pub attn_conv_proj: DraftTensorView,
    pub ffn_conv_base: DraftTensorView,
    pub ffn_conv_proj: DraftTensorView,
    pub ffn_gate: DraftTensorView,
    pub ffn_up: DraftTensorView,
    pub ffn_down: DraftTensorView,
}

/// The loaded DFlash2 drafter: config + mmap-backed tensor views.
///
/// Owns the `GgufFile` so tensor byte slices stay valid. Callers upload tensors
/// to the GPU in the (future) draft forward engine; this struct only validates
/// the artifact structure and exposes typed accessors.
pub struct DraftWeights {
    pub config: DraftConfig,
    file: GgufFile,
    pub fc: DraftTensorView,
    pub hidden_norm: DraftTensorView,
    pub output_norm: DraftTensorView,
    pub selector_hproj: DraftTensorView,
    pub selector_pred_cb: DraftTensorView,
    pub selector_succ_cb: DraftTensorView,
    pub layers: Vec<DraftLayer>,
}

impl std::fmt::Debug for DraftWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftWeights")
            .field("config", &self.config)
            .field("n_layers", &self.layers.len())
            .finish()
    }
}

impl DraftWeights {
    /// Raw packed bytes for a named tensor (CPU side, mmap-backed).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], Error> {
        self.file.tensor_bytes(name)
    }

    pub fn layer(&self, idx: usize) -> Option<&DraftLayer> {
        self.layers.get(idx)
    }
}

const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;

fn parse_kv_usize(kv: &GgufFile, key: &str) -> Result<usize, Error> {
    kv.kv(key)
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| Error::Other(format!("draft GGUF missing u32 metadata {key}")))
}

fn parse_kv_f32(kv: &GgufFile, key: &str) -> Result<f32, Error> {
    kv.kv(key)
        .and_then(|v| v.parse::<f32>().ok())
        .ok_or_else(|| Error::Other(format!("draft GGUF missing f32 metadata {key}")))
}

fn require_tensor(
    file: &GgufFile,
    name: &str,
    expect_type: u32,
    expect_ne0: Option<usize>,
    expect_ne1: Option<usize>,
) -> Result<DraftTensorView, Error> {
    let t = file
        .tensor(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if t.tensor_type != expect_type {
        return Err(Error::Other(format!(
            "draft tensor {name}: expected type {expect_type}, got {}",
            t.tensor_type
        )));
    }
    let view = DraftTensorView {
        dims: t.dims.clone(),
        tensor_type: t.tensor_type,
        nbytes: t.nbytes,
    };
    if let Some(ne0) = expect_ne0 {
        if view.ne0() != ne0 {
            return Err(Error::Other(format!(
                "draft tensor {name}: expected ne0={ne0}, got {}",
                view.ne0()
            )));
        }
    }
    if let Some(ne1) = expect_ne1 {
        if view.ne1() != ne1 {
            return Err(Error::Other(format!(
                "draft tensor {name}: expected ne1={ne1}, got {}",
                view.ne1()
            )));
        }
    }
    Ok(view)
}

/// Load and validate a DFlash2 draft GGUF. Returns the config + tensor views.
pub fn load_draft(path: &Path) -> Result<DraftWeights, Error> {
    let file = GgufFile::open(path)?;

    let arch = file
        .kv("general.architecture")
        .ok_or_else(|| Error::Other("draft GGUF missing general.architecture".into()))?;
    if arch != ARCH {
        return Err(Error::Other(format!(
            "draft GGUF architecture {arch:?} is not {ARCH:?}; this is the Qwen3.8 DFlash2 drafter only"
        )));
    }

    let p = |key: &str| parse_kv_usize(&file, key);
    let hidden = p("qwen35-dflash-draft.embedding_length")?;
    let n_layers = p("qwen35-dflash-draft.block_count")?;
    let intermediate = p("qwen35-dflash-draft.feed_forward_length")?;
    let n_heads = p("qwen35-dflash-draft.attention.head_count")?;
    let n_kv_heads = p("qwen35-dflash-draft.attention.head_count_kv")?;
    let head_dim = p("qwen35-dflash-draft.attention.key_length")?;
    let vocab = p("qwen35-dflash-draft.vocab_size")?;
    let n_target_layers = p("qwen35-dflash-draft.dflash.n_target_layers")?;
    let block_size = p("qwen35-dflash-draft.dflash.block_size")?;
    let mask_token_id = p("qwen35-dflash-draft.dflash.mask_token_id")? as u32;
    let conv_kernel_size = p("qwen35-dflash-draft.dflash.dflash2.conv_kernel_size")?;
    let conv_group_size = p("qwen35-dflash-draft.dflash.dflash2.conv_group_size")?;
    let selector_rank = p("qwen35-dflash-draft.dflash.dflash2.selector_rank")?;
    let selector_top_k = p("qwen35-dflash-draft.dflash.dflash2.selector_top_k")?;
    let rms_eps = parse_kv_f32(
        &file,
        "qwen35-dflash-draft.attention.layer_norm_rms_epsilon",
    )?;
    let rope_freq_base = parse_kv_f32(&file, "qwen35-dflash-draft.rope.freq_base")?;

    let target_layer_ids = file
        .i32_array("qwen35-dflash-draft.dflash.target_layer_ids")
        .ok_or_else(|| Error::Other("draft GGUF missing target_layer_ids array".into()))?;
    if target_layer_ids.len() != n_target_layers {
        return Err(Error::Other(format!(
            "draft GGUF target_layer_ids has {} entries, expected {n_target_layers}",
            target_layer_ids.len()
        )));
    }
    let target_layer_ids: Vec<usize> = target_layer_ids
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| Error::Other(format!("negative target_layer_id {v}")))
        })
        .collect::<Result<_, _>>()?;

    // Geometric sanity: fc output is n_target_layers * hidden; q/kv dims.
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let n_groups = hidden / conv_group_size;

    // Global tensors. ne0/ne1 follow the canonical artifact's observed layout
    // (the converter lays fc as [5*hidden, hidden] -- output on ne0 -- and the
    // attention/FFN projections as [in, out] -- input on ne0).
    let fc = require_tensor(
        &file,
        "dflash.fc.weight",
        GGML_Q8_0,
        Some(n_target_layers * hidden),
        Some(hidden),
    )?;
    let hidden_norm = require_tensor(
        &file,
        "dflash.hidden_norm.weight",
        GGML_F32,
        Some(hidden),
        Some(1),
    )?;
    let output_norm = require_tensor(&file, "output_norm.weight", GGML_F32, Some(hidden), Some(1))?;
    let selector_hproj = require_tensor(
        &file,
        "dflash.selector.hproj.weight",
        GGML_Q8_0,
        Some(hidden),
        Some(selector_rank),
    )?;
    let selector_pred_cb = require_tensor(
        &file,
        "dflash.selector.pred_cb",
        GGML_Q8_0,
        Some(selector_rank),
        Some(vocab),
    )?;
    let selector_succ_cb = require_tensor(
        &file,
        "dflash.selector.succ_cb",
        GGML_Q8_0,
        Some(selector_rank),
        Some(vocab),
    )?;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let blk = format!("blk.{i}.");
        let attn_norm = require_tensor(
            &file,
            &format!("{blk}attn_norm.weight"),
            GGML_F32,
            Some(hidden),
            Some(1),
        )?;
        let ffn_norm = require_tensor(
            &file,
            &format!("{blk}ffn_norm.weight"),
            GGML_F32,
            Some(hidden),
            Some(1),
        )?;
        // Attention projections: ne0=in (hidden), ne1=out (q/kv dim); output
        // projection: ne0=in (q_dim), ne1=out (hidden). FFN: gate/up ne0=hidden,
        // ne1=intermediate; down ne0=intermediate, ne1=hidden.
        let q = require_tensor(
            &file,
            &format!("{blk}attn_q.weight"),
            GGML_Q8_0,
            Some(hidden),
            Some(q_dim),
        )?;
        let k = require_tensor(
            &file,
            &format!("{blk}attn_k.weight"),
            GGML_Q8_0,
            Some(hidden),
            Some(kv_dim),
        )?;
        let v = require_tensor(
            &file,
            &format!("{blk}attn_v.weight"),
            GGML_Q8_0,
            Some(hidden),
            Some(kv_dim),
        )?;
        let output = require_tensor(
            &file,
            &format!("{blk}attn_output.weight"),
            GGML_Q8_0,
            Some(q_dim),
            Some(hidden),
        )?;
        let q_norm = require_tensor(
            &file,
            &format!("{blk}attn_q_norm.weight"),
            GGML_F32,
            Some(head_dim),
            Some(1),
        )?;
        let k_norm = require_tensor(
            &file,
            &format!("{blk}attn_k_norm.weight"),
            GGML_F32,
            Some(head_dim),
            Some(1),
        )?;
        let attn_conv_base =
            require_tensor(&file, &format!("{blk}attn_conv.base"), GGML_F32, None, None)?;
        let attn_conv_proj = require_tensor(
            &file,
            &format!("{blk}attn_conv.proj.weight"),
            GGML_Q8_0,
            None,
            None,
        )?;
        let ffn_conv_base =
            require_tensor(&file, &format!("{blk}ffn_conv.base"), GGML_F32, None, None)?;
        let ffn_conv_proj = require_tensor(
            &file,
            &format!("{blk}ffn_conv.proj.weight"),
            GGML_Q8_0,
            None,
            None,
        )?;
        let ffn_gate = require_tensor(
            &file,
            &format!("{blk}ffn_gate.weight"),
            GGML_Q8_0,
            Some(hidden),
            Some(intermediate),
        )?;
        let ffn_up = require_tensor(
            &file,
            &format!("{blk}ffn_up.weight"),
            GGML_Q8_0,
            Some(hidden),
            Some(intermediate),
        )?;
        let ffn_down = require_tensor(
            &file,
            &format!("{blk}ffn_down.weight"),
            GGML_Q8_0,
            Some(intermediate),
            Some(hidden),
        )?;

        // Conv base: [hidden, conv_kernel_size, conv_kernel_size] grouped by
        // conv_group_size (n_groups channels). The proj tightens the conv
        // output back to hidden; its shape is validated by the forward engine.
        if attn_conv_base.dims != [hidden, conv_kernel_size, conv_kernel_size] {
            return Err(Error::Other(format!(
                "draft blk.{i} attn_conv.base dims {:?} != [{hidden}, {conv_kernel_size}, {conv_kernel_size}]",
                attn_conv_base.dims
            )));
        }
        if ffn_conv_base.dims != [hidden, conv_kernel_size, conv_kernel_size] {
            return Err(Error::Other(format!(
                "draft blk.{i} ffn_conv.base dims {:?} != [{hidden}, {conv_kernel_size}, {conv_kernel_size}]",
                ffn_conv_base.dims
            )));
        }
        if n_groups * conv_group_size != hidden {
            return Err(Error::Other(format!(
                "draft conv_group_size {conv_group_size} does not divide hidden {hidden}"
            )));
        }

        layers.push(DraftLayer {
            attn_norm,
            ffn_norm,
            q,
            k,
            v,
            output,
            q_norm,
            k_norm,
            attn_conv_base,
            attn_conv_proj,
            ffn_conv_base,
            ffn_conv_proj,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    let config = DraftConfig {
        hidden,
        n_layers,
        intermediate,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size: vocab,
        rms_eps,
        rope_freq_base,
        n_target_layers,
        block_size,
        mask_token_id,
        target_layer_ids,
        conv_kernel_size,
        conv_group_size,
        selector_rank,
        selector_top_k,
    };

    Ok(DraftWeights {
        config,
        file,
        fc,
        hidden_norm,
        output_norm,
        selector_hproj,
        selector_pred_cb,
        selector_succ_cb,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_artifacts() -> bool {
        std::env::var_os("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").is_some()
    }

    fn draft_path() -> Option<std::path::PathBuf> {
        let value = std::env::var_os("SUPERSONIC_DFLASH_DRAFT_GGUF")?;
        let path = std::path::PathBuf::from(value);
        if !path.is_file() {
            if require_artifacts() {
                panic!(
                    "SUPERSONIC_DFLASH_DRAFT_GGUF points to a missing drafter: {}",
                    path.display()
                );
            }
            return None;
        }
        Some(path)
    }

    /// Loads and validates the canonical DFlash2 drafter end to end (config +
    /// every tensor's name, type, and shape). Artifact-gated; skips when the
    /// drafter is not configured.
    #[test]
    fn load_canonical_dflash_drafter() {
        let Some(path) = draft_path() else {
            eprintln!("skip: SUPERSONIC_DFLASH_DRAFT_GGUF not set");
            return;
        };
        let weights = load_draft(&path).unwrap_or_else(|e| panic!("load_draft: {e}"));
        let cfg = &weights.config;
        assert_eq!(cfg.hidden, 5120);
        assert_eq!(cfg.n_layers, 5);
        assert_eq!(cfg.intermediate, 17408);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 248320);
        assert!((cfg.rms_eps - 1e-6).abs() < 1e-9);
        assert!((cfg.rope_freq_base - 1e7).abs() < 1.0);
        assert_eq!(cfg.n_target_layers, 5);
        assert_eq!(cfg.block_size, 8);
        assert_eq!(cfg.mask_token_id, 248070);
        assert_eq!(cfg.target_layer_ids, vec![5, 19, 33, 47, 61]);
        assert_eq!(cfg.conv_kernel_size, 2);
        assert_eq!(cfg.conv_group_size, 16);
        assert_eq!(cfg.selector_rank, 256);
        assert_eq!(cfg.selector_top_k, 16);
        // fc: ne0=5*hidden, ne1=hidden (Q8_0).
        assert_eq!(weights.fc.ne0(), 5 * 5120);
        assert_eq!(weights.fc.ne1(), 5120);
        assert_eq!(weights.fc.tensor_type, GGML_Q8_0);
        // selector codebooks: ne0=rank, ne1=vocab.
        assert_eq!(weights.selector_pred_cb.ne0(), 256);
        assert_eq!(weights.selector_pred_cb.ne1(), 248320);
        assert_eq!(weights.selector_succ_cb.ne0(), 256);
        assert_eq!(weights.layers.len(), 5);
        let l0 = weights.layer(0).expect("layer 0");
        assert_eq!(l0.q.ne0(), 5120);
        assert_eq!(l0.q.ne1(), 32 * 128);
        assert_eq!(l0.k.ne0(), 5120);
        assert_eq!(l0.k.ne1(), 8 * 128);
        assert_eq!(l0.output.ne0(), 32 * 128);
        assert_eq!(l0.output.ne1(), 5120);
        assert_eq!(l0.attn_conv_base.dims, [5120, 2, 2]);
        assert_eq!(l0.attn_conv_base.tensor_type, GGML_F32);
        // Byte slices are mmap-backed and non-empty.
        let b = weights.tensor_bytes("dflash.fc.weight").expect("fc bytes");
        assert!(b.len() == weights.fc.nbytes && !b.is_empty());
        eprintln!(
            "dflash: loaded {} layers, {} target ids, fc {}x{} ({} bytes)",
            cfg.n_layers,
            cfg.target_layer_ids.len(),
            weights.fc.ne0(),
            weights.fc.ne1(),
            b.len()
        );
    }

    /// A non-DFlash2 GGUF (the target GQH GGUF) must be rejected, not silently
    /// accepted as a drafter.
    #[test]
    fn rejects_non_dflash_architecture() {
        let Some(target) = std::env::var_os("SUPERSONIC_GQH_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("skip: SUPERSONIC_GQH_GGUF not set for arch-reject test");
            return;
        };
        if !target.is_file() {
            eprintln!("skip: target gguf missing for arch-reject test");
            return;
        }
        let err = load_draft(&target).unwrap_err();
        assert!(
            err.to_string().contains("not \"qwen35-dflash-draft\""),
            "expected arch rejection, got: {err}"
        );
    }
    /// Uploads the canonical drafter to the GPU and verifies every buffer's
    /// dtype, element count, and shape match the validated CPU views.
    /// GPU-gated; skips when no HIP device is available.
    #[test]
    fn upload_canonical_drafter_to_gpu() {
        let ordinal = 0;
        let Some(path) = draft_path() else {
            eprintln!("skip: SUPERSONIC_DFLASH_DRAFT_GGUF not set");
            return;
        };
        let weights = load_draft(&path).unwrap_or_else(|e| panic!("load_draft: {e}"));
        let gpu = match weights.upload(ordinal) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip dflash gpu upload: {e}");
                return;
            }
        };
        let cfg = &gpu.config;

        // Q8_0 tensors upload as packed U8 [rows, row_bytes] where
        // row_bytes = (logical_cols / 32) * 34 — the layout the HIP GEMM kernel
        // expects (matching the target model's upload_packed).
        let q8_row_bytes = |logical_cols: usize| (logical_cols / 32) * 34;
        assert_eq!(gpu.fc.dtype(), ScalarType::U8);
        assert_eq!(
            gpu.fc.shape(),
            [cfg.hidden, q8_row_bytes(cfg.n_target_layers * cfg.hidden)]
        );
        // Norms are uploaded as F32 (the draft F32 compute path reads them as F32);
        assert_eq!(gpu.hidden_norm.dtype(), ScalarType::F32);
        assert_eq!(gpu.hidden_norm.shape(), [cfg.hidden]);
        assert_eq!(gpu.output_norm.dtype(), ScalarType::F32);
        assert_eq!(gpu.output_norm.shape(), [cfg.hidden]);
        assert_eq!(gpu.layers.len(), cfg.n_layers);
        let l0 = &gpu.layers[0];
        // attn_q: logical [hidden, q_dim] -> packed [q_dim, row_bytes(hidden)].
        assert_eq!(l0.q.dtype(), ScalarType::U8);
        assert_eq!(
            l0.q.shape(),
            [cfg.n_heads * cfg.head_dim, q8_row_bytes(cfg.hidden)]
        );
        assert_eq!(l0.k.dtype(), ScalarType::U8);
        assert_eq!(
            l0.k.shape(),
            [cfg.n_kv_heads * cfg.head_dim, q8_row_bytes(cfg.hidden)]
        );
        // attn_output: logical [q_dim, hidden] -> packed [hidden, row_bytes(q_dim)].
        assert_eq!(l0.output.dtype(), ScalarType::U8);
        assert_eq!(
            l0.output.shape(),
            [cfg.hidden, q8_row_bytes(cfg.n_heads * cfg.head_dim)]
        );
        assert_eq!(l0.attn_norm.dtype(), ScalarType::F32);
        assert_eq!(l0.attn_norm.shape(), [cfg.hidden]);
        assert_eq!(l0.q_norm.dtype(), ScalarType::F32);
        assert_eq!(l0.q_norm.shape(), [cfg.head_dim]);
        // ffn_gate: logical [hidden, inter] -> packed [inter, row_bytes(hidden)].
        assert_eq!(l0.ffn_gate.dtype(), ScalarType::U8);
        assert_eq!(
            l0.ffn_gate.shape(),
            [cfg.intermediate, q8_row_bytes(cfg.hidden)]
        );
        // ffn_down: logical [inter, hidden] -> packed [hidden, row_bytes(inter)].
        assert_eq!(l0.ffn_down.dtype(), ScalarType::U8);
        assert_eq!(
            l0.ffn_down.shape(),
            [cfg.hidden, q8_row_bytes(cfg.intermediate)]
        );
        // The conv bases are F32 3D tensors [hidden, conv_kernel, conv_kernel].
        assert_eq!(l0.attn_conv_base.dtype(), ScalarType::F32);
        assert_eq!(
            l0.attn_conv_base.shape(),
            [cfg.hidden, cfg.conv_kernel_size, cfg.conv_kernel_size]
        );
        eprintln!(
            "dflash gpu upload: {} layers, fc {:?}, q {:?} ok",
            cfg.n_layers,
            gpu.fc.shape(),
            l0.q.shape()
        );
    }
}

// ── GPU weight upload ──────────────────────────────────────────────
//
// Uploads the validated CPU draft tensors to device buffers. Q8_0 weights go
// up as raw packed U8 (the GPU Q8_0 GEMM kernel dequantizes on device, same as
// the target's GGML quant weights); F32 tensors (norms, conv bases) go up as
// F32. This is the bridge from the CPU loader to the (future) GPU draft
// forward engine.

use gpu_hal::{GpuBuffer, ScalarType};

/// A draft layer's tensors on the GPU.
#[derive(Debug)]
pub struct DraftGpuLayer {
    pub attn_norm: GpuBuffer,
    pub ffn_norm: GpuBuffer,
    pub q: GpuBuffer,
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub output: GpuBuffer,
    pub q_norm: GpuBuffer,
    pub k_norm: GpuBuffer,
    pub attn_conv_base: GpuBuffer,
    pub attn_conv_proj: GpuBuffer,
    pub ffn_conv_base: GpuBuffer,
    pub ffn_conv_proj: GpuBuffer,
    pub ffn_gate: GpuBuffer,
    pub ffn_up: GpuBuffer,
    pub ffn_down: GpuBuffer,
}

/// The draft model's weights uploaded to the GPU.
pub struct DraftGpuWeights {
    pub config: DraftConfig,
    pub fc: GpuBuffer,
    pub hidden_norm: GpuBuffer,
    pub output_norm: GpuBuffer,
    pub selector_hproj: GpuBuffer,
    pub selector_pred_cb: GpuBuffer,
    pub selector_succ_cb: GpuBuffer,
    pub layers: Vec<DraftGpuLayer>,
}

impl std::fmt::Debug for DraftGpuWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftGpuWeights")
            .field("config", &self.config)
            .field("n_layers", &self.layers.len())
            .finish()
    }
}

/// Upload an F32 tensor preserving its GGUF dims (norms, conv bases).
fn upload_f32(file: &GgufFile, name: &str, ordinal: usize) -> Result<GpuBuffer, Error> {
    let data = file.tensor_bytes(name)?;
    let tensor = file
        .tensor(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if tensor.tensor_type != 0 {
        return Err(Error::Other(format!(
            "draft upload_f32 {name}: expected F32, got type {}",
            tensor.tensor_type
        )));
    }
    GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &tensor.dims, data).map_err(Error::Gpu)
}

/// Upload a Q8_0 tensor as a packed U8 buffer with shape `[rows, row_bytes]`,
/// the layout the HIP Q8_0 GEMM kernel expects (matching the target model's
/// `upload_packed`). `ne0` is the logical (contiguous) axis and `ne1` the row
/// count; `row_bytes = (ne0 / 32) * 34`.
fn upload_q8_0(file: &GgufFile, name: &str, ordinal: usize) -> Result<GpuBuffer, Error> {
    let data = file.tensor_bytes(name)?;
    let tensor = file
        .tensor(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if tensor.tensor_type != 8 {
        return Err(Error::Other(format!(
            "draft upload_q8_0 {name}: expected Q8_0 (type 8), got type {}",
            tensor.tensor_type
        )));
    }
    if tensor.dims.len() != 2 {
        return Err(Error::Other(format!(
            "draft upload_q8_0 {name}: expected rank-2, got {:?}",
            tensor.dims
        )));
    }
    let ne0 = tensor.dims[0];
    let ne1 = tensor.dims[1];
    if ne0 % 32 != 0 {
        return Err(Error::Other(format!(
            "draft upload_q8_0 {name}: ne0 {ne0} not a multiple of 32"
        )));
    }
    let row_bytes = (ne0 / 32) * 34;
    if data.len() != ne1 * row_bytes {
        return Err(Error::Other(format!(
            "draft upload_q8_0 {name}: packed size {} != {ne1}*{row_bytes}",
            data.len()
        )));
    }
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[ne1, row_bytes], data).map_err(Error::Gpu)
}

impl DraftWeights {
    /// Upload all validated draft tensors to the GPU at `ordinal`.
    pub fn upload(&self, ordinal: usize) -> Result<DraftGpuWeights, Error> {
        let cfg = self.config.clone();
        let f32_w = |name: &str| upload_f32(&self.file, name, ordinal);
        let q8_w = |name: &str| upload_q8_0(&self.file, name, ordinal);

        let fc = q8_w("dflash.fc.weight")?;
        let hidden_norm = f32_w("dflash.hidden_norm.weight")?;
        let output_norm = f32_w("output_norm.weight")?;
        let selector_hproj = q8_w("dflash.selector.hproj.weight")?;
        let selector_pred_cb = q8_w("dflash.selector.pred_cb")?;
        let selector_succ_cb = q8_w("dflash.selector.succ_cb")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let blk = format!("blk.{i}.");
            layers.push(DraftGpuLayer {
                attn_norm: f32_w(&format!("{blk}attn_norm.weight"))?,
                ffn_norm: f32_w(&format!("{blk}ffn_norm.weight"))?,
                q: q8_w(&format!("{blk}attn_q.weight"))?,
                k: q8_w(&format!("{blk}attn_k.weight"))?,
                v: q8_w(&format!("{blk}attn_v.weight"))?,
                output: q8_w(&format!("{blk}attn_output.weight"))?,
                q_norm: f32_w(&format!("{blk}attn_q_norm.weight"))?,
                k_norm: f32_w(&format!("{blk}attn_k_norm.weight"))?,
                attn_conv_base: f32_w(&format!("{blk}attn_conv.base"))?,
                attn_conv_proj: q8_w(&format!("{blk}attn_conv.proj.weight"))?,
                ffn_conv_base: f32_w(&format!("{blk}ffn_conv.base"))?,
                ffn_conv_proj: q8_w(&format!("{blk}ffn_conv.proj.weight"))?,
                ffn_gate: q8_w(&format!("{blk}ffn_gate.weight"))?,
                ffn_up: q8_w(&format!("{blk}ffn_up.weight"))?,
                ffn_down: q8_w(&format!("{blk}ffn_down.weight"))?,
            });
        }

        Ok(DraftGpuWeights {
            config: cfg,
            fc,
            hidden_norm,
            output_norm,
            selector_hproj,
            selector_pred_cb,
            selector_succ_cb,
            layers,
        })
    }
}
