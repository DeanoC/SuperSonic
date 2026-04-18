//! Fused-kernel greedy decode validator for Gemma 4 E2B — Step 14.
//!
//! Same oracle contract as `gemma4_decode_validate` but replaces each layer's
//! attention half (input_norm → Q/K/V proj → q/k/v norm → RoPE → kv_append
//! → SWA/full attention → o_proj → post_attn_norm → residual) with a single
//! `g4::fused_attn_block` kernel launch that orchestrates those sub-ops via
//! `grid_barrier` + work-stealing inside the kernel. MLP + PLE + final norm
//! + lm_head remain separate primitive launches — fusion of those follows
//! the same pattern.
//!
//! Usage:
//!   cargo run --release --bin gemma4_fused_decode_validate -- \
//!     --model-dir <checkpoint> --oracle-json <oracle_state.json> \
//!     [--max-new-tokens N]

use std::ffi::c_void;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use clap::Parser;
use ::gemma4::config::{self as g4_config, AttnKind, Config, TextConfig};
use ::gemma4::weight_spec as g4_spec;
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::gemma4 as g4;
use memmap2::Mmap;
use safetensors::SafeTensors;

#[path = "../oracle.rs"]
mod oracle;
use oracle::OracleOutput;

#[derive(Parser, Debug)]
#[command(about = "Validate Gemma 4 end-to-end greedy decode against the PyTorch oracle")]
struct Cli {
    /// Path to a local Gemma 4 checkpoint directory (config.json + safetensors).
    #[arg(long)]
    model_dir: PathBuf,
    /// Oracle JSON file produced by `oracle/gemma4_oracle.py --emit-state`.
    #[arg(long)]
    oracle_json: PathBuf,
    /// Number of decode steps. Step 0 reproduces the prefill last-token path;
    /// steps 1..=N-1 compare against `oracle.decode_logits`. Must not exceed
    /// `oracle.generated_tokens` (and the oracle must have been run with at
    /// least that many `--max-new-tokens`).
    #[arg(long, default_value_t = 4)]
    max_new_tokens: usize,
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

fn upload_bf16(shape: &[usize], host: &[f32]) -> Result<GpuBuffer> {
    let bytes = f32_to_bf16_bytes(host);
    Ok(GpuBuffer::from_host_bytes(0, ScalarType::BF16, shape, &bytes)?)
}

fn download_bf16(buf: &GpuBuffer) -> Result<Vec<f32>> {
    Ok(bf16_bytes_to_f32(&buf.to_host_bytes()?))
}

struct UnbakedLoader {
    shards: Vec<Mmap>,
    index: std::collections::BTreeMap<String, usize>,
}

impl UnbakedLoader {
    fn open(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let raw: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let weight_map = raw["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow!("weight_map missing in {}", index_path.display()))?;
            let mut shard_files: Vec<String> = Vec::new();
            let mut shard_idx_map: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for v in weight_map.values() {
                let filename = v.as_str().unwrap_or("").to_string();
                if !shard_idx_map.contains_key(&filename) {
                    shard_idx_map.insert(filename.clone(), shard_files.len());
                    shard_files.push(filename);
                }
            }
            let mut shards = Vec::with_capacity(shard_files.len());
            for filename in &shard_files {
                let path = dir.join(filename);
                let file = File::open(&path)
                    .with_context(|| format!("open shard {}", path.display()))?;
                shards.push(unsafe { Mmap::map(&file)? });
            }
            let mut index = std::collections::BTreeMap::new();
            for (tensor_name, filename) in weight_map {
                let filename = filename.as_str().unwrap_or("");
                if let Some(&shard_idx) = shard_idx_map.get(filename) {
                    index.insert(tensor_name.clone(), shard_idx);
                }
            }
            Ok(Self { shards, index })
        } else {
            let single = dir.join("model.safetensors");
            if !single.exists() {
                bail!("no safetensors found in {}", dir.display());
            }
            let file = File::open(&single)
                .with_context(|| format!("open {}", single.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };
            let st = SafeTensors::deserialize(&mmap)?;
            let mut index = std::collections::BTreeMap::new();
            for name in st.names() {
                index.insert(name.to_string(), 0);
            }
            Ok(Self { shards: vec![mmap], index })
        }
    }

    fn tensor_bytes<'a>(&'a self, name: &str) -> Result<(Vec<usize>, &'a [u8])> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let st = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = st.tensor(name)?;
        if view.dtype() != safetensors::Dtype::BF16 {
            bail!("tensor {name} is {:?}, expected BF16", view.dtype());
        }
        Ok((view.shape().to_vec(), view.data()))
    }

    fn load_bf16_to_gpu(&self, name: &str) -> Result<GpuBuffer> {
        let (shape, bytes) = self.tensor_bytes(name)?;
        Ok(GpuBuffer::from_host_bytes(0, ScalarType::BF16, &shape, bytes)?)
    }
}

fn build_rope_table_from_inv_freq(
    inv_freq: &[f32],
    head_dim: usize,
    max_pos: usize,
    attention_scaling: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim % 2 == 0);
    let half = head_dim / 2;
    assert_eq!(inv_freq.len(), half);

    let mut cos = vec![0.0f32; max_pos * head_dim];
    let mut sin = vec![0.0f32; max_pos * head_dim];
    for p in 0..max_pos {
        for i in 0..half {
            let theta = (p as f32) * inv_freq[i];
            let (s, c) = theta.sin_cos();
            let c = c * attention_scaling;
            let s = s * attention_scaling;
            cos[p * head_dim + i] = c;
            sin[p * head_dim + i] = s;
            cos[p * head_dim + i + half] = c;
            sin[p * head_dim + i + half] = s;
        }
    }
    (cos, sin)
}

fn build_sliding_rope_table(
    head_dim: usize,
    rope_theta: f64,
    max_pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2 * i) as f64 / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(exponent)) as f32);
    }
    build_rope_table_from_inv_freq(&inv_freq, head_dim, max_pos, 1.0)
}

fn build_proportional_rope_table(
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f64,
    max_pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let rope_angles = (partial_rotary_factor * (head_dim as f64) / 2.0) as usize;
    assert!(rope_angles <= half, "rope_angles {rope_angles} > head_dim/2 {half}");

    let mut inv_freq = vec![0.0f32; half];
    for j in 0..rope_angles {
        let exponent = (2 * j) as f64 / head_dim as f64;
        inv_freq[j] = (1.0 / rope_theta.powf(exponent)) as f32;
    }
    build_rope_table_from_inv_freq(&inv_freq, head_dim, max_pos, 1.0)
}

fn load_scaled_embed_row(
    loader: &UnbakedLoader,
    weight_name: &str,
    token_id: u32,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    let (shape, bytes) = loader.tensor_bytes(weight_name)?;
    if shape.len() != 2 || shape[1] != hidden_size {
        bail!(
            "{weight_name} shape {:?} does not match expected [vocab, {hidden_size}]",
            shape
        );
    }
    let vocab = shape[0];
    if (token_id as usize) >= vocab {
        bail!("token id {token_id} out of range (vocab={vocab})");
    }
    let row_bytes = hidden_size * 2;
    let off = token_id as usize * row_bytes;
    let slice = &bytes[off..off + row_bytes];
    let row = bf16_bytes_to_f32(slice);
    let scale_bf16 = bf16::from_f32((hidden_size as f32).sqrt());
    let scale = scale_bf16.to_f32();
    Ok(row.iter().map(|v| v * scale).collect())
}

/// Read one row from `embed_tokens_per_layer.weight` and apply the
/// `Gemma4TextScaledWordEmbedding` scale for the PLE table: `sqrt(ple_hidden)`.
/// For Gemma 4 E2B `ple_hidden = 256` so the scale is 16.0, which is exact in
/// BF16 — multiplication by a power of two just shifts the exponent and
/// introduces no rounding. Returned length is `num_layers * ple_hidden`.
fn load_ple_raw_row(
    loader: &UnbakedLoader,
    weight_name: &str,
    token_id: u32,
    expected_row_dim: usize,
    ple_hidden: usize,
) -> Result<Vec<f32>> {
    let (shape, bytes) = loader.tensor_bytes(weight_name)?;
    if shape.len() != 2 || shape[1] != expected_row_dim {
        bail!(
            "{weight_name} shape {:?} does not match expected [vocab, {expected_row_dim}]",
            shape
        );
    }
    let vocab = shape[0];
    if (token_id as usize) >= vocab {
        bail!("token id {token_id} out of range (vocab={vocab})");
    }
    let row_bytes = expected_row_dim * 2;
    let off = token_id as usize * row_bytes;
    let raw = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
    let scale = (ple_hidden as f32).sqrt();
    Ok(raw.iter().map(|v| v * scale).collect())
}

/// Compute `per_layer_inputs` for a single decode-step input token, mirroring
/// HF's `Gemma4TextModel.project_per_layer_inputs` + `get_per_layer_inputs`.
///
/// Formula (all BF16-resident, f32 accumulations internally):
///   main_embed_scaled = embed_tokens[tok] * bf16(sqrt(hidden_size))
///   proj = per_layer_model_projection @ main_embed_scaled           // [num_layers * ple]
///   proj = proj * bf16(hidden_size^-0.5)                            // scalar-cast mul
///   proj = proj.reshape(num_layers, ple)
///   proj = per_layer_projection_norm(proj)                          // per-row RMSNorm
///   ple_raw = embed_tokens_per_layer[tok] * sqrt(ple)               // 16.0, exact in BF16
///   out = (proj + ple_raw) * bf16(2^-0.5)
///
/// Returns the combined `out` as BF16 bytes, shape `[num_layers, ple]`.
fn compute_per_layer_inputs(
    loader: &UnbakedLoader,
    weight_prefix: &str,
    tcfg: &TextConfig,
    token_id: u32,
    per_layer_model_projection_w: &GpuBuffer,
    per_layer_projection_norm_w: &GpuBuffer,
    counter: &mut GpuBuffer,
) -> Result<Vec<u8>> {
    let hidden_size = tcfg.hidden_size;
    let num_layers = tcfg.num_hidden_layers;
    let ple_hidden = tcfg.hidden_size_per_layer_input;
    let eps = tcfg.rms_norm_eps as f32;
    let dtype = ScalarType::BF16;
    let total = num_layers * ple_hidden;

    let main_embed_host = load_scaled_embed_row(
        loader,
        &format!("{weight_prefix}.embed_tokens.weight"),
        token_id,
        hidden_size,
    )?;
    let main_embed_gpu = upload_bf16(&[hidden_size], &main_embed_host)?;

    let mut proj = GpuBuffer::zeros(0, dtype, &[total])?;
    g4::matvec(
        0, dtype, &mut proj, &main_embed_gpu, per_layer_model_projection_w,
        hidden_size, total, counter,
    )?;

    // HF applies the scale as a Python float against a BF16 tensor, so the
    // effective multiplier is `bf16(hidden_size^-0.5)`. Mirror that rounding.
    let proj_scale = bf16::from_f32((hidden_size as f32).powf(-0.5)).to_f32();
    let mut proj_host = download_bf16(&proj)?;
    for v in proj_host.iter_mut() {
        *v *= proj_scale;
    }

    let proj_reshaped = upload_bf16(&[num_layers, ple_hidden], &proj_host)?;
    let mut proj_normed = GpuBuffer::zeros(0, dtype, &[num_layers, ple_hidden])?;
    g4::rms_norm_per_row(
        0, dtype, &mut proj_normed, &proj_reshaped,
        Some(per_layer_projection_norm_w), eps, num_layers, ple_hidden,
    )?;
    let proj_normed_host = download_bf16(&proj_normed)?;

    let ple_raw = load_ple_raw_row(
        loader,
        &format!("{weight_prefix}.embed_tokens_per_layer.weight"),
        token_id,
        total,
        ple_hidden,
    )?;

    let combine_scale = bf16::from_f32(2.0f32.powf(-0.5)).to_f32();
    let combined: Vec<f32> = proj_normed_host
        .iter()
        .zip(ple_raw.iter())
        .map(|(p, r)| (p + r) * combine_scale)
        .collect();
    Ok(f32_to_bf16_bytes(&combined))
}

fn seed_kv_cache_from_oracle(
    oracle_k: &[u8],
    oracle_k_shape: &[usize],
    num_kv_heads: usize,
    prompt_tokens: usize,
    head_dim: usize,
    max_t: usize,
    positions_to_copy: usize,
) -> Result<GpuBuffer> {
    if oracle_k_shape != [1, num_kv_heads, prompt_tokens, head_dim] {
        bail!(
            "oracle KV shape {:?} != expected [1, {num_kv_heads}, {prompt_tokens}, {head_dim}]",
            oracle_k_shape
        );
    }
    if positions_to_copy > prompt_tokens {
        bail!(
            "positions_to_copy {positions_to_copy} > prompt_tokens {prompt_tokens}",
        );
    }
    let bf16_sz = 2;
    let total_bytes = num_kv_heads * max_t * head_dim * bf16_sz;
    let mut buf = vec![0u8; total_bytes];
    let row_bytes = head_dim * bf16_sz;
    for h in 0..num_kv_heads {
        for t in 0..positions_to_copy {
            let src_off = ((h * prompt_tokens) + t) * row_bytes;
            let dst_off = ((h * max_t) + t) * row_bytes;
            buf[dst_off..dst_off + row_bytes]
                .copy_from_slice(&oracle_k[src_off..src_off + row_bytes]);
        }
    }
    Ok(GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[num_kv_heads, max_t, head_dim],
        &buf,
    )?)
}

/// Copy a single timestep slot (all heads) from `src` into `dst` on the device.
/// Both buffers must be shape `[num_kv_heads, max_t, head_dim]` BF16 with the
/// same `max_t`. `pos` is the t-index to copy.
fn copy_kv_slot(
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
    num_kv_heads: usize,
    max_t: usize,
    head_dim: usize,
    pos: usize,
) -> Result<()> {
    let row_bytes = head_dim * 2;
    for h in 0..num_kv_heads {
        let byte_off = ((h * max_t) + pos) * row_bytes;
        let src_ptr = src.offset_ptr(byte_off);
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(byte_off) as *mut c_void };
        gpu_hal::copy_d2d(0, dst_ptr, src_ptr, row_bytes)
            .map_err(|e| anyhow!("copy_kv_slot: {e}"))?;
    }
    Ok(())
}

struct CompareStats {
    cos_sim: f32,
    max_abs: f32,
    rel_err_norm: f32,
}

fn compare_vectors(got: &[f32], want: &[f32]) -> Result<CompareStats> {
    if got.len() != want.len() {
        bail!("length mismatch got={} want={}", got.len(), want.len());
    }
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut diff_sq = 0.0f64;
    let mut want_sq = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        dot += g as f64 * w as f64;
        ng += g as f64 * g as f64;
        nw += w as f64 * w as f64;
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff_sq += (g - w) as f64 * (g - w) as f64;
        want_sq += w as f64 * w as f64;
    }
    let cos_sim = if ng > 0.0 && nw > 0.0 {
        (dot / (ng.sqrt() * nw.sqrt())) as f32
    } else {
        0.0
    };
    let rel_err_norm = if want_sq > 0.0 {
        (diff_sq.sqrt() / want_sq.sqrt()) as f32
    } else {
        0.0
    };
    Ok(CompareStats { cos_sim, max_abs, rel_err_norm })
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

fn top_k_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Preloaded per-layer weights. Kept together so the step loop can iterate
/// layers without re-reading safetensors every step.
struct LayerWeights {
    kind: AttnKind,
    head_dim: usize,
    intermediate_size: usize,
    shared_kv: bool,
    /// Source layer index for shared-KV layers (only valid if `shared_kv`).
    kv_source: usize,
    layer_scalar: f32,

    input_norm: GpuBuffer,
    q_proj: GpuBuffer,
    q_norm: GpuBuffer,
    k_proj: Option<GpuBuffer>,
    v_proj: Option<GpuBuffer>,
    k_norm: Option<GpuBuffer>,
    o_proj: GpuBuffer,
    post_attn_norm: GpuBuffer,
    pre_ff_norm: GpuBuffer,
    post_ff_norm: GpuBuffer,
    gate_proj: GpuBuffer,
    up_proj: GpuBuffer,
    down_proj: GpuBuffer,
    per_layer_input_gate_w: GpuBuffer,
    per_layer_projection_w: GpuBuffer,
    post_per_layer_input_norm_w: GpuBuffer,
}

fn load_layer_weights(
    loader: &UnbakedLoader,
    tcfg: &TextConfig,
    weight_prefix: &str,
    layer_idx: usize,
) -> Result<LayerWeights> {
    let kind = tcfg
        .attn_kind(layer_idx)
        .ok_or_else(|| anyhow!("layer {layer_idx}: no attention kind"))?;
    let head_dim = tcfg.head_dim_for(kind);
    let intermediate_size = g4_spec::mlp_intermediate(tcfg, layer_idx);
    let source = tcfg.kv_source_layer(layer_idx);
    let shared_kv = source.is_some();
    let kv_source = source.unwrap_or(layer_idx);

    let specs = g4_spec::layer_tensors(tcfg, weight_prefix, layer_idx);
    let want = |short: &str| -> Result<String> {
        specs
            .iter()
            .find(|s| s.name.ends_with(short))
            .map(|s| s.name.clone())
            .ok_or_else(|| anyhow!("no tensor spec matching *.{short}"))
    };

    let layer_scalar_value: f32 = {
        let (shape, bytes) = loader.tensor_bytes(&want("layer_scalar")?)?;
        if shape != [1] {
            bail!("layer_scalar shape {:?} != [1]", shape);
        }
        bf16_bytes_to_f32(bytes)[0]
    };

    let (k_proj, v_proj, k_norm) = if shared_kv {
        (None, None, None)
    } else {
        (
            Some(loader.load_bf16_to_gpu(&want("self_attn.k_proj.weight")?)?),
            Some(loader.load_bf16_to_gpu(&want("self_attn.v_proj.weight")?)?),
            Some(loader.load_bf16_to_gpu(&want("self_attn.k_norm.weight")?)?),
        )
    };

    Ok(LayerWeights {
        kind,
        head_dim,
        intermediate_size,
        shared_kv,
        kv_source,
        layer_scalar: layer_scalar_value,
        input_norm: loader.load_bf16_to_gpu(&want("input_layernorm.weight")?)?,
        q_proj: loader.load_bf16_to_gpu(&want("self_attn.q_proj.weight")?)?,
        q_norm: loader.load_bf16_to_gpu(&want("self_attn.q_norm.weight")?)?,
        k_proj,
        v_proj,
        k_norm,
        o_proj: loader.load_bf16_to_gpu(&want("self_attn.o_proj.weight")?)?,
        post_attn_norm: loader.load_bf16_to_gpu(&want("post_attention_layernorm.weight")?)?,
        pre_ff_norm: loader.load_bf16_to_gpu(&want("pre_feedforward_layernorm.weight")?)?,
        post_ff_norm: loader.load_bf16_to_gpu(&want("post_feedforward_layernorm.weight")?)?,
        gate_proj: loader.load_bf16_to_gpu(&want("mlp.gate_proj.weight")?)?,
        up_proj: loader.load_bf16_to_gpu(&want("mlp.up_proj.weight")?)?,
        down_proj: loader.load_bf16_to_gpu(&want("mlp.down_proj.weight")?)?,
        per_layer_input_gate_w: loader.load_bf16_to_gpu(&want("per_layer_input_gate.weight")?)?,
        per_layer_projection_w: loader.load_bf16_to_gpu(&want("per_layer_projection.weight")?)?,
        post_per_layer_input_norm_w: loader
            .load_bf16_to_gpu(&want("post_per_layer_input_norm.weight")?)?,
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    gpu_hal::set_device(0).map_err(|e| anyhow!("set_device: {e}"))?;

    let config: Config = g4_config::load_config(&cli.model_dir)
        .map_err(|e| anyhow!("load_config: {e}"))?;
    let tcfg: &TextConfig = &config.text_config;

    let oracle_bytes = std::fs::read(&cli.oracle_json)
        .with_context(|| format!("read {}", cli.oracle_json.display()))?;
    let oracle: OracleOutput = serde_json::from_slice(&oracle_bytes)
        .context("parse oracle JSON")?;
    let prompt_token_ids = oracle
        .prompt_token_ids
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing prompt_token_ids; re-run gemma4_oracle.py"))?;
    let prompt_tokens = prompt_token_ids.len();
    let kv_caches = oracle
        .kv_caches
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing kv_caches (need --emit-state)"))?;
    // PLE is now computed in Rust via `compute_per_layer_inputs`. When the
    // oracle also dumped `per_layer_inputs_by_step`, cross-check per step.
    let pli_by_step_oracle = oracle.per_layer_inputs_by_step.as_ref();

    let last_token_id = *prompt_token_ids
        .last()
        .ok_or_else(|| anyhow!("prompt_tokens == 0"))?;
    let hidden_size = tcfg.hidden_size;
    let eps = tcfg.rms_norm_eps as f32;
    let ple_hidden = tcfg.hidden_size_per_layer_input;
    let num_layers = tcfg.num_hidden_layers;
    let dtype = ScalarType::BF16;
    let vocab_size = tcfg.vocab_size;
    let num_q_heads = tcfg.num_attention_heads;
    let num_kv_heads = tcfg.num_key_value_heads;

    let max_new_tokens = cli.max_new_tokens;
    if max_new_tokens == 0 {
        bail!("--max-new-tokens must be >= 1");
    }
    if max_new_tokens > oracle.generated_tokens {
        bail!(
            "--max-new-tokens {} exceeds oracle.generated_tokens {}; rerun oracle with more tokens",
            max_new_tokens, oracle.generated_tokens
        );
    }
    if let Some(pli) = pli_by_step_oracle {
        if pli.len() < max_new_tokens {
            bail!(
                "oracle per_layer_inputs_by_step has {} entries, need {}",
                pli.len(), max_new_tokens
            );
        }
    }
    if oracle.decode_logits.len() < max_new_tokens.saturating_sub(1) {
        bail!(
            "oracle decode_logits has {} entries, need at least {}",
            oracle.decode_logits.len(),
            max_new_tokens.saturating_sub(1),
        );
    }

    let max_t = prompt_tokens + max_new_tokens;

    let loader = UnbakedLoader::open(&cli.model_dir)?;
    let weight_prefix = "model.language_model";
    let mut counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;

    // Fused attention block scratch. workspace sized for full-attn layers
    // (head_dim=512) which strictly dominate sliding layers (head_dim=256).
    let full_head_dim = tcfg.head_dim_for(AttnKind::Full);
    let workspace_elems = g4::fused_attn_block_workspace_elems(
        hidden_size, num_q_heads, num_kv_heads, full_head_dim, max_t,
    );
    let mut fused_workspace =
        GpuBuffer::zeros(0, ScalarType::F32, &[workspace_elems])?;
    let mut fused_matvec_counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut fused_barrier_counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut fused_barrier_flag = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;

    // Fused MLP+PLE scratch, sized for the largest intermediate
    // (`double_wide_mlp` layers at intermediate=12288).
    let max_intermediate = (0..num_layers)
        .map(|l| g4_spec::mlp_intermediate(tcfg, l))
        .max()
        .unwrap_or(tcfg.intermediate_size);
    let mlp_workspace_elems = g4::fused_mlp_ple_workspace_elems(
        hidden_size, max_intermediate, ple_hidden,
    );
    let mut fused_mlp_workspace =
        GpuBuffer::zeros(0, ScalarType::F32, &[mlp_workspace_elems])?;
    // Per-layer PLI slot (single layer's [ple_hidden] BF16 vector). Populated
    // via D2D copy from the full pli_gpu table each layer.
    let mut pli_slot = GpuBuffer::zeros(0, ScalarType::BF16, &[ple_hidden])?;

    println!(
        "[cfg] prompt_tokens={prompt_tokens} last_token_id={last_token_id} \
         max_new_tokens={max_new_tokens} max_t={max_t} num_layers={num_layers} \
         hidden={hidden_size} ple_hidden={ple_hidden} \
         fused_workspace_elems={workspace_elems}"
    );

    // --- Preload every layer's weights (one pass over safetensors) ---
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(load_layer_weights(&loader, tcfg, weight_prefix, i)?);
    }

    // --- lm_head is tied to embed_tokens on E2B (tie_word_embeddings=true) ---
    let lm_head_w = loader.load_bf16_to_gpu(&format!("{weight_prefix}.embed_tokens.weight"))?;
    let final_norm_w = loader.load_bf16_to_gpu(&format!("{weight_prefix}.norm.weight"))?;

    // --- Global PLE projection weights for Rust-side `project_per_layer_inputs`. ---
    let per_layer_model_projection_w =
        loader.load_bf16_to_gpu(&format!("{weight_prefix}.per_layer_model_projection.weight"))?;
    let per_layer_projection_norm_w =
        loader.load_bf16_to_gpu(&format!("{weight_prefix}.per_layer_projection_norm.weight"))?;

    // --- Preallocate K/V caches for every layer, seed from the oracle's
    //     prefill dump. Shared layers mirror their source layer's oracle dump
    //     because HF fills only 15 cache slots (one per non-shared layer) and
    //     shared layers alias into them. We leave slot `prompt_tokens-1`
    //     unwritten so step 0's `kv_append` exercises the write path — the
    //     same behavior `gemma4_layer0_validate --chain` relies on. ---
    let mut k_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
    let mut v_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let w = &layers[l];
        let src_slot = if w.shared_kv { w.kv_source } else { l };
        let kv = kv_caches
            .get(src_slot)
            .ok_or_else(|| anyhow!("oracle kv_caches has no slot {src_slot} (for layer {l})"))?;
        if kv.layer != src_slot {
            bail!(
                "oracle kv_caches[{src_slot}].layer = {} != {src_slot}",
                kv.layer
            );
        }
        let k_bytes = B64.decode(&kv.k).context("decode kv.k base64")?;
        let v_bytes = B64.decode(&kv.v).context("decode kv.v base64")?;
        let positions_to_copy = prompt_tokens.saturating_sub(1);
        let kc = seed_kv_cache_from_oracle(
            &k_bytes, &kv.k_shape, num_kv_heads, prompt_tokens, w.head_dim, max_t,
            positions_to_copy,
        )?;
        let vc = seed_kv_cache_from_oracle(
            &v_bytes, &kv.v_shape, num_kv_heads, prompt_tokens, w.head_dim, max_t,
            positions_to_copy,
        )?;
        k_caches.push(kc);
        v_caches.push(vc);
    }

    // --- RoPE tables: build once per kind, sized to max_t so every step's
    //     position has a valid row. Matches the construction used in
    //     gemma4_layer0_validate but with a larger max_pos. ---
    let sliding_rope = tcfg.rope_for(AttnKind::Sliding);
    let full_rope = tcfg.rope_for(AttnKind::Full);
    let sliding_head_dim = tcfg.head_dim_for(AttnKind::Sliding);
    let full_head_dim = tcfg.head_dim_for(AttnKind::Full);

    let (scos_h, ssin_h) = build_sliding_rope_table(sliding_head_dim, sliding_rope.rope_theta, max_t);
    let (fcos_h, fsin_h) = build_proportional_rope_table(
        full_head_dim, full_rope.rope_theta, full_rope.partial_rotary_factor, max_t,
    );
    let sliding_cos = upload_bf16(&[max_t, sliding_head_dim], &scos_h)?;
    let sliding_sin = upload_bf16(&[max_t, sliding_head_dim], &ssin_h)?;
    let full_cos = upload_bf16(&[max_t, full_head_dim], &fcos_h)?;
    let full_sin = upload_bf16(&[max_t, full_head_dim], &fsin_h)?;

    let cap = tcfg.final_logit_softcapping.unwrap_or(30.0) as f32;

    // --- Decode loop ---
    let mut generated_ids_rust: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut overall_pass = true;

    for step in 0..max_new_tokens {
        let pos = prompt_tokens - 1 + step;
        let input_token_id: u32 = if step == 0 {
            last_token_id
        } else {
            generated_ids_rust[step - 1]
        };

        let h_in_host = load_scaled_embed_row(
            &loader,
            &format!("{weight_prefix}.embed_tokens.weight"),
            input_token_id,
            hidden_size,
        )?;
        let mut h_running = upload_bf16(&[hidden_size], &h_in_host)?;

        // Compute this step's per_layer_inputs in Rust, mirroring HF's
        // `project_per_layer_inputs`. Shape: [num_layers, ple_hidden] BF16.
        let pli_bytes = compute_per_layer_inputs(
            &loader, weight_prefix, tcfg, input_token_id,
            &per_layer_model_projection_w, &per_layer_projection_norm_w,
            &mut counter,
        )?;
        let expected_pli_bytes = num_layers * ple_hidden * 2;
        if pli_bytes.len() != expected_pli_bytes {
            bail!(
                "compute_per_layer_inputs returned {} bytes, expected {}",
                pli_bytes.len(), expected_pli_bytes
            );
        }

        // Cross-check against oracle PLE when available — pure diagnostic.
        if let Some(oracle_pli) = pli_by_step_oracle {
            let want_bytes = B64
                .decode(&oracle_pli[step])
                .with_context(|| format!("decode oracle per_layer_inputs_by_step[{step}]"))?;
            let got_f32 = bf16_bytes_to_f32(&pli_bytes);
            let want_f32 = bf16_bytes_to_f32(&want_bytes);
            let stats = compare_vectors(&got_f32, &want_f32)?;
            println!(
                "[step{step}] pli_vs_oracle: cos_sim={:.6} max_abs={:.6} rel_err={:.6}",
                stats.cos_sim, stats.max_abs, stats.rel_err_norm,
            );
        }

        // Upload the full PLI table once per step; per-layer slices are
        // copied into `pli_slot` inside the loop with a small D2D.
        let pli_gpu = GpuBuffer::from_host_bytes(
            0, dtype, &[num_layers, ple_hidden], &pli_bytes,
        )?;

        for layer_idx in 0..num_layers {
            let w = &layers[layer_idx];
            let head_dim = w.head_dim;
            let rotary_dim = head_dim;
            let q_dim = num_q_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let sliding_window = match w.kind {
                AttnKind::Sliding => tcfg.sliding_window as i32,
                AttnKind::Full => 0,
            };
            let (cos_table, sin_table) = match w.kind {
                AttnKind::Sliding => (&sliding_cos, &sliding_sin),
                AttnKind::Full => (&full_cos, &full_sin),
            };

            // Fused attention block: input_norm → Q/K/V proj → q/k/v norm →
            // RoPE → kv_append → SWA/full attn → o_proj → post_attn_norm →
            // residual, in a single kernel launch. For shared-KV layers the
            // kernel skips K/V staging and reads directly from the cache,
            // which must already hold the inherited slot at `pos` (copied
            // from the source layer below, earlier in this iteration).
            let _ = (q_dim, kv_dim, rotary_dim); // still used for workspace sizing logic above
            let mut h_mid = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
            g4::fused_attn_block(
                0, dtype,
                &h_running, &mut h_mid,
                &w.input_norm,
                &w.q_proj,
                w.k_proj.as_ref(),
                w.v_proj.as_ref(),
                &w.q_norm,
                w.k_norm.as_ref(),
                &w.o_proj,
                &w.post_attn_norm,
                cos_table, sin_table,
                &mut k_caches[layer_idx], &mut v_caches[layer_idx],
                &mut fused_workspace,
                &mut fused_matvec_counter,
                &mut fused_barrier_counter, &mut fused_barrier_flag,
                hidden_size, num_q_heads, num_kv_heads, head_dim, head_dim,
                sliding_window, pos, max_t,
                w.shared_kv,
                eps, 1.0f32,
            )?;

            // After a non-shared layer wrote its K/V at slot `pos`, replicate
            // the slot into every shared layer that sources from this one.
            // Source → shared indices monotonically increase so dependents
            // later in this iteration see the fresh slot.
            if !w.shared_kv {
                for shared_layer in (layer_idx + 1)..num_layers {
                    let s = &layers[shared_layer];
                    if s.shared_kv && s.kv_source == layer_idx {
                        let (lo, hi) = k_caches.split_at_mut(shared_layer);
                        copy_kv_slot(&lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                        let (lo, hi) = v_caches.split_at_mut(shared_layer);
                        copy_kv_slot(&lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                    }
                }
            }

            // Fused MLP + PLE half: pre_ff_norm → gate/up → gelu*up → down →
            // post_ff_norm → residual → per_layer_input_gate → gelu*pli →
            // per_layer_projection → post_per_layer_input_norm →
            // (+)*layer_scalar, all in one kernel launch. Reads the
            // per-layer-input slice straight from `pli_gpu` at the correct
            // offset — no host-side reupload per layer.
            let pli_byte_off = layer_idx * ple_hidden * 2;
            let pli_slice_ptr = pli_gpu.offset_ptr(pli_byte_off);
            // Build a thin borrowed GpuBuffer view via from_host_bytes is
            // awkward; instead, expose the underlying pointer to the FFI by
            // constructing a temporary GpuBuffer whose ptr aliases the right
            // offset. Simpler: copy the slice into a dedicated scratch buffer
            // owned at outer scope and pass that.
            unsafe {
                gpu_hal::copy_d2d(
                    0,
                    pli_slot.as_mut_ptr(),
                    pli_slice_ptr,
                    ple_hidden * 2,
                ).map_err(|e| anyhow!("copy pli slice: {e}"))?;
            }

            let mut h_new = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
            g4::fused_mlp_ple(
                0, dtype,
                &h_mid, &mut h_new,
                &w.pre_ff_norm,
                &w.gate_proj, &w.up_proj, &w.down_proj, &w.post_ff_norm,
                &pli_slot,
                &w.per_layer_input_gate_w, &w.per_layer_projection_w,
                &w.post_per_layer_input_norm_w,
                &mut fused_mlp_workspace,
                &mut fused_matvec_counter,
                &mut fused_barrier_counter, &mut fused_barrier_flag,
                hidden_size, w.intermediate_size, ple_hidden,
                eps, w.layer_scalar,
            )?;
            h_running = h_new;
        }

        // --- Final norm + tied lm_head + softcap ---
        let mut post_norm = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut post_norm, &h_running, Some(&final_norm_w), eps, hidden_size)?;

        let mut logits_gpu = GpuBuffer::zeros(0, dtype, &[vocab_size])?;
        g4::matvec(
            0, dtype, &mut logits_gpu, &post_norm, &lm_head_w,
            hidden_size, vocab_size, &mut counter,
        )?;
        let mut logits_host = download_bf16(&logits_gpu)?;
        for v in logits_host.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }

        // --- Compare against oracle ---
        let (want_logits, source_tag) = if step == 0 {
            (&oracle.prefill_logits, "prefill_logits")
        } else {
            (&oracle.decode_logits[step - 1], "decode_logits[step-1]")
        };
        if want_logits.len() != vocab_size {
            bail!(
                "oracle {} len {} != vocab_size {}",
                source_tag, want_logits.len(), vocab_size
            );
        }
        let stats = compare_vectors(&logits_host, want_logits)?;
        let got_arg = argmax(&logits_host);
        let want_arg = argmax(want_logits);
        let top5_got = top_k_indices(&logits_host, 5);
        let top5_want = top_k_indices(want_logits, 5);
        let top5_overlap = top5_got.iter().filter(|i| top5_want.contains(*i)).count();
        let expected_gen_id = oracle.generated_token_ids[step];
        let match_argmax = got_arg as u32 == expected_gen_id;
        let match_oracle_arg = got_arg == want_arg;

        println!(
            "[step{step}] input_tok={input_token_id} pos={pos}  \
             vs {source_tag}: cos_sim={:.6} max_abs={:.6} rel_err={:.6}",
            stats.cos_sim, stats.max_abs, stats.rel_err_norm,
        );
        println!(
            "  argmax got={got_arg} want={want_arg} expected_gen_id={expected_gen_id} \
             argmax_vs_logits={} argmax_vs_gen_id={} top5_overlap={top5_overlap}/5",
            if match_oracle_arg { "MATCH" } else { "MISMATCH" },
            if match_argmax { "MATCH" } else { "MISMATCH" },
        );
        println!("  top5_got={:?} top5_want={:?}", top5_got, top5_want);

        generated_ids_rust.push(got_arg as u32);

        if stats.cos_sim < 0.999 || !match_argmax || !match_oracle_arg {
            overall_pass = false;
            println!(
                "[step{step}] FAIL: cos_sim<0.999 or argmax mismatch; continuing to collect diagnostics"
            );
        }
    }

    // --- Final summary ---
    println!("[summary] generated_ids_rust = {:?}", generated_ids_rust);
    println!(
        "[summary] oracle.generated_token_ids[0..{max_new_tokens}] = {:?}",
        &oracle.generated_token_ids[..max_new_tokens]
    );
    let ids_match = generated_ids_rust
        .iter()
        .zip(oracle.generated_token_ids.iter().take(max_new_tokens))
        .all(|(a, b)| a == b);
    if !ids_match {
        overall_pass = false;
    }

    if !overall_pass {
        bail!("decode validation failed (see per-step diagnostics above)");
    }
    println!("PASS");
    Ok(())
}
