//! End-to-end single-layer correctness check for the Gemma 4 decode path.
//!
//! Given a Gemma 4 E2B checkpoint directory and an oracle JSON produced with
//! `oracle/gemma4_oracle.py --emit-state`, this binary runs a single
//! transformer layer through the Rust-side primitive kernels and compares the
//! resulting *pre-PLE* hidden state against the oracle's snapshot. Works for
//! both sliding-window (SWA) layers and full-attention layers.
//!
//! Layer 0 input is reconstructed from the tokenizer-output prompt IDs
//! (emitted by the oracle as `prompt_token_ids`) by looking up the last
//! token's row in `embed_tokens.weight` and scaling by `sqrt(hidden_size)` —
//! Gemma 4's `Gemma4TextScaledWordEmbedding` multiplies by embed_scale at
//! lookup time. For layer N > 0, the input hidden is sourced from the
//! oracle's `prefill_per_layer_hidden[N-1]` (layer N-1's post-block,
//! post-PLE+layer_scalar output) — this side-steps PLE plumbing entirely
//! so the full-attention path can be validated in isolation.
//!
//! The K/V cache is seeded from the oracle's `kv_caches[layer]`. The last
//! entry is truncated before our kernel runs, then re-appended via
//! `g4::kv_append` after the current-token K/V are computed — this exercises
//! the full K/V path end-to-end.
//!
//! Usage:
//!   cargo run --release --bin gemma4_layer0_validate -- \
//!     --model-dir <checkpoint> --oracle-json <oracle_state.json> [--layer N]
//!
//! Expected outcome: cosine similarity ≥ 0.999 between our Rust kernel's
//! pre-PLE hidden state at the last prompt token and
//! `prefill_per_layer_pre_ple[layer]`.

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
#[command(about = "Validate a single Gemma 4 SWA layer against the PyTorch oracle")]
struct Cli {
    /// Path to a local Gemma 4 checkpoint directory (config.json + safetensors).
    #[arg(long)]
    model_dir: PathBuf,
    /// Oracle JSON file produced by `oracle/gemma4_oracle.py --emit-state`.
    #[arg(long)]
    oracle_json: PathBuf,
    /// Layer index to validate (0..num_hidden_layers). Default 0.
    #[arg(long, default_value_t = 0)]
    layer: usize,
    /// If set, chain layers 0..=layer end-to-end starting from the scaled
    /// embed_tokens lookup (no oracle seeding between layers). Otherwise the
    /// binary only runs the target layer and seeds its input hidden from the
    /// oracle's `prefill_per_layer_hidden[layer-1]` (or embed_tokens for layer 0).
    #[arg(long)]
    chain: bool,
}

// -----------------------------------------------------------------------------
// BF16 byte helpers
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Safetensors loader (unbaked): memmap every shard once, load tensors by name
// -----------------------------------------------------------------------------

struct UnbakedLoader {
    shards: Vec<Mmap>,
    /// Tensor name → index into `shards`.
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

    /// Return the raw BF16 byte view of a tensor in its original on-disk layout.
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

// -----------------------------------------------------------------------------
// Build a Gemma 4 RoPE cos/sin table of shape [max_pos, head_dim] matching
// HF's `emb = cat((freqs, freqs), dim=-1)` layout.
//
// Given an inv_freq vector of length head_dim/2 (possibly zero-padded for the
// "nope" portion of proportional RoPE), produce:
//   cos[p, i]        = cos(p * inv_freq[i])         for i in 0..half
//   sin[p, i]        = sin(p * inv_freq[i])         for i in 0..half
//   cos[p, i+half]   = cos[p, i]                    (duplicate per HF cat)
//   sin[p, i+half]   = sin[p, i]                    (duplicate per HF cat)
//
// `attention_scaling` multiplies both cos and sin (HF applies this
// post-emb). For Gemma 4 proportional RoPE attention_factor == 1.0.
// -----------------------------------------------------------------------------

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

/// Sliding-attention RoPE (rope_type="default"). Applies to all head_dim/2
/// frequency slots (partial_rotary_factor=1.0 on E2B sliding layers).
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

/// Full-attention RoPE for Gemma 4 (rope_type="proportional"). Mirrors HF's
/// `_compute_proportional_rope_parameters` in `modeling_rope_utils.py`.
///
/// Differences from default RoPE:
///   * Only the first `rope_angles = floor(partial_rotary_factor * head_dim / 2)`
///     frequency slots are populated; the rest are zero ("nope" positions).
///   * inv_freq[j] = 1 / rope_theta^(2j / head_dim) — note the denominator is
///     `head_dim`, not `dim = head_dim * partial_rotary_factor` as in default.
///   * attention_factor is always 1.0 per HF's implementation (and the
///     kernel is called with `rotary_dim = head_dim` so the nope slots simply
///     pass through via cos=1, sin=0).
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

// -----------------------------------------------------------------------------
// Extract the last prompt token's embedding row from `embed_tokens.weight`
// without allocating the full embedding table on the GPU. Applies Gemma 4's
// `embed_scale = sqrt(hidden_size)` scaling.
// -----------------------------------------------------------------------------

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
    // HF stores `embed_scale` as a BF16 scalar (`self.embed_scale.to(self.weight.dtype)`),
    // so the round-to-nearest step from sqrt(hidden_size) happens before the multiply.
    // Mirror that precision exactly so our layer-0 input matches HF bit-for-bit at
    // BF16 resolution.
    let scale_bf16 = bf16::from_f32((hidden_size as f32).sqrt());
    let scale = scale_bf16.to_f32();
    Ok(row.iter().map(|v| v * scale).collect())
}

// -----------------------------------------------------------------------------
// KV cache setup: oracle stores [1, num_kv_heads, prompt_tokens, head_dim]
// (BF16). Our kernel layout is [num_kv_heads, max_T, head_dim]. With
// `max_T = prompt_tokens`, the byte layout matches the oracle for positions
// 0..prompt_tokens — we pre-fill all positions except the last, then let
// `kv_append` write the last slot. head_dim is 256 for SWA layers and 512
// for full-attention layers on E2B.
// -----------------------------------------------------------------------------

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
    // Oracle layout: [1, num_kv_heads, prompt_tokens, head_dim] row-major,
    //   offset(h, t, d) = (h * prompt_tokens + t) * head_dim + d
    // Our layout:     [num_kv_heads, max_T, head_dim] row-major,
    //   offset(h, t, d) = (h * max_T + t) * head_dim + d
    // Both iterate d contiguously, but the t-stride differs when prompt_tokens
    // != max_T. Copy a head-row at a time per timestep.
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

// -----------------------------------------------------------------------------
// Vector stats: cosine similarity and max abs diff
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    gpu_hal::set_device(0).map_err(|e| anyhow!("set_device: {e}"))?;

    // --- Config + oracle ---
    let config: Config = g4_config::load_config(&cli.model_dir)
        .map_err(|e| anyhow!("load_config: {e}"))?;
    let tcfg: &TextConfig = &config.text_config;

    if cli.layer >= tcfg.num_hidden_layers {
        bail!("--layer {} >= num_hidden_layers {}", cli.layer, tcfg.num_hidden_layers);
    }

    let oracle_bytes = std::fs::read(&cli.oracle_json)
        .with_context(|| format!("read {}", cli.oracle_json.display()))?;
    let oracle: OracleOutput = serde_json::from_slice(&oracle_bytes)
        .context("parse oracle JSON")?;
    let prompt_token_ids = oracle
        .prompt_token_ids
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing prompt_token_ids; re-run gemma4_oracle.py"))?;
    if prompt_token_ids.len() != oracle.prompt_tokens {
        bail!(
            "prompt_token_ids length {} != prompt_tokens {}",
            prompt_token_ids.len(),
            oracle.prompt_tokens
        );
    }
    let kv_caches = oracle
        .kv_caches
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing kv_caches (need --emit-state)"))?;
    let pre_ple = oracle
        .prefill_per_layer_pre_ple
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing prefill_per_layer_pre_ple"))?;
    let per_layer_hidden = oracle
        .prefill_per_layer_hidden
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing prefill_per_layer_hidden"))?;
    let per_layer_inputs_b64 = oracle
        .per_layer_inputs
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing per_layer_inputs; re-run gemma4_oracle.py"))?;
    let per_layer_inputs_shape = oracle
        .per_layer_inputs_shape
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing per_layer_inputs_shape"))?;

    if cli.layer >= pre_ple.len() {
        bail!("prefill_per_layer_pre_ple has only {} entries", pre_ple.len());
    }

    let prompt_tokens = oracle.prompt_tokens;
    let last_token_id = *prompt_token_ids
        .last()
        .ok_or_else(|| anyhow!("prompt_tokens == 0"))?;
    let pos = prompt_tokens - 1;
    let hidden_size = tcfg.hidden_size;
    let eps = tcfg.rms_norm_eps as f32;
    let ple_hidden = tcfg.hidden_size_per_layer_input;
    let max_t = prompt_tokens;
    let dtype = ScalarType::BF16;

    if per_layer_inputs_shape.len() != 2
        || per_layer_inputs_shape[0] != tcfg.num_hidden_layers
        || per_layer_inputs_shape[1] != ple_hidden
    {
        bail!(
            "per_layer_inputs shape {:?} != [{}, {}]",
            per_layer_inputs_shape, tcfg.num_hidden_layers, ple_hidden
        );
    }
    let all_pli_bytes = B64
        .decode(per_layer_inputs_b64)
        .context("decode per_layer_inputs base64")?;

    let loader = UnbakedLoader::open(&cli.model_dir)?;
    let weight_prefix = "model.language_model";
    let mut counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;

    // --- Determine chain range and seed h_in ---
    // --chain runs layers [0, target] starting from the scaled embedding.
    // Without --chain, we only run the target layer, seeded from either the
    // scaled embedding (layer 0) or the oracle's post-PLE output of layer-1.
    let start_layer = if cli.chain { 0 } else { cli.layer };
    let h_in_seed_host: Vec<f32> = if start_layer == 0 {
        load_scaled_embed_row(
            &loader,
            &format!("{weight_prefix}.embed_tokens.weight"),
            last_token_id,
            hidden_size,
        )?
    } else {
        let b64 = &per_layer_hidden[start_layer - 1];
        let bytes = B64.decode(b64).context("decode prefill_per_layer_hidden base64")?;
        let full = bf16_bytes_to_f32(&bytes);
        if full.len() != hidden_size {
            bail!(
                "prefill_per_layer_hidden[{}] len {} != hidden_size {}",
                start_layer - 1,
                full.len(),
                hidden_size
            );
        }
        full
    };
    let mut h_running_host = h_in_seed_host;

    println!(
        "[run] prompt_tokens={prompt_tokens} last_token_id={last_token_id} target_layer={} chain={} (start_layer={})",
        cli.layer, cli.chain, start_layer,
    );

    let mut final_pre_ple: Option<Vec<f32>> = None;
    let mut final_post_ple: Option<Vec<f32>> = None;
    let mut final_layer_scalar: f32 = 1.0;

    for layer in start_layer..=cli.layer {
        let kind = tcfg
            .attn_kind(layer)
            .ok_or_else(|| anyhow!("layer {} has no attention kind", layer))?;
        let head_dim = tcfg.head_dim_for(kind);
        let rotary_dim = head_dim;
        let num_q_heads = tcfg.num_attention_heads;
        let num_kv_heads = tcfg.num_key_value_heads;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_size = g4_spec::mlp_intermediate(tcfg, layer);
        let rope = tcfg.rope_for(kind);
        let rope_theta = rope.rope_theta;
        let partial_rotary_factor = rope.partial_rotary_factor;
        let sliding_window = match kind {
            AttnKind::Sliding => tcfg.sliding_window as i32,
            AttnKind::Full => 0,
        };

        // Shared-KV: layers in the tail (last num_kv_shared_layers) reuse the
        // K/V cache of an earlier non-shared layer of the same attention kind.
        // They don't project their own K/V at runtime — HF slots 0..14 already
        // contain all prompt positions for those caches.
        let kv_source = tcfg.kv_source_layer(layer);
        let shared_kv = kv_source.is_some();
        let kv_slot = kv_source.unwrap_or(layer);
        if kv_slot >= kv_caches.len() {
            bail!(
                "oracle kv_caches has {} slots but layer {} needs slot {} (shared={})",
                kv_caches.len(),
                layer,
                kv_slot,
                shared_kv,
            );
        }
        let kv = &kv_caches[kv_slot];
        if kv.layer != kv_slot {
            bail!(
                "oracle kv_caches[{kv_slot}].layer = {} != {kv_slot}; cache layout changed?",
                kv.layer
            );
        }

        // --- Per-layer weight loading ---
        let layer_spec = g4_spec::layer_tensors(tcfg, weight_prefix, layer);
        let want = |short: &str| {
            layer_spec
                .iter()
                .find(|s| s.name.ends_with(short))
                .map(|s| s.name.clone())
                .ok_or_else(|| anyhow!("no tensor spec matching *.{short}"))
        };
        let input_norm = loader.load_bf16_to_gpu(&want("input_layernorm.weight")?)?;
        let q_proj = loader.load_bf16_to_gpu(&want("self_attn.q_proj.weight")?)?;
        let o_proj = loader.load_bf16_to_gpu(&want("self_attn.o_proj.weight")?)?;
        let q_norm = loader.load_bf16_to_gpu(&want("self_attn.q_norm.weight")?)?;
        // K/V projections and K-norm are only needed on non-shared layers.
        // Shared layers pull K/V verbatim from the source layer's oracle dump.
        let (k_proj, v_proj, k_norm) = if shared_kv {
            (None, None, None)
        } else {
            (
                Some(loader.load_bf16_to_gpu(&want("self_attn.k_proj.weight")?)?),
                Some(loader.load_bf16_to_gpu(&want("self_attn.v_proj.weight")?)?),
                Some(loader.load_bf16_to_gpu(&want("self_attn.k_norm.weight")?)?),
            )
        };
        let post_attn_norm = loader.load_bf16_to_gpu(&want("post_attention_layernorm.weight")?)?;
        let pre_ff_norm = loader.load_bf16_to_gpu(&want("pre_feedforward_layernorm.weight")?)?;
        let post_ff_norm = loader.load_bf16_to_gpu(&want("post_feedforward_layernorm.weight")?)?;
        let gate_proj = loader.load_bf16_to_gpu(&want("mlp.gate_proj.weight")?)?;
        let up_proj = loader.load_bf16_to_gpu(&want("mlp.up_proj.weight")?)?;
        let down_proj = loader.load_bf16_to_gpu(&want("mlp.down_proj.weight")?)?;
        let per_layer_input_gate_w = loader.load_bf16_to_gpu(&want("per_layer_input_gate.weight")?)?;
        let per_layer_projection_w = loader.load_bf16_to_gpu(&want("per_layer_projection.weight")?)?;
        let post_per_layer_input_norm_w =
            loader.load_bf16_to_gpu(&want("post_per_layer_input_norm.weight")?)?;
        let layer_scalar_value: f32 = {
            let (shape, bytes) = loader.tensor_bytes(&want("layer_scalar")?)?;
            if shape != [1] {
                bail!("layer_scalar shape {:?} != [1]", shape);
            }
            bf16_bytes_to_f32(bytes)[0]
        };

        let mut h_in = upload_bf16(&[hidden_size], &h_running_host)?;

        println!(
            "  [layer{}] kind={:?} head_dim={head_dim} imm={intermediate_size} rope_theta={rope_theta} partial_rotary_factor={partial_rotary_factor} layer_scalar={layer_scalar_value:.4} shared_kv={} kv_slot={kv_slot}",
            layer, kind, shared_kv,
        );

        // --- RoPE tables: sliding uses default RoPE with full rotation;
        //     full-attn uses proportional RoPE with partial_rotary_factor=0.25 ---
        let (cos_host, sin_host) = match kind {
            AttnKind::Sliding => build_sliding_rope_table(head_dim, rope_theta, prompt_tokens),
            AttnKind::Full => build_proportional_rope_table(
                head_dim, rope_theta, partial_rotary_factor, prompt_tokens,
            ),
        };
        let cos_table = upload_bf16(&[prompt_tokens, head_dim], &cos_host)?;
        let sin_table = upload_bf16(&[prompt_tokens, head_dim], &sin_host)?;

        // --- KV caches seeded from oracle ---
        // Non-shared layers: copy positions 0..prompt_tokens-1 and let the
        // kernel re-append the last slot via kv_append (exercises the K/V path).
        // Shared layers: copy all prompt_tokens positions from the source
        // layer's oracle dump; no kernel K/V compute, no kv_append.
        let positions_to_copy = if shared_kv {
            prompt_tokens
        } else {
            prompt_tokens.saturating_sub(1)
        };
        let k_oracle_bytes = B64.decode(&kv.k).context("decode kv.k base64")?;
        let v_oracle_bytes = B64.decode(&kv.v).context("decode kv.v base64")?;
        let k_cache = seed_kv_cache_from_oracle(
            &k_oracle_bytes, &kv.k_shape, num_kv_heads, prompt_tokens, head_dim, max_t,
            positions_to_copy,
        )?;
        let v_cache = seed_kv_cache_from_oracle(
            &v_oracle_bytes, &kv.v_shape, num_kv_heads, prompt_tokens, head_dim, max_t,
            positions_to_copy,
        )?;

        // --- Forward pass ---
        // residual = h_in
        let residual = h_in.clone_device()?;

        // x = rms_norm(h_in, input_layernorm.weight)
        let mut x = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x, &h_in, Some(&input_norm), eps, hidden_size)?;

        // Q = matvec(q_proj, x) always; K/V skipped on shared layers.
        let mut q = GpuBuffer::zeros(0, dtype, &[num_q_heads, head_dim])?;
        g4::matvec(0, dtype, &mut q, &x, &q_proj, hidden_size, q_dim, &mut counter)?;

        // Per-head Q_norm (weight per head_dim).
        let mut q_normed = GpuBuffer::zeros(0, dtype, &[num_q_heads, head_dim])?;
        g4::rms_norm_per_row(0, dtype, &mut q_normed, &q, Some(&q_norm), eps, num_q_heads, head_dim)?;

        // RoPE on Q (always).
        g4::rope_decode(0, dtype, &mut q_normed, &cos_table, &sin_table, num_q_heads, head_dim, rotary_dim, pos)?;

        // K/V compute, norm, RoPE-on-K, and kv_append only on non-shared layers.
        let (mut k_cache, mut v_cache) = (k_cache, v_cache);
        if !shared_kv {
            let k_proj = k_proj.as_ref().expect("k_proj must be loaded on non-shared layers");
            let v_proj = v_proj.as_ref().expect("v_proj must be loaded on non-shared layers");
            let k_norm = k_norm.as_ref().expect("k_norm must be loaded on non-shared layers");

            let mut k = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::matvec(0, dtype, &mut k, &x, k_proj, hidden_size, kv_dim, &mut counter)?;
            let mut v = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::matvec(0, dtype, &mut v, &x, v_proj, hidden_size, kv_dim, &mut counter)?;

            let mut k_normed = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::rms_norm_per_row(0, dtype, &mut k_normed, &k, Some(k_norm), eps, num_kv_heads, head_dim)?;
            let mut v_normed = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::rms_norm_per_row(0, dtype, &mut v_normed, &v, None, eps, num_kv_heads, head_dim)?;

            g4::rope_decode(0, dtype, &mut k_normed, &cos_table, &sin_table, num_kv_heads, head_dim, rotary_dim, pos)?;

            g4::kv_append(
                0, dtype, &k_normed, &v_normed, &mut k_cache, &mut v_cache,
                num_kv_heads, head_dim, pos, max_t,
            )?;
        }

        // Attention
        let mut attn_out = GpuBuffer::zeros(0, dtype, &[num_q_heads, head_dim])?;
        let mut scores = GpuBuffer::zeros(0, ScalarType::F32, &[num_q_heads, max_t])?;
        g4::swa_attn_decode(
            0, dtype, &q_normed, &k_cache, &v_cache, &mut scores, &mut attn_out,
            num_q_heads, num_kv_heads, head_dim, prompt_tokens, max_t, sliding_window, 1.0,
        )?;

        // o = o_proj @ attn_out.flatten
        let attn_flat_shape = [q_dim];
        let attn_flat = {
            let bytes = attn_out.to_host_bytes()?;
            GpuBuffer::from_host_bytes(0, dtype, &attn_flat_shape, &bytes)?
        };
        let mut o = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::matvec(0, dtype, &mut o, &attn_flat, &o_proj, q_dim, hidden_size, &mut counter)?;

        // x2 = rms_norm(o, post_attention_layernorm.weight); h = residual + x2
        let mut x2 = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x2, &o, Some(&post_attn_norm), eps, hidden_size)?;
        let residual_h = download_bf16(&residual)?;
        let x2_h = download_bf16(&x2)?;
        let h1_h: Vec<f32> = residual_h.iter().zip(x2_h.iter()).map(|(a, b)| a + b).collect();
        let h_mid = upload_bf16(&[hidden_size], &h1_h)?;

        let residual2 = h_mid.clone_device()?;

        let mut x3 = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x3, &h_mid, Some(&pre_ff_norm), eps, hidden_size)?;

        let mut gate = GpuBuffer::zeros(0, dtype, &[intermediate_size])?;
        g4::matvec(0, dtype, &mut gate, &x3, &gate_proj, hidden_size, intermediate_size, &mut counter)?;
        let mut up_buf = GpuBuffer::zeros(0, dtype, &[intermediate_size])?;
        g4::matvec(0, dtype, &mut up_buf, &x3, &up_proj, hidden_size, intermediate_size, &mut counter)?;
        let mut y = GpuBuffer::zeros(0, dtype, &[intermediate_size])?;
        g4::gelu_tanh_gate_mul(0, dtype, &mut y, &gate, &up_buf, intermediate_size)?;

        let mut m = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::matvec(0, dtype, &mut m, &y, &down_proj, intermediate_size, hidden_size, &mut counter)?;

        let mut x4 = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x4, &m, Some(&post_ff_norm), eps, hidden_size)?;
        let residual2_h = download_bf16(&residual2)?;
        let x4_h = download_bf16(&x4)?;
        let h_pre_ple: Vec<f32> = residual2_h.iter().zip(x4_h.iter()).map(|(a, b)| a + b).collect();

        // --- PLE branch: gate → gelu_tanh → * per_layer_input[N] → projection →
        //     post_per_layer_input_norm → residual add → * layer_scalar ---
        let bytes_per_layer = ple_hidden * 2;
        let pli_off = layer * bytes_per_layer;
        let pli_slice = &all_pli_bytes[pli_off..pli_off + bytes_per_layer];
        let per_layer_input_f32 = bf16_bytes_to_f32(pli_slice);
        let per_layer_input_gpu = upload_bf16(&[ple_hidden], &per_layer_input_f32)?;

        let ple_residual = upload_bf16(&[hidden_size], &h_pre_ple)?;
        let h_in_ple = ple_residual.clone_device()?;

        let mut gated = GpuBuffer::zeros(0, dtype, &[ple_hidden])?;
        g4::matvec(
            0, dtype, &mut gated, &h_in_ple, &per_layer_input_gate_w,
            hidden_size, ple_hidden, &mut counter,
        )?;

        let mut gated_act = GpuBuffer::zeros(0, dtype, &[ple_hidden])?;
        g4::gelu_tanh_gate_mul(
            0, dtype, &mut gated_act, &gated, &per_layer_input_gpu, ple_hidden,
        )?;

        let mut projected = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::matvec(
            0, dtype, &mut projected, &gated_act, &per_layer_projection_w,
            ple_hidden, hidden_size, &mut counter,
        )?;

        let mut normed = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(
            0, dtype, &mut normed, &projected, Some(&post_per_layer_input_norm_w),
            eps, hidden_size,
        )?;

        let ple_residual_h = download_bf16(&ple_residual)?;
        let normed_h = download_bf16(&normed)?;
        let h_post_ple: Vec<f32> = ple_residual_h
            .iter()
            .zip(normed_h.iter())
            .map(|(a, b)| (a + b) * layer_scalar_value)
            .collect();

        if layer == cli.layer {
            final_pre_ple = Some(h_pre_ple.clone());
            final_post_ple = Some(h_post_ple.clone());
            final_layer_scalar = layer_scalar_value;
        }

        // Feed this layer's post-PLE into the next iteration.
        h_running_host = h_post_ple;

        // Suppress unused warning; reserved for future diagnostic use.
        let _ = (q_proj.as_ptr(), o_proj.as_ptr());
        let _: *const c_void = gate_proj.as_ptr();
    }

    // --- Compare final-layer outputs against oracle ---
    let final_pre_ple = final_pre_ple.ok_or_else(|| anyhow!("no layers were run"))?;
    let final_post_ple = final_post_ple.ok_or_else(|| anyhow!("no layers were run"))?;

    let pre_ple_want_bytes = B64
        .decode(&pre_ple[cli.layer])
        .context("decode prefill_per_layer_pre_ple base64")?;
    let pre_ple_want = bf16_bytes_to_f32(&pre_ple_want_bytes);
    if pre_ple_want.len() != hidden_size {
        bail!("pre_ple[{}] len {} != hidden_size {}", cli.layer, pre_ple_want.len(), hidden_size);
    }
    let pre_ple_stats = compare_vectors(&final_pre_ple, &pre_ple_want)?;
    println!(
        "[layer{} pre-PLE]  cos_sim={:.6}  max_abs={:.6}  rel_err_norm={:.6}",
        cli.layer, pre_ple_stats.cos_sim, pre_ple_stats.max_abs, pre_ple_stats.rel_err_norm
    );

    let post_ple_want_bytes = B64
        .decode(&per_layer_hidden[cli.layer])
        .context("decode prefill_per_layer_hidden base64")?;
    let post_ple_want = bf16_bytes_to_f32(&post_ple_want_bytes);
    if post_ple_want.len() != hidden_size {
        bail!(
            "per_layer_hidden[{}] len {} != hidden_size {}",
            cli.layer, post_ple_want.len(), hidden_size
        );
    }

    // HF's `capture_outputs(tie_last_hidden_states=True)` overwrites
    // `hidden_states[-1]` with the post-final-norm `last_hidden_state`, so
    // `prefill_per_layer_hidden[N-1]` is not pre-norm like layers 0..N-2 —
    // it's already been passed through `model.norm`. Apply final norm to
    // our output before comparing so the stages line up.
    let is_last_layer = cli.layer + 1 == tcfg.num_hidden_layers;
    let (post_ple_got, compare_tag): (Vec<f32>, &str) = if is_last_layer {
        let norm_w = loader.load_bf16_to_gpu(&format!("{weight_prefix}.norm.weight"))?;
        let in_gpu = upload_bf16(&[hidden_size], &final_post_ple)?;
        let mut out_gpu = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut out_gpu, &in_gpu, Some(&norm_w), eps, hidden_size)?;
        (download_bf16(&out_gpu)?, "post-PLE+final-norm")
    } else {
        (final_post_ple.clone(), "post-PLE")
    };
    let post_ple_stats = compare_vectors(&post_ple_got, &post_ple_want)?;
    println!(
        "[layer{} {}] cos_sim={:.6}  max_abs={:.6}  rel_err_norm={:.6} (layer_scalar={:.4})",
        cli.layer, compare_tag,
        post_ple_stats.cos_sim, post_ple_stats.max_abs,
        post_ple_stats.rel_err_norm, final_layer_scalar,
    );
    let first: Vec<String> = post_ple_got.iter().take(6).map(|v| format!("{v:+.4}")).collect();
    let want_first: Vec<String> = post_ple_want.iter().take(6).map(|v| format!("{v:+.4}")).collect();
    println!("  got[..6]  = [{}]", first.join(", "));
    println!("  want[..6] = [{}]", want_first.join(", "));

    if post_ple_stats.cos_sim < 0.999 {
        bail!(
            "post-PLE cosine similarity {:.6} below acceptance threshold 0.999",
            post_ple_stats.cos_sim
        );
    }

    // --- Step 10: logits check (only at the last layer) ---
    // lm_head is tied to embed_tokens on Gemma 4 E2B (tie_word_embeddings=true);
    // there is no separate lm_head.weight tensor. Reuse post_ple_got (post-
    // final-norm hidden at the last prompt position) as the matvec input.
    // HF applies `final_logit_softcapping` (30.0) inside the model forward, so
    // the oracle's `prefill_logits` are already softcapped — we must mirror
    // that on the host after downloading the raw matvec result.
    if is_last_layer {
        if !cli.chain {
            println!("[logits] skipped (last-layer logit check requires --chain for an end-to-end input)");
        } else {
            let vocab_size = tcfg.vocab_size;
            let lm_head_w =
                loader.load_bf16_to_gpu(&format!("{weight_prefix}.embed_tokens.weight"))?;
            let normed_hidden = upload_bf16(&[hidden_size], &post_ple_got)?;
            let mut logits_gpu = GpuBuffer::zeros(0, dtype, &[vocab_size])?;
            g4::matvec(
                0, dtype, &mut logits_gpu, &normed_hidden, &lm_head_w,
                hidden_size, vocab_size, &mut counter,
            )?;
            let mut logits_host = download_bf16(&logits_gpu)?;
            let cap = tcfg.final_logit_softcapping.unwrap_or(30.0) as f32;
            for v in logits_host.iter_mut() {
                *v = cap * (*v / cap).tanh();
            }

            if oracle.prefill_logits.len() != vocab_size {
                bail!(
                    "oracle.prefill_logits len {} != vocab_size {}",
                    oracle.prefill_logits.len(), vocab_size
                );
            }
            let logit_stats = compare_vectors(&logits_host, &oracle.prefill_logits)?;

            let argmax_got = argmax(&logits_host);
            let argmax_want = argmax(&oracle.prefill_logits);
            let top5_got = top_k_indices(&logits_host, 5);
            let top5_want = top_k_indices(&oracle.prefill_logits, 5);
            let top5_overlap = top5_got.iter().filter(|i| top5_want.contains(*i)).count();

            println!(
                "[logits] cos_sim={:.6}  max_abs={:.6}  rel_err_norm={:.6}  softcap={}",
                logit_stats.cos_sim, logit_stats.max_abs, logit_stats.rel_err_norm, cap
            );
            let got_val = logits_host[argmax_got];
            let want_val = oracle.prefill_logits[argmax_want];
            println!(
                "  argmax got={} ({:+.4})  want={} ({:+.4})  {}",
                argmax_got, got_val, argmax_want, want_val,
                if argmax_got == argmax_want { "MATCH" } else { "MISMATCH" }
            );
            println!(
                "  top5 got={:?}  want={:?}  overlap={}/5",
                top5_got, top5_want, top5_overlap
            );

            if logit_stats.cos_sim < 0.999 {
                bail!(
                    "logit cosine similarity {:.6} below acceptance threshold 0.999",
                    logit_stats.cos_sim
                );
            }
            if argmax_got != argmax_want {
                bail!(
                    "argmax mismatch: got {} want {}",
                    argmax_got, argmax_want
                );
            }
        }
    }

    println!("PASS");
    Ok(())
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
