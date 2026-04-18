//! Fully self-contained end-to-end validator for Gemma 4 E2B.
//!
//! Unlike `gemma4_decode_validate` (which seeds per-layer K/V caches from the
//! oracle's prefill dump before running any decode steps), this binary runs
//! the prompt tokens through the same Rust primitive kernels position by
//! position (Phase A), building K/V caches from scratch. After the last
//! prompt token it applies `final_norm` + tied `lm_head` + softcap and
//! compares the logits against `oracle.prefill_logits`. Then Phase B runs
//! greedy decode steps, again reusing the same kernels, and compares each
//! step against `oracle.decode_logits[step]` and
//! `oracle.generated_token_ids[step+1]`.
//!
//! Usage:
//!   cargo run --release --bin gemma4_e2e_validate -- \
//!     --model-dir <checkpoint> --oracle-json <oracle_state.json> \
//!     [--max-new-tokens N]
//!
//! Acceptance:
//!   * Phase A prefill logits cos_sim >= 0.999, argmax == generated_token_ids[0]
//!   * Phase B step k vs decode_logits[k] cos_sim >= 0.999,
//!     argmax == generated_token_ids[k+1]
//!   * generated_ids_rust == oracle.generated_token_ids[0..max_new_tokens]
//!
//! The oracle is still consulted for validation but nothing it produces is
//! fed into the runtime state of the forward pass — the only things copied
//! out of the oracle JSON are the prompt token IDs and the target logits.

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
#[command(about = "Fully self-contained Gemma 4 E2B forward-pass validator (Rust prefill + Rust decode)")]
struct Cli {
    /// Path to a local Gemma 4 checkpoint directory (config.json + safetensors).
    #[arg(long)]
    model_dir: PathBuf,
    /// Oracle JSON file produced by `oracle/gemma4_oracle.py --emit-state`.
    #[arg(long)]
    oracle_json: PathBuf,
    /// Total number of generated output tokens to verify. Phase A produces
    /// token 0 from the prompt's last-position logits; Phase B produces the
    /// remaining `max_new_tokens - 1` tokens greedily. Must not exceed the
    /// oracle's `generated_tokens`.
    #[arg(long, default_value_t = 4)]
    max_new_tokens: usize,
    /// Skip the optional per-layer K/V cache sanity check against
    /// `oracle.kv_caches` at the end of Phase A.
    #[arg(long, default_value_t = false)]
    skip_kv_check: bool,
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

/// Compute `per_layer_inputs` for a single input token (see
/// `gemma4_decode_validate.rs` for formula derivation).
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

/// Replicate a contiguous range of KV-cache slots from `src` to `dst`. Layout
/// is `[num_kv_heads, max_t, head_dim]`, so for each kv_head the `count`
/// consecutive slots starting at `pos_base` are contiguous in memory
/// (`count * head_dim * 2` bytes). Used by the prefill path to copy all
/// prompt-token K/V slots into shared-KV layers in one pass per head.
fn copy_kv_slots_range(
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
    num_kv_heads: usize,
    max_t: usize,
    head_dim: usize,
    pos_base: usize,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let elem_bytes = 2usize; // BF16
    let bytes_per_head = count * head_dim * elem_bytes;
    for h in 0..num_kv_heads {
        let byte_off = ((h * max_t) + pos_base) * head_dim * elem_bytes;
        let src_ptr = src.offset_ptr(byte_off);
        let dst_ptr =
            unsafe { (dst.as_mut_ptr() as *mut u8).add(byte_off) as *mut c_void };
        gpu_hal::copy_d2d(0, dst_ptr, src_ptr, bytes_per_head)
            .map_err(|e| anyhow!("copy_kv_slots_range: {e}"))?;
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

struct LayerWeights {
    kind: AttnKind,
    head_dim: usize,
    intermediate_size: usize,
    shared_kv: bool,
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

/// Immutable state shared across every forward-pass call.
struct ForwardCtx<'a> {
    tcfg: &'a TextConfig,
    loader: &'a UnbakedLoader,
    weight_prefix: &'a str,
    layers: &'a [LayerWeights],
    per_layer_model_projection_w: &'a GpuBuffer,
    per_layer_projection_norm_w: &'a GpuBuffer,
    final_norm_w: &'a GpuBuffer,
    lm_head_w: &'a GpuBuffer,
    sliding_cos: &'a GpuBuffer,
    sliding_sin: &'a GpuBuffer,
    full_cos: &'a GpuBuffer,
    full_sin: &'a GpuBuffer,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    eps: f32,
    ple_hidden: usize,
    num_layers: usize,
    vocab_size: usize,
    cap: f32,
    max_t: usize,
}

/// Runs one decode-style forward pass for a single (pos, token_id) pair: look
/// up the scaled embedding, chain through all 35 layers (with kv_append +
/// shared-KV replication at slot `pos`), optionally apply final norm +
/// lm_head + softcap and return host-side logits.
///
/// Layer body is identical to `gemma4_decode_validate`'s step loop — kept in
/// one place so Phase A and Phase B use the exact same code path.
fn run_forward_pass(
    ctx: &ForwardCtx,
    input_token_id: u32,
    pos: usize,
    k_caches: &mut [GpuBuffer],
    v_caches: &mut [GpuBuffer],
    counter: &mut GpuBuffer,
    compute_logits: bool,
) -> Result<Option<Vec<f32>>> {
    let dtype = ScalarType::BF16;
    let hidden_size = ctx.hidden_size;
    let num_q_heads = ctx.num_q_heads;
    let num_kv_heads = ctx.num_kv_heads;
    let eps = ctx.eps;
    let ple_hidden = ctx.ple_hidden;
    let num_layers = ctx.num_layers;
    let max_t = ctx.max_t;

    let h_in_host = load_scaled_embed_row(
        ctx.loader,
        &format!("{}.embed_tokens.weight", ctx.weight_prefix),
        input_token_id,
        hidden_size,
    )?;
    let mut h_running = upload_bf16(&[hidden_size], &h_in_host)?;

    let pli_bytes = compute_per_layer_inputs(
        ctx.loader,
        ctx.weight_prefix,
        ctx.tcfg,
        input_token_id,
        ctx.per_layer_model_projection_w,
        ctx.per_layer_projection_norm_w,
        counter,
    )?;
    let expected_pli_bytes = num_layers * ple_hidden * 2;
    if pli_bytes.len() != expected_pli_bytes {
        bail!(
            "compute_per_layer_inputs returned {} bytes, expected {}",
            pli_bytes.len(),
            expected_pli_bytes
        );
    }

    for layer_idx in 0..num_layers {
        let w = &ctx.layers[layer_idx];
        let head_dim = w.head_dim;
        let rotary_dim = head_dim;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let sliding_window = match w.kind {
            AttnKind::Sliding => ctx.tcfg.sliding_window as i32,
            AttnKind::Full => 0,
        };
        let (cos_table, sin_table) = match w.kind {
            AttnKind::Sliding => (ctx.sliding_cos, ctx.sliding_sin),
            AttnKind::Full => (ctx.full_cos, ctx.full_sin),
        };

        let residual = h_running.clone_device()?;

        let mut x = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x, &h_running, Some(&w.input_norm), eps, hidden_size)?;

        let mut q = GpuBuffer::zeros(0, dtype, &[num_q_heads, head_dim])?;
        g4::matvec(0, dtype, &mut q, &x, &w.q_proj, hidden_size, q_dim, counter)?;

        let mut q_normed = GpuBuffer::zeros(0, dtype, &[num_q_heads, head_dim])?;
        g4::rms_norm_per_row(0, dtype, &mut q_normed, &q, Some(&w.q_norm), eps, num_q_heads, head_dim)?;
        g4::rope_decode(0, dtype, &mut q_normed, cos_table, sin_table, num_q_heads, head_dim, rotary_dim, pos)?;

        if !w.shared_kv {
            let k_proj = w.k_proj.as_ref().expect("k_proj on non-shared layer");
            let v_proj = w.v_proj.as_ref().expect("v_proj on non-shared layer");
            let k_norm = w.k_norm.as_ref().expect("k_norm on non-shared layer");

            let mut k = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::matvec(0, dtype, &mut k, &x, k_proj, hidden_size, kv_dim, counter)?;
            let mut v = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::matvec(0, dtype, &mut v, &x, v_proj, hidden_size, kv_dim, counter)?;

            let mut k_normed = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::rms_norm_per_row(0, dtype, &mut k_normed, &k, Some(k_norm), eps, num_kv_heads, head_dim)?;
            let mut v_normed = GpuBuffer::zeros(0, dtype, &[num_kv_heads, head_dim])?;
            g4::rms_norm_per_row(0, dtype, &mut v_normed, &v, None, eps, num_kv_heads, head_dim)?;

            g4::rope_decode(0, dtype, &mut k_normed, cos_table, sin_table, num_kv_heads, head_dim, rotary_dim, pos)?;

            g4::kv_append(
                0, dtype, &k_normed, &v_normed,
                &mut k_caches[layer_idx], &mut v_caches[layer_idx],
                num_kv_heads, head_dim, pos, max_t,
            )?;

            for shared_layer in (layer_idx + 1)..num_layers {
                let s = &ctx.layers[shared_layer];
                if s.shared_kv && s.kv_source == layer_idx {
                    let (lo, hi) = k_caches.split_at_mut(shared_layer);
                    copy_kv_slot(&lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                    let (lo, hi) = v_caches.split_at_mut(shared_layer);
                    copy_kv_slot(&lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                }
            }
        }

        let kv_len = pos + 1;
        let mut attn_out = GpuBuffer::zeros(0, dtype, &[num_q_heads, head_dim])?;
        let mut scores = GpuBuffer::zeros(0, ScalarType::F32, &[num_q_heads, max_t])?;
        g4::swa_attn_decode(
            0, dtype, &q_normed, &k_caches[layer_idx], &v_caches[layer_idx],
            &mut scores, &mut attn_out,
            num_q_heads, num_kv_heads, head_dim, kv_len, max_t, sliding_window, 1.0,
        )?;

        let attn_flat = {
            let bytes = attn_out.to_host_bytes()?;
            GpuBuffer::from_host_bytes(0, dtype, &[q_dim], &bytes)?
        };
        let mut o = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::matvec(0, dtype, &mut o, &attn_flat, &w.o_proj, q_dim, hidden_size, counter)?;

        let mut x2 = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x2, &o, Some(&w.post_attn_norm), eps, hidden_size)?;
        let residual_h = download_bf16(&residual)?;
        let x2_h = download_bf16(&x2)?;
        let h1_h: Vec<f32> = residual_h.iter().zip(x2_h.iter()).map(|(a, b)| a + b).collect();
        let h_mid = upload_bf16(&[hidden_size], &h1_h)?;

        let residual2 = h_mid.clone_device()?;

        let mut x3 = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x3, &h_mid, Some(&w.pre_ff_norm), eps, hidden_size)?;

        let mut gate = GpuBuffer::zeros(0, dtype, &[w.intermediate_size])?;
        g4::matvec(0, dtype, &mut gate, &x3, &w.gate_proj, hidden_size, w.intermediate_size, counter)?;
        let mut up_buf = GpuBuffer::zeros(0, dtype, &[w.intermediate_size])?;
        g4::matvec(0, dtype, &mut up_buf, &x3, &w.up_proj, hidden_size, w.intermediate_size, counter)?;
        let mut y = GpuBuffer::zeros(0, dtype, &[w.intermediate_size])?;
        g4::gelu_tanh_gate_mul(0, dtype, &mut y, &gate, &up_buf, w.intermediate_size)?;

        let mut m = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::matvec(0, dtype, &mut m, &y, &w.down_proj, w.intermediate_size, hidden_size, counter)?;

        let mut x4 = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(0, dtype, &mut x4, &m, Some(&w.post_ff_norm), eps, hidden_size)?;
        let residual2_h = download_bf16(&residual2)?;
        let x4_h = download_bf16(&x4)?;
        let h_pre_ple: Vec<f32> =
            residual2_h.iter().zip(x4_h.iter()).map(|(a, b)| a + b).collect();

        let bytes_per_layer = ple_hidden * 2;
        let pli_off = layer_idx * bytes_per_layer;
        let pli_slice = &pli_bytes[pli_off..pli_off + bytes_per_layer];
        let per_layer_input_f32 = bf16_bytes_to_f32(pli_slice);
        let per_layer_input_gpu = upload_bf16(&[ple_hidden], &per_layer_input_f32)?;

        let ple_residual = upload_bf16(&[hidden_size], &h_pre_ple)?;
        let h_in_ple = ple_residual.clone_device()?;

        let mut gated = GpuBuffer::zeros(0, dtype, &[ple_hidden])?;
        g4::matvec(
            0, dtype, &mut gated, &h_in_ple, &w.per_layer_input_gate_w,
            hidden_size, ple_hidden, counter,
        )?;
        let mut gated_act = GpuBuffer::zeros(0, dtype, &[ple_hidden])?;
        g4::gelu_tanh_gate_mul(
            0, dtype, &mut gated_act, &gated, &per_layer_input_gpu, ple_hidden,
        )?;
        let mut projected = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::matvec(
            0, dtype, &mut projected, &gated_act, &w.per_layer_projection_w,
            ple_hidden, hidden_size, counter,
        )?;
        let mut normed = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        g4::rms_norm(
            0, dtype, &mut normed, &projected, Some(&w.post_per_layer_input_norm_w),
            eps, hidden_size,
        )?;
        let ple_residual_h = download_bf16(&ple_residual)?;
        let normed_h = download_bf16(&normed)?;
        let h_post_ple: Vec<f32> = ple_residual_h
            .iter()
            .zip(normed_h.iter())
            .map(|(a, b)| (a + b) * w.layer_scalar)
            .collect();

        h_running = upload_bf16(&[hidden_size], &h_post_ple)?;
    }

    if !compute_logits {
        return Ok(None);
    }

    let mut post_norm = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
    g4::rms_norm(0, dtype, &mut post_norm, &h_running, Some(ctx.final_norm_w), eps, hidden_size)?;

    let mut logits_gpu = GpuBuffer::zeros(0, dtype, &[ctx.vocab_size])?;
    g4::matvec(
        0, dtype, &mut logits_gpu, &post_norm, ctx.lm_head_w,
        hidden_size, ctx.vocab_size, counter,
    )?;
    let mut logits_host = download_bf16(&logits_gpu)?;
    let cap = ctx.cap;
    for v in logits_host.iter_mut() {
        *v = cap * (*v / cap).tanh();
    }
    Ok(Some(logits_host))
}

/// Gather `embed_tokens_per_layer` rows for a batch of tokens and apply the
/// `sqrt(ple_hidden)` scale on the host side (matches HF's
/// `embed_tokens_per_layer[tok] * sqrt(hidden_size_per_layer_input)` cast).
/// Returns a contiguous `[seq_len * row_dim]` f32 vector, ready to convert
/// to BF16 and upload as a `[seq_len, row_dim]` device tensor.
fn gather_ple_raw_batch(
    loader: &UnbakedLoader,
    weight_name: &str,
    token_ids: &[u32],
    row_dim: usize,
    ple_hidden: usize,
) -> Result<Vec<f32>> {
    let (shape, bytes) = loader.tensor_bytes(weight_name)?;
    if shape.len() != 2 || shape[1] != row_dim {
        bail!(
            "{weight_name} shape {:?} does not match expected [vocab, {row_dim}]",
            shape
        );
    }
    let vocab = shape[0];
    let row_bytes = row_dim * 2;
    let scale = (ple_hidden as f32).sqrt();
    let mut out = Vec::with_capacity(token_ids.len() * row_dim);
    for &tok in token_ids {
        if (tok as usize) >= vocab {
            bail!("token id {tok} out of range (vocab={vocab})");
        }
        let off = tok as usize * row_bytes;
        let row = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
        out.extend(row.iter().map(|v| v * scale));
    }
    Ok(out)
}

/// Compute per-layer inputs for the entire prompt in one shot.
/// Produces a device tensor of shape `[seq_len, num_layers, ple_hidden]` where
/// `pli[s, l, :]` is the per-layer-input for prompt token `s` at layer `l`.
/// Mirrors HF's `Gemma4TextModel.{get,project}_per_layer_inputs` but uses
/// batched primitives and one host-side gather of `embed_tokens_per_layer`
/// (table is too large to upload in full).
fn compute_per_layer_inputs_batched(
    ctx: &ForwardCtx,
    token_ids_gpu: &GpuBuffer,
    prompt_token_ids: &[u32],
    counter: &mut GpuBuffer,
) -> Result<GpuBuffer> {
    let dtype = ScalarType::BF16;
    let seq_len = prompt_token_ids.len();
    let hidden_size = ctx.hidden_size;
    let num_layers = ctx.num_layers;
    let ple_hidden = ctx.ple_hidden;
    let vocab_size = ctx.vocab_size;
    let eps = ctx.eps;
    let total = num_layers * ple_hidden;

    // 1) main_embed[s, :] = embed_tokens[tok_s] * sqrt(hidden)  (BF16-rounded scale).
    let embed_scale = bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
    let mut main_embed_batch = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
    g4::embed_gather_scaled(
        0, dtype, &mut main_embed_batch, token_ids_gpu, ctx.lm_head_w,
        seq_len, hidden_size, vocab_size, embed_scale,
    )?;

    // 2) proj[s, :] = per_layer_model_projection @ main_embed[s, :]  → [S, total]
    let mut proj = GpuBuffer::zeros(0, dtype, &[seq_len, total])?;
    g4::matvec_batched(
        0, dtype, &mut proj, &main_embed_batch, ctx.per_layer_model_projection_w,
        seq_len, hidden_size, total, counter,
    )?;

    // 3) proj *= hidden^-0.5  (BF16-rounded scale — matches HF Python-float * tensor).
    let proj_scale = bf16::from_f32((hidden_size as f32).powf(-0.5)).to_f32();
    g4::scalar_mul_inplace(0, dtype, &mut proj, proj_scale, seq_len * total)?;

    // 4) rms_norm over last dim: view [S, total] as [S * num_layers, ple_hidden].
    let mut proj_normed = GpuBuffer::zeros(0, dtype, &[seq_len, num_layers, ple_hidden])?;
    g4::rms_norm_rows(
        0, dtype, &mut proj_normed, &proj, Some(ctx.per_layer_projection_norm_w),
        eps, seq_len * num_layers, ple_hidden,
    )?;

    // 5) ple_raw[s, l, :] = embed_tokens_per_layer[tok_s, l*ple_hidden..(l+1)*ple_hidden]
    //    * sqrt(ple_hidden)  (host-side gather; full 4.6GB table never uploaded).
    let ple_raw_host = gather_ple_raw_batch(
        ctx.loader,
        &format!("{}.embed_tokens_per_layer.weight", ctx.weight_prefix),
        prompt_token_ids,
        total,
        ple_hidden,
    )?;
    let ple_raw_gpu = upload_bf16(&[seq_len, num_layers, ple_hidden], &ple_raw_host)?;

    // 6) pli = (proj_normed + ple_raw) * 2^-0.5  (BF16-rounded scale).
    let combine_scale = bf16::from_f32(2.0f32.powf(-0.5)).to_f32();
    let mut pli = GpuBuffer::zeros(0, dtype, &[seq_len, num_layers, ple_hidden])?;
    g4::add_scaled_residual(
        0, dtype, &mut pli, &proj_normed, &ple_raw_gpu,
        combine_scale, seq_len * num_layers * ple_hidden,
    )?;

    Ok(pli)
}

/// Single-launch-per-primitive prefill forward pass over the whole prompt.
/// Replaces Phase A's per-position `run_forward_pass` loop with batched
/// kernels that process all `seq_len` tokens in parallel inside each
/// primitive. Returns the softcapped logits at the last prompt position.
fn run_prefill(
    ctx: &ForwardCtx,
    prompt_token_ids: &[u32],
    k_caches: &mut [GpuBuffer],
    v_caches: &mut [GpuBuffer],
    counter: &mut GpuBuffer,
) -> Result<Vec<f32>> {
    let dtype = ScalarType::BF16;
    let seq_len = prompt_token_ids.len();
    let hidden_size = ctx.hidden_size;
    let num_q_heads = ctx.num_q_heads;
    let num_kv_heads = ctx.num_kv_heads;
    let eps = ctx.eps;
    let ple_hidden = ctx.ple_hidden;
    let num_layers = ctx.num_layers;
    let max_t = ctx.max_t;
    let vocab_size = ctx.vocab_size;

    if seq_len == 0 {
        bail!("run_prefill: seq_len must be > 0");
    }

    // Upload prompt token IDs once; reused for both main-embed and PLE gathers.
    let mut id_bytes: Vec<u8> = Vec::with_capacity(seq_len * 4);
    for &id in prompt_token_ids {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    let token_ids_gpu =
        GpuBuffer::from_host_bytes(0, ScalarType::U32, &[seq_len], &id_bytes)?;

    // h_running[s, :] = embed_tokens[tok_s] * sqrt(hidden)  (starting layer input).
    let embed_scale = bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
    let mut h_running = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
    g4::embed_gather_scaled(
        0, dtype, &mut h_running, &token_ids_gpu, ctx.lm_head_w,
        seq_len, hidden_size, vocab_size, embed_scale,
    )?;

    // Per-layer inputs for the whole prompt.
    let pli = compute_per_layer_inputs_batched(ctx, &token_ids_gpu, prompt_token_ids, counter)?;

    for layer_idx in 0..num_layers {
        let w = &ctx.layers[layer_idx];
        let head_dim = w.head_dim;
        let rotary_dim = head_dim;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let sliding_window = match w.kind {
            AttnKind::Sliding => ctx.tcfg.sliding_window as i32,
            AttnKind::Full => 0,
        };
        let (cos_table, sin_table) = match w.kind {
            AttnKind::Sliding => (ctx.sliding_cos, ctx.sliding_sin),
            AttnKind::Full => (ctx.full_cos, ctx.full_sin),
        };

        let residual = h_running.clone_device()?;

        let mut x = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::rms_norm_rows(
            0, dtype, &mut x, &h_running, Some(&w.input_norm),
            eps, seq_len, hidden_size,
        )?;

        let mut q = GpuBuffer::zeros(0, dtype, &[seq_len, num_q_heads, head_dim])?;
        g4::matvec_batched(
            0, dtype, &mut q, &x, &w.q_proj,
            seq_len, hidden_size, q_dim, counter,
        )?;
        let mut q_normed = GpuBuffer::zeros(0, dtype, &[seq_len, num_q_heads, head_dim])?;
        g4::rms_norm_rows(
            0, dtype, &mut q_normed, &q, Some(&w.q_norm),
            eps, seq_len * num_q_heads, head_dim,
        )?;
        g4::rope_prefill(
            0, dtype, &mut q_normed, cos_table, sin_table,
            seq_len, num_q_heads, head_dim, rotary_dim, 0,
        )?;

        if !w.shared_kv {
            let k_proj = w.k_proj.as_ref().expect("k_proj on non-shared layer");
            let v_proj = w.v_proj.as_ref().expect("v_proj on non-shared layer");
            let k_norm = w.k_norm.as_ref().expect("k_norm on non-shared layer");

            let mut k = GpuBuffer::zeros(0, dtype, &[seq_len, num_kv_heads, head_dim])?;
            g4::matvec_batched(
                0, dtype, &mut k, &x, k_proj,
                seq_len, hidden_size, kv_dim, counter,
            )?;
            let mut v = GpuBuffer::zeros(0, dtype, &[seq_len, num_kv_heads, head_dim])?;
            g4::matvec_batched(
                0, dtype, &mut v, &x, v_proj,
                seq_len, hidden_size, kv_dim, counter,
            )?;

            let mut k_normed = GpuBuffer::zeros(0, dtype, &[seq_len, num_kv_heads, head_dim])?;
            g4::rms_norm_rows(
                0, dtype, &mut k_normed, &k, Some(k_norm),
                eps, seq_len * num_kv_heads, head_dim,
            )?;
            let mut v_normed = GpuBuffer::zeros(0, dtype, &[seq_len, num_kv_heads, head_dim])?;
            g4::rms_norm_rows(
                0, dtype, &mut v_normed, &v, None,
                eps, seq_len * num_kv_heads, head_dim,
            )?;

            g4::rope_prefill(
                0, dtype, &mut k_normed, cos_table, sin_table,
                seq_len, num_kv_heads, head_dim, rotary_dim, 0,
            )?;

            g4::kv_append_prefill(
                0, dtype, &k_normed, &v_normed,
                &mut k_caches[layer_idx], &mut v_caches[layer_idx],
                seq_len, num_kv_heads, head_dim, 0, max_t,
            )?;

            // Replicate to layers that share this one's KV. The shared layers'
            // indices are monotonically larger than `layer_idx`, so the
            // replicated slots will be in place before the dependent layer's
            // attention reads.
            for shared_layer in (layer_idx + 1)..num_layers {
                let s = &ctx.layers[shared_layer];
                if s.shared_kv && s.kv_source == layer_idx {
                    let (lo, hi) = k_caches.split_at_mut(shared_layer);
                    copy_kv_slots_range(
                        &lo[layer_idx], &mut hi[0],
                        num_kv_heads, max_t, head_dim, 0, seq_len,
                    )?;
                    let (lo, hi) = v_caches.split_at_mut(shared_layer);
                    copy_kv_slots_range(
                        &lo[layer_idx], &mut hi[0],
                        num_kv_heads, max_t, head_dim, 0, seq_len,
                    )?;
                }
            }
        }

        let mut attn_out = GpuBuffer::zeros(0, dtype, &[seq_len, num_q_heads, head_dim])?;
        let mut scores = GpuBuffer::zeros(0, ScalarType::F32, &[seq_len, num_q_heads, max_t])?;
        g4::attn_prefill(
            0, dtype, &q_normed, &k_caches[layer_idx], &v_caches[layer_idx],
            &mut scores, &mut attn_out,
            seq_len, num_q_heads, num_kv_heads, head_dim, 0, max_t,
            sliding_window, 1.0,
        )?;

        let mut o = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::matvec_batched(
            0, dtype, &mut o, &attn_out, &w.o_proj,
            seq_len, q_dim, hidden_size, counter,
        )?;

        let mut x2 = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::rms_norm_rows(
            0, dtype, &mut x2, &o, Some(&w.post_attn_norm),
            eps, seq_len, hidden_size,
        )?;
        let mut h_mid = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::add_residual(
            0, dtype, &mut h_mid, &residual, &x2, seq_len * hidden_size,
        )?;

        let residual2 = h_mid.clone_device()?;

        let mut x3 = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::rms_norm_rows(
            0, dtype, &mut x3, &h_mid, Some(&w.pre_ff_norm),
            eps, seq_len, hidden_size,
        )?;

        let mut gate = GpuBuffer::zeros(0, dtype, &[seq_len, w.intermediate_size])?;
        g4::matvec_batched(
            0, dtype, &mut gate, &x3, &w.gate_proj,
            seq_len, hidden_size, w.intermediate_size, counter,
        )?;
        let mut up_buf = GpuBuffer::zeros(0, dtype, &[seq_len, w.intermediate_size])?;
        g4::matvec_batched(
            0, dtype, &mut up_buf, &x3, &w.up_proj,
            seq_len, hidden_size, w.intermediate_size, counter,
        )?;
        let mut y = GpuBuffer::zeros(0, dtype, &[seq_len, w.intermediate_size])?;
        g4::gelu_tanh_gate_mul(
            0, dtype, &mut y, &gate, &up_buf, seq_len * w.intermediate_size,
        )?;

        let mut m = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::matvec_batched(
            0, dtype, &mut m, &y, &w.down_proj,
            seq_len, w.intermediate_size, hidden_size, counter,
        )?;

        let mut x4 = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::rms_norm_rows(
            0, dtype, &mut x4, &m, Some(&w.post_ff_norm),
            eps, seq_len, hidden_size,
        )?;
        let mut h_pre_ple = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::add_residual(
            0, dtype, &mut h_pre_ple, &residual2, &x4, seq_len * hidden_size,
        )?;

        // PLE branch: extract this layer's PLI slice, then apply the
        // gate-project-norm-residual chain across all S tokens.
        let mut pli_slice = GpuBuffer::zeros(0, dtype, &[seq_len, ple_hidden])?;
        g4::gather_layer_slice(
            0, dtype, &mut pli_slice, &pli,
            seq_len, num_layers, ple_hidden, layer_idx,
        )?;

        let mut gated = GpuBuffer::zeros(0, dtype, &[seq_len, ple_hidden])?;
        g4::matvec_batched(
            0, dtype, &mut gated, &h_pre_ple, &w.per_layer_input_gate_w,
            seq_len, hidden_size, ple_hidden, counter,
        )?;
        let mut gated_act = GpuBuffer::zeros(0, dtype, &[seq_len, ple_hidden])?;
        g4::gelu_tanh_gate_mul(
            0, dtype, &mut gated_act, &gated, &pli_slice, seq_len * ple_hidden,
        )?;
        let mut projected = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::matvec_batched(
            0, dtype, &mut projected, &gated_act, &w.per_layer_projection_w,
            seq_len, ple_hidden, hidden_size, counter,
        )?;
        let mut normed = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::rms_norm_rows(
            0, dtype, &mut normed, &projected, Some(&w.post_per_layer_input_norm_w),
            eps, seq_len, hidden_size,
        )?;
        let mut h_new = GpuBuffer::zeros(0, dtype, &[seq_len, hidden_size])?;
        g4::add_scaled_residual(
            0, dtype, &mut h_new, &h_pre_ple, &normed,
            w.layer_scalar, seq_len * hidden_size,
        )?;
        h_running = h_new;
    }

    // Final norm + lm_head + softcap on the last-position hidden only.
    let last_byte_off = (seq_len - 1) * hidden_size * 2;
    let mut last_hidden = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
    unsafe {
        let src_ptr = h_running.offset_ptr(last_byte_off);
        gpu_hal::copy_d2d(0, last_hidden.as_mut_ptr(), src_ptr, hidden_size * 2)
            .map_err(|e| anyhow!("copy last hidden: {e}"))?;
    }

    let mut post_norm = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
    g4::rms_norm(
        0, dtype, &mut post_norm, &last_hidden, Some(ctx.final_norm_w),
        eps, hidden_size,
    )?;
    let mut logits_gpu = GpuBuffer::zeros(0, dtype, &[vocab_size])?;
    g4::matvec(
        0, dtype, &mut logits_gpu, &post_norm, ctx.lm_head_w,
        hidden_size, vocab_size, counter,
    )?;
    let mut logits_host = download_bf16(&logits_gpu)?;
    let cap = ctx.cap;
    for v in logits_host.iter_mut() {
        *v = cap * (*v / cap).tanh();
    }
    Ok(logits_host)
}

/// Reshape our K/V cache (shape `[num_kv_heads, max_t, head_dim]`, first
/// `prompt_tokens` slots filled by Phase A) into a flat vector matching the
/// oracle's `[1, num_kv_heads, prompt_tokens, head_dim]` ordering, so the two
/// can be compared element-wise with `compare_vectors`.
fn extract_filled_kv_to_oracle_layout(
    cache: &GpuBuffer,
    num_kv_heads: usize,
    max_t: usize,
    prompt_tokens: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let host = cache.to_host_bytes()?;
    let full = bf16_bytes_to_f32(&host);
    let expected = num_kv_heads * max_t * head_dim;
    if full.len() != expected {
        bail!(
            "cache len {} != expected {} (layout [{num_kv_heads}, {max_t}, {head_dim}])",
            full.len(), expected
        );
    }
    let mut out = vec![0.0f32; num_kv_heads * prompt_tokens * head_dim];
    for h in 0..num_kv_heads {
        for t in 0..prompt_tokens {
            let src_off = (h * max_t + t) * head_dim;
            let dst_off = (h * prompt_tokens + t) * head_dim;
            out[dst_off..dst_off + head_dim]
                .copy_from_slice(&full[src_off..src_off + head_dim]);
        }
    }
    Ok(out)
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
    let kv_caches_oracle = oracle
        .kv_caches
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing kv_caches (need --emit-state)"))?;

    if prompt_tokens == 0 {
        bail!("prompt_tokens == 0");
    }
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
    // Phase B runs `max_new_tokens - 1` steps (Phase A covers the first token).
    // That needs `decode_logits[0..max_new_tokens-1]` and
    // `generated_token_ids[0..max_new_tokens]`.
    if max_new_tokens >= 2 && oracle.decode_logits.len() < max_new_tokens - 1 {
        bail!(
            "oracle decode_logits has {} entries, need at least {}",
            oracle.decode_logits.len(),
            max_new_tokens - 1
        );
    }
    if oracle.generated_token_ids.len() < max_new_tokens {
        bail!(
            "oracle generated_token_ids has {} entries, need at least {}",
            oracle.generated_token_ids.len(),
            max_new_tokens
        );
    }

    let max_t = prompt_tokens + max_new_tokens;

    let loader = UnbakedLoader::open(&cli.model_dir)?;
    let weight_prefix = "model.language_model";
    let mut counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;

    println!(
        "[cfg] prompt_tokens={prompt_tokens} prompt_token_ids={:?} \
         max_new_tokens={max_new_tokens} max_t={max_t} num_layers={num_layers} \
         hidden={hidden_size} ple_hidden={ple_hidden}",
        prompt_token_ids,
    );

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(load_layer_weights(&loader, tcfg, weight_prefix, i)?);
    }

    let lm_head_w = loader.load_bf16_to_gpu(&format!("{weight_prefix}.embed_tokens.weight"))?;
    let final_norm_w = loader.load_bf16_to_gpu(&format!("{weight_prefix}.norm.weight"))?;
    let per_layer_model_projection_w =
        loader.load_bf16_to_gpu(&format!("{weight_prefix}.per_layer_model_projection.weight"))?;
    let per_layer_projection_norm_w =
        loader.load_bf16_to_gpu(&format!("{weight_prefix}.per_layer_projection_norm.weight"))?;

    // Zero-initialize K/V caches. Phase A fills slots 0..prompt_tokens; Phase B
    // fills slots prompt_tokens..prompt_tokens+max_new_tokens-1.
    let mut k_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
    let mut v_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let hd = layers[l].head_dim;
        k_caches.push(GpuBuffer::zeros(0, dtype, &[num_kv_heads, max_t, hd])?);
        v_caches.push(GpuBuffer::zeros(0, dtype, &[num_kv_heads, max_t, hd])?);
    }

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

    let ctx = ForwardCtx {
        tcfg,
        loader: &loader,
        weight_prefix,
        layers: &layers,
        per_layer_model_projection_w: &per_layer_model_projection_w,
        per_layer_projection_norm_w: &per_layer_projection_norm_w,
        final_norm_w: &final_norm_w,
        lm_head_w: &lm_head_w,
        sliding_cos: &sliding_cos,
        sliding_sin: &sliding_sin,
        full_cos: &full_cos,
        full_sin: &full_sin,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        eps,
        ple_hidden,
        num_layers,
        vocab_size,
        cap,
        max_t,
    };

    let mut overall_pass = true;
    let mut generated_ids_rust: Vec<u32> = Vec::with_capacity(max_new_tokens);

    // ===== PHASE A: batched Rust prefill =====
    //
    // Single invocation of `run_prefill` processes every prompt token through
    // every layer using batched primitives — one kernel launch per sub-op per
    // layer, independent of the prompt length. No Rust-side iteration over
    // positions. Returns the softcapped logits at the last prompt position.
    println!("\n[phase A] batched Rust prefill over {prompt_tokens} prompt tokens");
    let logits = run_prefill(
        &ctx, prompt_token_ids, &mut k_caches, &mut v_caches, &mut counter,
    )?;
    if oracle.prefill_logits.len() != vocab_size {
        bail!(
            "oracle.prefill_logits len {} != vocab_size {}",
            oracle.prefill_logits.len(), vocab_size
        );
    }
    let stats = compare_vectors(&logits, &oracle.prefill_logits)?;
    let got_arg = argmax(&logits);
    let want_arg = argmax(&oracle.prefill_logits);
    let top5_got = top_k_indices(&logits, 5);
    let top5_want = top_k_indices(&oracle.prefill_logits, 5);
    let top5_overlap = top5_got.iter().filter(|i| top5_want.contains(*i)).count();
    let expected_gen_id = oracle.generated_token_ids[0];
    let match_gen = got_arg as u32 == expected_gen_id;
    let match_oracle = got_arg == want_arg;

    println!(
        "[phase A] last-pos vs prefill_logits: cos_sim={:.6} max_abs={:.6} rel_err={:.6}",
        stats.cos_sim, stats.max_abs, stats.rel_err_norm,
    );
    println!(
        "  argmax got={got_arg} want={want_arg} expected_gen_id={expected_gen_id} \
         argmax_vs_logits={} argmax_vs_gen_id={} top5_overlap={top5_overlap}/5",
        if match_oracle { "MATCH" } else { "MISMATCH" },
        if match_gen { "MATCH" } else { "MISMATCH" },
    );
    println!("  top5_got={:?} top5_want={:?}", top5_got, top5_want);
    if stats.cos_sim < 0.999 || !match_gen || !match_oracle {
        overall_pass = false;
        println!("[phase A] FAIL");
    }
    let phase_a_argmax = got_arg as u32;
    generated_ids_rust.push(phase_a_argmax);

    // ===== Optional intermediate KV sanity check =====
    if !cli.skip_kv_check {
        println!("\n[kv check] comparing filled K/V (slots 0..{prompt_tokens}) vs oracle");
        let mut kv_ok = true;
        for kv in kv_caches_oracle {
            let l = kv.layer;
            if l >= num_layers {
                continue;
            }
            let hd = layers[l].head_dim;
            let expected_shape = vec![1usize, num_kv_heads, prompt_tokens, hd];
            if kv.k_shape != expected_shape || kv.v_shape != expected_shape {
                bail!(
                    "oracle kv_caches[{l}] shape k={:?} v={:?} != expected {:?}",
                    kv.k_shape, kv.v_shape, expected_shape,
                );
            }
            let ok_bytes = B64.decode(&kv.k).context("decode oracle kv.k")?;
            let ov_bytes = B64.decode(&kv.v).context("decode oracle kv.v")?;
            let ok_f32 = bf16_bytes_to_f32(&ok_bytes);
            let ov_f32 = bf16_bytes_to_f32(&ov_bytes);
            let our_k = extract_filled_kv_to_oracle_layout(
                &k_caches[l], num_kv_heads, max_t, prompt_tokens, hd,
            )?;
            let our_v = extract_filled_kv_to_oracle_layout(
                &v_caches[l], num_kv_heads, max_t, prompt_tokens, hd,
            )?;
            let ks = compare_vectors(&our_k, &ok_f32)?;
            let vs = compare_vectors(&our_v, &ov_f32)?;
            let tag = match layers[l].kind {
                AttnKind::Sliding => "SWA",
                AttnKind::Full => "FULL",
            };
            println!(
                "  layer {l:2} ({tag}): k cos_sim={:.6} max_abs={:.6}  v cos_sim={:.6} max_abs={:.6}",
                ks.cos_sim, ks.max_abs, vs.cos_sim, vs.max_abs,
            );
            if ks.cos_sim < 0.999 || vs.cos_sim < 0.999 {
                kv_ok = false;
            }
        }
        if !kv_ok {
            println!("[kv check] WARN: some layer cos_sim < 0.999 — logit check is the load-bearing signal");
        }
    }

    // ===== PHASE B: Rust decode =====
    println!("\n[phase B] Rust decode ({} steps)", max_new_tokens.saturating_sub(1));
    for step in 0..max_new_tokens.saturating_sub(1) {
        let pos = prompt_tokens + step;
        let input_token_id: u32 = if step == 0 {
            phase_a_argmax
        } else {
            generated_ids_rust[step]
        };
        let logits_opt = run_forward_pass(
            &ctx, input_token_id, pos, &mut k_caches, &mut v_caches, &mut counter, true,
        )?;
        let logits = logits_opt.expect("decode step must emit logits");
        let want_logits = &oracle.decode_logits[step];
        if want_logits.len() != vocab_size {
            bail!(
                "oracle.decode_logits[{step}] len {} != vocab_size {}",
                want_logits.len(), vocab_size
            );
        }
        let stats = compare_vectors(&logits, want_logits)?;
        let got_arg = argmax(&logits);
        let want_arg = argmax(want_logits);
        let top5_got = top_k_indices(&logits, 5);
        let top5_want = top_k_indices(want_logits, 5);
        let top5_overlap = top5_got.iter().filter(|i| top5_want.contains(*i)).count();
        let expected_gen_id = oracle.generated_token_ids[step + 1];
        let match_gen = got_arg as u32 == expected_gen_id;
        let match_oracle = got_arg == want_arg;

        println!(
            "[phase B] step={step} input_tok={input_token_id} pos={pos} \
             vs decode_logits[{step}]: cos_sim={:.6} max_abs={:.6} rel_err={:.6}",
            stats.cos_sim, stats.max_abs, stats.rel_err_norm,
        );
        println!(
            "  argmax got={got_arg} want={want_arg} expected_gen_id={expected_gen_id} \
             argmax_vs_logits={} argmax_vs_gen_id={} top5_overlap={top5_overlap}/5",
            if match_oracle { "MATCH" } else { "MISMATCH" },
            if match_gen { "MATCH" } else { "MISMATCH" },
        );
        println!("  top5_got={:?} top5_want={:?}", top5_got, top5_want);

        generated_ids_rust.push(got_arg as u32);

        if stats.cos_sim < 0.999 || !match_gen || !match_oracle {
            overall_pass = false;
            println!("[phase B] step {step} FAIL");
        }
    }

    println!("\n[summary] generated_ids_rust                    = {:?}", generated_ids_rust);
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
        bail!("end-to-end validation failed (see per-phase diagnostics above)");
    }
    println!("PASS");
    Ok(())
}
