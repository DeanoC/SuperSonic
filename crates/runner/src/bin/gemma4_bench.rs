//! Throughput benchmark for Gemma 4 E2B decode — Step 17.
//!
//! Measures ms/tok and tokens/sec for the fused-per-layer path (35 ×
//! `fused_attn_block` + 35 × `fused_mlp_ple` + per-layer shared-KV D2D copies
//! + per-layer PLI slot D2D copies) against the persistent megakernel path
//! (a single `persistent_decode` launch). Both paths run the same forward
//! math on the same prompt and weights, with the same KV initialization.
//!
//! Timing excludes the per-step PLE computation and the final norm / lm_head
//! / softcap epilogue — the point is to isolate what Step 16 bought us at
//! the kernel-launch layer, not to measure the full host-wrapped decode.
//!
//! Usage:
//!   cargo run --release --bin gemma4_bench -- \
//!     --model-dir <checkpoint> --oracle-json <oracle_state.json> \
//!     [--iters 50] [--warmup 5]

use std::ffi::{c_int, c_void};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use clap::Parser;
use ::gemma4::config::{self as g4_config, AttnKind, Config, TextConfig};
use ::gemma4::weight_spec as g4_spec;
use gpu_hal::{GpuBuffer, GpuEvent, ScalarType};
use half::bf16;
use kernel_ffi::gemma4 as g4;
use kernel_ffi::gemma4::Gemma4DecodeLayerDesc;
use memmap2::Mmap;
use safetensors::SafeTensors;

#[path = "../oracle.rs"]
mod oracle;
use oracle::OracleOutput;

#[derive(Parser, Debug)]
#[command(about = "Benchmark Gemma 4 E2B decode: persistent megakernel vs fused-per-layer")]
struct Cli {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    oracle_json: PathBuf,
    /// Number of timed decode steps per path (after warmup).
    #[arg(long, default_value_t = 50)]
    iters: usize,
    /// Number of warmup decode steps per path before timing starts.
    #[arg(long, default_value_t = 5)]
    warmup: usize,
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
    assert!(rope_angles <= half);

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
        bail!("positions_to_copy > prompt_tokens");
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
        .ok_or_else(|| anyhow!("oracle JSON missing prompt_token_ids"))?;
    let kv_caches = oracle
        .kv_caches
        .as_ref()
        .ok_or_else(|| anyhow!("oracle JSON missing kv_caches (need --emit-state)"))?;

    let prompt_tokens = prompt_token_ids.len();
    let last_token_id = *prompt_token_ids
        .last()
        .ok_or_else(|| anyhow!("prompt_tokens == 0"))?;
    let hidden_size = tcfg.hidden_size;
    let eps = tcfg.rms_norm_eps as f32;
    let ple_hidden = tcfg.hidden_size_per_layer_input;
    let num_layers = tcfg.num_hidden_layers;
    let dtype = ScalarType::BF16;
    let num_q_heads = tcfg.num_attention_heads;
    let num_kv_heads = tcfg.num_key_value_heads;

    let iters = cli.iters;
    let warmup = cli.warmup;
    if iters == 0 {
        bail!("--iters must be >= 1");
    }
    // Each path advances `pos` from `prompt_tokens - 1` through its own warmup
    // + timed iterations into its own KV cache buffers. Size max_t for one
    // path; the second path resets `pos` and reuses its separate cache set.
    let max_t = prompt_tokens + warmup + iters + 4;

    let loader = UnbakedLoader::open(&cli.model_dir)?;
    let weight_prefix = "model.language_model";

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(load_layer_weights(&loader, tcfg, weight_prefix, i)?);
    }

    // Per-step PLI inputs are precomputed once (using the last prompt token)
    // so the bench loop can reuse the same BF16 buffer for every iteration —
    // the forward-pass timing excludes PLE recompute by design.
    let per_layer_model_projection_w =
        loader.load_bf16_to_gpu(&format!("{weight_prefix}.per_layer_model_projection.weight"))?;
    let per_layer_projection_norm_w =
        loader.load_bf16_to_gpu(&format!("{weight_prefix}.per_layer_projection_norm.weight"))?;
    let mut counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;

    let pli_bytes = compute_per_layer_inputs(
        &loader, weight_prefix, tcfg, last_token_id,
        &per_layer_model_projection_w, &per_layer_projection_norm_w,
        &mut counter,
    )?;
    let pli_gpu = GpuBuffer::from_host_bytes(
        0, dtype, &[num_layers, ple_hidden], &pli_bytes,
    )?;

    // Pre-upload the initial hidden BF16 vector (scaled embed of the last
    // prompt token). The bench loop uploads a fresh copy into `h_running`
    // each iteration so the same input is fed regardless of previous writes.
    let h_in_host = load_scaled_embed_row(
        &loader,
        &format!("{weight_prefix}.embed_tokens.weight"),
        last_token_id,
        hidden_size,
    )?;
    let h_in_bf16 = f32_to_bf16_bytes(&h_in_host);

    // RoPE tables sized for max_t.
    let sliding_rope = tcfg.rope_for(AttnKind::Sliding);
    let full_rope = tcfg.rope_for(AttnKind::Full);
    let sliding_head_dim = tcfg.head_dim_for(AttnKind::Sliding);
    let full_head_dim = tcfg.head_dim_for(AttnKind::Full);

    let (scos_h, ssin_h) =
        build_sliding_rope_table(sliding_head_dim, sliding_rope.rope_theta, max_t);
    let (fcos_h, fsin_h) = build_proportional_rope_table(
        full_head_dim, full_rope.rope_theta, full_rope.partial_rotary_factor, max_t,
    );
    let sliding_cos = upload_bf16(&[max_t, sliding_head_dim], &scos_h)?;
    let sliding_sin = upload_bf16(&[max_t, sliding_head_dim], &ssin_h)?;
    let full_cos = upload_bf16(&[max_t, full_head_dim], &fcos_h)?;
    let full_sin = upload_bf16(&[max_t, full_head_dim], &fsin_h)?;

    // Shared helper to seed a fresh set of cache buffers from the oracle dump.
    let seed_all_caches = |one_per_layer: bool| -> Result<(Vec<GpuBuffer>, Vec<GpuBuffer>)> {
        let mut k_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
        let mut v_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let w = &layers[l];
            if !one_per_layer && w.shared_kv {
                // Placeholder; shared layers get pointer-aliased by the
                // descriptor to the source layer's cache.
                k_caches.push(GpuBuffer::zeros(0, ScalarType::BF16, &[1])?);
                v_caches.push(GpuBuffer::zeros(0, ScalarType::BF16, &[1])?);
                continue;
            }
            let src_slot = if w.shared_kv { w.kv_source } else { l };
            let kv = kv_caches
                .get(src_slot)
                .ok_or_else(|| anyhow!("oracle kv_caches has no slot {src_slot}"))?;
            let k_bytes = B64.decode(&kv.k).context("decode kv.k")?;
            let v_bytes = B64.decode(&kv.v).context("decode kv.v")?;
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
        Ok((k_caches, v_caches))
    };

    // --- Fused-path state: one KV buffer per layer (35 total). ---
    let (mut fused_k_caches, mut fused_v_caches) = seed_all_caches(true)?;

    let fused_ws_elems = g4::fused_attn_block_workspace_elems(
        hidden_size, num_q_heads, num_kv_heads, full_head_dim, max_t,
    );
    let mut fused_attn_workspace = GpuBuffer::zeros(0, ScalarType::F32, &[fused_ws_elems])?;
    let max_intermediate = (0..num_layers)
        .map(|l| g4_spec::mlp_intermediate(tcfg, l))
        .max()
        .unwrap_or(tcfg.intermediate_size);
    let fused_mlp_ws_elems = g4::fused_mlp_ple_workspace_elems(
        hidden_size, max_intermediate, ple_hidden,
    );
    let mut fused_mlp_workspace =
        GpuBuffer::zeros(0, ScalarType::F32, &[fused_mlp_ws_elems])?;
    let mut fused_matvec_counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut fused_barrier_counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut fused_barrier_flag = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut fused_pli_slot = GpuBuffer::zeros(0, ScalarType::BF16, &[ple_hidden])?;

    // --- Mega-path state: one KV buffer per owning layer; shared layers
    //     have a tiny placeholder buffer and the descriptor aliases the
    //     owning layer's pointers so the kernel sees the same memory. ---
    let (mega_k_caches, mega_v_caches) = seed_all_caches(false)?;

    let mega_ws_elems = g4::persistent_decode_workspace_elems(
        hidden_size, num_q_heads, num_kv_heads, full_head_dim, max_t,
        max_intermediate, ple_hidden,
    );
    let mut mega_workspace = GpuBuffer::zeros(0, ScalarType::F32, &[mega_ws_elems])?;
    let mut mega_matvec_counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut mega_barrier_counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    let mut mega_barrier_flag = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;

    // Build the descriptor array once. Pointers into `mega_k_caches` /
    // `mega_v_caches` stay valid for the rest of the program.
    let mut descs: Vec<Gemma4DecodeLayerDesc> = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let w = &layers[l];
        let kind_code: c_int = match w.kind {
            AttnKind::Sliding => 0,
            AttnKind::Full => 1,
        };
        let sliding_window = match w.kind {
            AttnKind::Sliding => tcfg.sliding_window as c_int,
            AttnKind::Full => 0,
        };
        let (cos_table_buf, sin_table_buf) = match w.kind {
            AttnKind::Sliding => (&sliding_cos, &sliding_sin),
            AttnKind::Full => (&full_cos, &full_sin),
        };

        let src_idx = if w.shared_kv { w.kv_source } else { l };
        let k_buf = &mega_k_caches[src_idx];
        let v_buf = &mega_v_caches[src_idx];

        let k_proj_ptr = w.k_proj.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
        let v_proj_ptr = w.v_proj.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
        let k_norm_ptr = w.k_norm.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

        descs.push(Gemma4DecodeLayerDesc {
            layer_type: kind_code,
            shared_kv: if w.shared_kv { 1 } else { 0 },
            num_q_heads: num_q_heads as c_int,
            num_kv_heads: num_kv_heads as c_int,
            head_dim: w.head_dim as c_int,
            rotary_dim: w.head_dim as c_int,
            sliding_window,
            intermediate_size: w.intermediate_size as c_int,
            kv_max_t: max_t as c_int,
            layer_scalar: w.layer_scalar,
            input_norm_w: w.input_norm.as_ptr(),
            q_proj_w: w.q_proj.as_ptr(),
            k_proj_w: k_proj_ptr,
            v_proj_w: v_proj_ptr,
            q_norm_w: w.q_norm.as_ptr(),
            k_norm_w: k_norm_ptr,
            o_proj_w: w.o_proj.as_ptr(),
            post_attn_norm_w: w.post_attn_norm.as_ptr(),
            pre_ff_norm_w: w.pre_ff_norm.as_ptr(),
            gate_proj_w: w.gate_proj.as_ptr(),
            up_proj_w: w.up_proj.as_ptr(),
            down_proj_w: w.down_proj.as_ptr(),
            post_ff_norm_w: w.post_ff_norm.as_ptr(),
            per_layer_input_gate_w: w.per_layer_input_gate_w.as_ptr(),
            per_layer_projection_w: w.per_layer_projection_w.as_ptr(),
            post_per_layer_input_norm_w: w.post_per_layer_input_norm_w.as_ptr(),
            cos_table: cos_table_buf.as_ptr(),
            sin_table: sin_table_buf.as_ptr(),
            kv_cache_k: k_buf.as_ptr() as *mut c_void,
            kv_cache_v: v_buf.as_ptr() as *mut c_void,
        });
    }
    let desc_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            descs.len() * std::mem::size_of::<Gemma4DecodeLayerDesc>(),
        )
    };
    let layers_gpu = GpuBuffer::from_host_bytes(
        0, ScalarType::U8, &[desc_bytes.len()], desc_bytes,
    )?;

    let mut h_running = GpuBuffer::from_host_bytes(
        0, dtype, &[hidden_size], &h_in_bf16,
    )?;

    println!(
        "[cfg] prompt_tokens={prompt_tokens} last_token_id={last_token_id} \
         iters={iters} warmup={warmup} max_t={max_t} num_layers={num_layers} \
         hidden={hidden_size} ple_hidden={ple_hidden}"
    );

    // -------------------------------------------------------------------
    // Fused-per-layer path: 35 × (fused_attn_block + fused_mlp_ple), plus
    // shared-KV slot replication via copy_kv_slot, plus PLI slice D2D copy.
    // -------------------------------------------------------------------
    let fused_forward = |h_in: &mut GpuBuffer,
                        pos: usize,
                        k_caches: &mut [GpuBuffer],
                        v_caches: &mut [GpuBuffer],
                        attn_ws: &mut GpuBuffer,
                        mlp_ws: &mut GpuBuffer,
                        mv_counter: &mut GpuBuffer,
                        bar_counter: &mut GpuBuffer,
                        bar_flag: &mut GpuBuffer,
                        pli_slot: &mut GpuBuffer|
     -> Result<()> {
        let mut h_running_local = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
        // Copy h_in → h_running_local so each iteration starts from the
        // canonical BF16 embed row. This matches what the real decoder
        // would do (input vector arrives per step).
        gpu_hal::copy_d2d(
            0,
            h_running_local.as_mut_ptr(),
            h_in.as_ptr(),
            hidden_size * 2,
        )?;
        for layer_idx in 0..num_layers {
            let w = &layers[layer_idx];
            let head_dim = w.head_dim;
            let sliding_window = match w.kind {
                AttnKind::Sliding => tcfg.sliding_window as i32,
                AttnKind::Full => 0,
            };
            let (cos_table, sin_table) = match w.kind {
                AttnKind::Sliding => (&sliding_cos, &sliding_sin),
                AttnKind::Full => (&full_cos, &full_sin),
            };
            let mut h_mid = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
            g4::fused_attn_block(
                0, dtype,
                &h_running_local, &mut h_mid,
                &w.input_norm,
                &w.q_proj,
                w.k_proj.as_ref(), w.v_proj.as_ref(),
                &w.q_norm,
                w.k_norm.as_ref(),
                &w.o_proj,
                &w.post_attn_norm,
                cos_table, sin_table,
                &mut k_caches[layer_idx], &mut v_caches[layer_idx],
                attn_ws,
                mv_counter,
                bar_counter, bar_flag,
                hidden_size, num_q_heads, num_kv_heads, head_dim, head_dim,
                sliding_window, pos, max_t,
                w.shared_kv,
                eps, 1.0f32,
            )?;

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

            let pli_byte_off = layer_idx * ple_hidden * 2;
            let pli_slice_ptr = pli_gpu.offset_ptr(pli_byte_off);
            gpu_hal::copy_d2d(
                0,
                pli_slot.as_mut_ptr(),
                pli_slice_ptr,
                ple_hidden * 2,
            )?;

            let mut h_new = GpuBuffer::zeros(0, dtype, &[hidden_size])?;
            g4::fused_mlp_ple(
                0, dtype,
                &h_mid, &mut h_new,
                &w.pre_ff_norm,
                &w.gate_proj, &w.up_proj, &w.down_proj, &w.post_ff_norm,
                pli_slot,
                &w.per_layer_input_gate_w, &w.per_layer_projection_w,
                &w.post_per_layer_input_norm_w,
                mlp_ws,
                mv_counter,
                bar_counter, bar_flag,
                hidden_size, w.intermediate_size, ple_hidden,
                eps, w.layer_scalar,
            )?;
            h_running_local = h_new;
        }
        // Feed the final running hidden back out for callers that care.
        gpu_hal::copy_d2d(
            0,
            h_in.as_mut_ptr(),
            h_running_local.as_ptr(),
            hidden_size * 2,
        )?;
        Ok(())
    };

    // --- Fused warmup ---
    let mut pos_cursor = prompt_tokens - 1;
    for _ in 0..warmup {
        fused_forward(
            &mut h_running, pos_cursor,
            &mut fused_k_caches, &mut fused_v_caches,
            &mut fused_attn_workspace, &mut fused_mlp_workspace,
            &mut fused_matvec_counter,
            &mut fused_barrier_counter, &mut fused_barrier_flag,
            &mut fused_pli_slot,
        )?;
        pos_cursor += 1;
    }
    gpu_hal::sync(0)?;

    // --- Fused timed ---
    let fused_start = GpuEvent::new(0)?;
    let fused_end = GpuEvent::new(0)?;
    fused_start.record()?;
    for _ in 0..iters {
        fused_forward(
            &mut h_running, pos_cursor,
            &mut fused_k_caches, &mut fused_v_caches,
            &mut fused_attn_workspace, &mut fused_mlp_workspace,
            &mut fused_matvec_counter,
            &mut fused_barrier_counter, &mut fused_barrier_flag,
            &mut fused_pli_slot,
        )?;
        pos_cursor += 1;
    }
    fused_end.record()?;
    fused_end.synchronize()?;
    let fused_total_ms = GpuEvent::elapsed_ms(&fused_start, &fused_end)?;
    let fused_ms_per_tok = fused_total_ms / iters as f32;
    let fused_tok_per_s = 1000.0 / fused_ms_per_tok;

    // -------------------------------------------------------------------
    // Persistent megakernel path: single `persistent_decode` launch per step.
    // -------------------------------------------------------------------
    let mut mega_forward = |h_io: &mut GpuBuffer, pos: usize| -> Result<()> {
        // Reload the canonical input row each iteration so the pass starts
        // from the same embed vector regardless of what previous iterations
        // wrote back to `h_io`.
        gpu_hal::copy_h2d(
            0,
            h_io.as_mut_ptr(),
            h_in_bf16.as_ptr() as *const c_void,
            hidden_size * 2,
        )?;
        g4::persistent_decode(
            0, dtype,
            &layers_gpu,
            h_io,
            &pli_gpu,
            &mut mega_workspace,
            &mut mega_matvec_counter,
            &mut mega_barrier_counter,
            &mut mega_barrier_flag,
            num_layers, hidden_size, ple_hidden, pos, eps, 1.0f32,
        )?;
        Ok(())
    };

    // Reset h_running and pos_cursor for mega path — it writes into its own
    // cache buffers (`mega_k_caches`/`mega_v_caches`) so positions can reuse
    // the range [prompt_tokens-1, prompt_tokens-1+warmup+iters).
    let mut mega_pos = prompt_tokens - 1;
    for _ in 0..warmup {
        mega_forward(&mut h_running, mega_pos)?;
        mega_pos += 1;
    }
    gpu_hal::sync(0)?;

    let mega_start = GpuEvent::new(0)?;
    let mega_end = GpuEvent::new(0)?;
    mega_start.record()?;
    for _ in 0..iters {
        mega_forward(&mut h_running, mega_pos)?;
        mega_pos += 1;
    }
    mega_end.record()?;
    mega_end.synchronize()?;
    let mega_total_ms = GpuEvent::elapsed_ms(&mega_start, &mega_end)?;
    let mega_ms_per_tok = mega_total_ms / iters as f32;
    let mega_tok_per_s = 1000.0 / mega_ms_per_tok;

    // ---- Report ----
    println!();
    println!("=== Gemma 4 E2B decode forward-pass benchmark ===");
    println!("iters={iters} warmup={warmup}");
    println!();
    println!("fused-per-layer (35 × fused_attn_block + fused_mlp_ple + shared-KV D2D + PLI slice D2D):");
    println!("  total = {:.3} ms   ms/tok = {:.3}   tokens/sec = {:.2}",
             fused_total_ms, fused_ms_per_tok, fused_tok_per_s);
    println!();
    println!("persistent megakernel (single persistent_decode launch):");
    println!("  total = {:.3} ms   ms/tok = {:.3}   tokens/sec = {:.2}",
             mega_total_ms, mega_ms_per_tok, mega_tok_per_s);
    println!();
    let speedup = fused_ms_per_tok / mega_ms_per_tok;
    println!("speedup (fused → mega): {:.3}x  ({:.3} ms/tok saved)",
             speedup, fused_ms_per_tok - mega_ms_per_tok);

    Ok(())
}

/// Compute `per_layer_inputs` for a single decode-step input token. Moved
/// below `main` because the bench loop only invokes it once at setup; see
/// `gemma4_mega_decode_validate.rs` for the commented reference version.
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
    let proj_host_bytes = proj.to_host_bytes()?;
    let mut proj_host = bf16_bytes_to_f32(&proj_host_bytes);
    for v in proj_host.iter_mut() {
        *v *= proj_scale;
    }

    let proj_reshaped = upload_bf16(&[num_layers, ple_hidden], &proj_host)?;
    let mut proj_normed = GpuBuffer::zeros(0, dtype, &[num_layers, ple_hidden])?;
    g4::rms_norm_per_row(
        0, dtype, &mut proj_normed, &proj_reshaped,
        Some(per_layer_projection_norm_w), eps, num_layers, ple_hidden,
    )?;
    let proj_normed_bytes = proj_normed.to_host_bytes()?;
    let proj_normed_host = bf16_bytes_to_f32(&proj_normed_bytes);

    let (shape, bytes) = loader.tensor_bytes(&format!("{weight_prefix}.embed_tokens_per_layer.weight"))?;
    if shape.len() != 2 || shape[1] != total {
        bail!("embed_tokens_per_layer shape mismatch");
    }
    let row_bytes = total * 2;
    let off = token_id as usize * row_bytes;
    let raw = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
    let scale = (ple_hidden as f32).sqrt();
    let ple_raw: Vec<f32> = raw.iter().map(|v| v * scale).collect();

    let combine_scale = bf16::from_f32(2.0f32.powf(-0.5)).to_f32();
    let combined: Vec<f32> = proj_normed_host
        .iter()
        .zip(ple_raw.iter())
        .map(|(p, r)| (p + r) * combine_scale)
        .collect();
    Ok(f32_to_bf16_bytes(&combined))
}
