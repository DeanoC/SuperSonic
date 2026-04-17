//! Gemma 4 decode engine — loads weights, runs batched Rust prefill, and
//! runs per-step decode through the persistent megakernel.
//!
//! This is the minimal "production" wrapper around the primitives validated
//! by `gemma4_e2e_validate` (prefill) and `gemma4_mega_decode_validate`
//! (decode). The engine owns the mmap-backed safetensors loader (needed for
//! per-token `embed_tokens_per_layer` gathers — that table is ~4.6 GB and we
//! never upload it in full), the preloaded per-layer and global GPU weights,
//! the RoPE tables, the K/V caches, the megakernel scratch buffers, and the
//! on-device layer descriptor array.
//!
//! The engine does not load any oracle state — `prefill` builds the caches
//! entirely from the provided prompt token IDs.

use std::ffi::{c_int, c_void};
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ::gemma4::config::{AttnKind, Config, TextConfig};
use ::gemma4::weight_spec as g4_spec;
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::gemma4 as g4;
use kernel_ffi::gemma4::Gemma4DecodeLayerDesc;
use memmap2::Mmap;
use safetensors::SafeTensors;

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

fn upload_bf16(device: usize, shape: &[usize], host: &[f32]) -> Result<GpuBuffer> {
    let bytes = f32_to_bf16_bytes(host);
    Ok(GpuBuffer::from_host_bytes(device, ScalarType::BF16, shape, &bytes)?)
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

    fn load_bf16_to_gpu(&self, device: usize, name: &str) -> Result<GpuBuffer> {
        let (shape, bytes) = self.tensor_bytes(name)?;
        Ok(GpuBuffer::from_host_bytes(device, ScalarType::BF16, &shape, bytes)?)
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
    let row = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
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
        bail!("{weight_name} shape {:?} != [vocab, {expected_row_dim}]", shape);
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

fn gather_ple_raw_batch(
    loader: &UnbakedLoader,
    weight_name: &str,
    token_ids: &[u32],
    row_dim: usize,
    ple_hidden: usize,
) -> Result<Vec<f32>> {
    let (shape, bytes) = loader.tensor_bytes(weight_name)?;
    if shape.len() != 2 || shape[1] != row_dim {
        bail!("{weight_name} shape {:?} != [vocab, {row_dim}]", shape);
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

fn copy_kv_slots_range(
    device: usize,
    src: &GpuBuffer,
    dst: &mut GpuBuffer,
    num_kv_heads: usize,
    max_t: usize,
    head_dim: usize,
    t_start: usize,
    count: usize,
) -> Result<()> {
    let row_bytes = head_dim * 2;
    for h in 0..num_kv_heads {
        let byte_off = ((h * max_t) + t_start) * row_bytes;
        let bytes = count * row_bytes;
        let src_ptr = src.offset_ptr(byte_off);
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(byte_off) as *mut c_void };
        gpu_hal::copy_d2d(device, dst_ptr, src_ptr, bytes)
            .map_err(|e| anyhow!("copy_kv_slots_range: {e}"))?;
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
    device: usize,
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
            Some(loader.load_bf16_to_gpu(device, &want("self_attn.k_proj.weight")?)?),
            Some(loader.load_bf16_to_gpu(device, &want("self_attn.v_proj.weight")?)?),
            Some(loader.load_bf16_to_gpu(device, &want("self_attn.k_norm.weight")?)?),
        )
    };

    Ok(LayerWeights {
        kind,
        head_dim,
        intermediate_size,
        shared_kv,
        kv_source,
        layer_scalar: layer_scalar_value,
        input_norm: loader.load_bf16_to_gpu(device, &want("input_layernorm.weight")?)?,
        q_proj: loader.load_bf16_to_gpu(device, &want("self_attn.q_proj.weight")?)?,
        q_norm: loader.load_bf16_to_gpu(device, &want("self_attn.q_norm.weight")?)?,
        k_proj,
        v_proj,
        k_norm,
        o_proj: loader.load_bf16_to_gpu(device, &want("self_attn.o_proj.weight")?)?,
        post_attn_norm: loader.load_bf16_to_gpu(device, &want("post_attention_layernorm.weight")?)?,
        pre_ff_norm: loader.load_bf16_to_gpu(device, &want("pre_feedforward_layernorm.weight")?)?,
        post_ff_norm: loader.load_bf16_to_gpu(device, &want("post_feedforward_layernorm.weight")?)?,
        gate_proj: loader.load_bf16_to_gpu(device, &want("mlp.gate_proj.weight")?)?,
        up_proj: loader.load_bf16_to_gpu(device, &want("mlp.up_proj.weight")?)?,
        down_proj: loader.load_bf16_to_gpu(device, &want("mlp.down_proj.weight")?)?,
        per_layer_input_gate_w: loader.load_bf16_to_gpu(device, &want("per_layer_input_gate.weight")?)?,
        per_layer_projection_w: loader.load_bf16_to_gpu(device, &want("per_layer_projection.weight")?)?,
        post_per_layer_input_norm_w: loader
            .load_bf16_to_gpu(device, &want("post_per_layer_input_norm.weight")?)?,
    })
}

pub struct Gemma4Engine {
    tcfg: TextConfig,
    loader: UnbakedLoader,
    weight_prefix: &'static str,
    max_t: usize,
    device: usize,

    layers: Vec<LayerWeights>,
    lm_head_w: GpuBuffer, // tied to embed_tokens on E2B/E4B
    final_norm_w: GpuBuffer,
    per_layer_model_projection_w: GpuBuffer,
    per_layer_projection_norm_w: GpuBuffer,

    sliding_cos: GpuBuffer,
    sliding_sin: GpuBuffer,
    full_cos: GpuBuffer,
    full_sin: GpuBuffer,

    k_caches: Vec<GpuBuffer>,
    v_caches: Vec<GpuBuffer>,

    // Megakernel-side state (built once, reused every decode step).
    #[allow(dead_code)]
    descs: Vec<Gemma4DecodeLayerDesc>, // kept alive so the GPU-side `layers_gpu` pointers stay valid
    layers_gpu: GpuBuffer,
    mega_workspace: GpuBuffer,
    mega_matvec_counter: GpuBuffer,
    mega_barrier_counter: GpuBuffer,
    mega_barrier_flag: GpuBuffer,

    // Shared counter reused across one-shot primitives (matvec row-steal).
    counter: GpuBuffer,
}

impl Gemma4Engine {
    /// Load all weights for a Gemma 4 dense variant and initialize GPU state.
    /// `max_t` is the maximum token position the engine will ever be asked to
    /// handle (prompt length + max new tokens) — K/V caches are sized once.
    pub fn load(
        model_dir: &Path,
        weight_prefix: &'static str,
        max_t: usize,
        device: usize,
    ) -> Result<Self> {
        gpu_hal::set_device(device).map_err(|e| anyhow!("set_device: {e}"))?;
        let config: Config = ::gemma4::config::load_config(model_dir)
            .map_err(|e| anyhow!("load_config: {e}"))?;
        let tcfg = config.text_config;

        let loader = UnbakedLoader::open(model_dir)?;
        let num_layers = tcfg.num_hidden_layers;
        let dtype = ScalarType::BF16;

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(load_layer_weights(&loader, device, &tcfg, weight_prefix, i)?);
        }

        let lm_head_w = loader.load_bf16_to_gpu(device, &format!("{weight_prefix}.embed_tokens.weight"))?;
        let final_norm_w = loader.load_bf16_to_gpu(device, &format!("{weight_prefix}.norm.weight"))?;
        let per_layer_model_projection_w = loader
            .load_bf16_to_gpu(device, &format!("{weight_prefix}.per_layer_model_projection.weight"))?;
        let per_layer_projection_norm_w = loader
            .load_bf16_to_gpu(device, &format!("{weight_prefix}.per_layer_projection_norm.weight"))?;

        let num_kv_heads = tcfg.num_key_value_heads;
        let num_q_heads = tcfg.num_attention_heads;

        // One K/V buffer per layer — prefill fills them all; decode reads
        // shared-layer descriptors via pointer-alias into the source's buffer.
        let mut k_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
        let mut v_caches: Vec<GpuBuffer> = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let hd = layers[l].head_dim;
            k_caches.push(GpuBuffer::zeros(device, dtype, &[num_kv_heads, max_t, hd])?);
            v_caches.push(GpuBuffer::zeros(device, dtype, &[num_kv_heads, max_t, hd])?);
        }

        let sliding_head_dim = tcfg.head_dim_for(AttnKind::Sliding);
        let full_head_dim = tcfg.head_dim_for(AttnKind::Full);
        let sliding_rope = tcfg.rope_for(AttnKind::Sliding);
        let full_rope = tcfg.rope_for(AttnKind::Full);
        let (scos_h, ssin_h) =
            build_sliding_rope_table(sliding_head_dim, sliding_rope.rope_theta, max_t);
        let (fcos_h, fsin_h) = build_proportional_rope_table(
            full_head_dim,
            full_rope.rope_theta,
            full_rope.partial_rotary_factor,
            max_t,
        );
        let sliding_cos = upload_bf16(device, &[max_t, sliding_head_dim], &scos_h)?;
        let sliding_sin = upload_bf16(device, &[max_t, sliding_head_dim], &ssin_h)?;
        let full_cos = upload_bf16(device, &[max_t, full_head_dim], &fcos_h)?;
        let full_sin = upload_bf16(device, &[max_t, full_head_dim], &fsin_h)?;

        // --- Persistent-decode descriptor array. Shared layers alias their
        //     source's K/V cache pointers so the megakernel sees a single
        //     coherent memory region for the source → shared dependency.
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
            let k_buf = &k_caches[src_idx];
            let v_buf = &v_caches[src_idx];

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
            device,
            ScalarType::U8,
            &[desc_bytes.len()],
            desc_bytes,
        )?;

        let ple_hidden = tcfg.hidden_size_per_layer_input;
        let max_intermediate = (0..num_layers)
            .map(|l| g4_spec::mlp_intermediate(&tcfg, l))
            .max()
            .unwrap_or(tcfg.intermediate_size);
        let workspace_elems = g4::persistent_decode_workspace_elems(
            tcfg.hidden_size, num_q_heads, num_kv_heads, full_head_dim, max_t,
            max_intermediate, ple_hidden,
        );
        let mega_workspace = GpuBuffer::zeros(device, ScalarType::F32, &[workspace_elems])?;
        let mega_matvec_counter = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;
        let mega_barrier_counter = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;
        let mega_barrier_flag = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;
        let counter = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;

        Ok(Self {
            tcfg,
            loader,
            weight_prefix,
            max_t,
            device,
            layers,
            lm_head_w,
            final_norm_w,
            per_layer_model_projection_w,
            per_layer_projection_norm_w,
            sliding_cos,
            sliding_sin,
            full_cos,
            full_sin,
            k_caches,
            v_caches,
            descs,
            layers_gpu,
            mega_workspace,
            mega_matvec_counter,
            mega_barrier_counter,
            mega_barrier_flag,
            counter,
        })
    }

    pub fn text_config(&self) -> &TextConfig {
        &self.tcfg
    }

    pub fn max_t(&self) -> usize {
        self.max_t
    }

    /// Run batched Rust prefill over the whole prompt and return the softcapped
    /// logits at the last prompt position. Populates every K/V cache buffer;
    /// shared-KV layers get their slots replicated from the source layer's
    /// cache via D2D copies.
    pub fn prefill(&mut self, prompt_token_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = prompt_token_ids.len();
        if seq_len == 0 {
            bail!("prefill: empty prompt");
        }
        if seq_len > self.max_t {
            bail!("prefill: prompt_len {seq_len} > max_t {}", self.max_t);
        }
        let device = self.device;
        let dtype = ScalarType::BF16;
        let hidden_size = self.tcfg.hidden_size;
        let num_q_heads = self.tcfg.num_attention_heads;
        let num_kv_heads = self.tcfg.num_key_value_heads;
        let eps = self.tcfg.rms_norm_eps as f32;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let num_layers = self.tcfg.num_hidden_layers;
        let max_t = self.max_t;
        let vocab_size = self.tcfg.vocab_size;

        let mut id_bytes: Vec<u8> = Vec::with_capacity(seq_len * 4);
        for &id in prompt_token_ids {
            id_bytes.extend_from_slice(&id.to_le_bytes());
        }
        let token_ids_gpu =
            GpuBuffer::from_host_bytes(device, ScalarType::U32, &[seq_len], &id_bytes)?;

        let embed_scale = bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
        let mut h_running = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
        g4::embed_gather_scaled(
            device, dtype, &mut h_running, &token_ids_gpu, &self.lm_head_w,
            seq_len, hidden_size, vocab_size, embed_scale,
        )?;

        let pli = self.compute_per_layer_inputs_batched(&token_ids_gpu, prompt_token_ids)?;

        for layer_idx in 0..num_layers {
            let w = &self.layers[layer_idx];
            let head_dim = w.head_dim;
            let rotary_dim = head_dim;
            let q_dim = num_q_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let sliding_window = match w.kind {
                AttnKind::Sliding => self.tcfg.sliding_window as i32,
                AttnKind::Full => 0,
            };
            let (cos_table, sin_table) = match w.kind {
                AttnKind::Sliding => (&self.sliding_cos, &self.sliding_sin),
                AttnKind::Full => (&self.full_cos, &self.full_sin),
            };

            let residual = h_running.clone_device()?;

            let mut x = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::rms_norm_rows(
                device, dtype, &mut x, &h_running, Some(&w.input_norm),
                eps, seq_len, hidden_size,
            )?;

            let mut q = GpuBuffer::zeros(device, dtype, &[seq_len, num_q_heads, head_dim])?;
            g4::matvec_batched(
                device, dtype, &mut q, &x, &w.q_proj,
                seq_len, hidden_size, q_dim, &mut self.counter,
            )?;
            let mut q_normed =
                GpuBuffer::zeros(device, dtype, &[seq_len, num_q_heads, head_dim])?;
            g4::rms_norm_rows(
                device, dtype, &mut q_normed, &q, Some(&w.q_norm),
                eps, seq_len * num_q_heads, head_dim,
            )?;
            g4::rope_prefill(
                device, dtype, &mut q_normed, cos_table, sin_table,
                seq_len, num_q_heads, head_dim, rotary_dim, 0,
            )?;

            if !w.shared_kv {
                let k_proj = w.k_proj.as_ref().expect("k_proj on non-shared layer");
                let v_proj = w.v_proj.as_ref().expect("v_proj on non-shared layer");
                let k_norm = w.k_norm.as_ref().expect("k_norm on non-shared layer");

                let mut k = GpuBuffer::zeros(device, dtype, &[seq_len, num_kv_heads, head_dim])?;
                g4::matvec_batched(
                    device, dtype, &mut k, &x, k_proj,
                    seq_len, hidden_size, kv_dim, &mut self.counter,
                )?;
                let mut v = GpuBuffer::zeros(device, dtype, &[seq_len, num_kv_heads, head_dim])?;
                g4::matvec_batched(
                    device, dtype, &mut v, &x, v_proj,
                    seq_len, hidden_size, kv_dim, &mut self.counter,
                )?;

                let mut k_normed =
                    GpuBuffer::zeros(device, dtype, &[seq_len, num_kv_heads, head_dim])?;
                g4::rms_norm_rows(
                    device, dtype, &mut k_normed, &k, Some(k_norm),
                    eps, seq_len * num_kv_heads, head_dim,
                )?;
                let mut v_normed =
                    GpuBuffer::zeros(device, dtype, &[seq_len, num_kv_heads, head_dim])?;
                g4::rms_norm_rows(
                    device, dtype, &mut v_normed, &v, None,
                    eps, seq_len * num_kv_heads, head_dim,
                )?;

                g4::rope_prefill(
                    device, dtype, &mut k_normed, cos_table, sin_table,
                    seq_len, num_kv_heads, head_dim, rotary_dim, 0,
                )?;

                g4::kv_append_prefill(
                    device, dtype, &k_normed, &v_normed,
                    &mut self.k_caches[layer_idx], &mut self.v_caches[layer_idx],
                    seq_len, num_kv_heads, head_dim, 0, max_t,
                )?;

                // Replicate to every later layer that shares this one's KV.
                for shared_layer in (layer_idx + 1)..num_layers {
                    let s = &self.layers[shared_layer];
                    if s.shared_kv && s.kv_source == layer_idx {
                        let (lo, hi) = self.k_caches.split_at_mut(shared_layer);
                        copy_kv_slots_range(device,
                            &lo[layer_idx], &mut hi[0],
                            num_kv_heads, max_t, head_dim, 0, seq_len,
                        )?;
                        let (lo, hi) = self.v_caches.split_at_mut(shared_layer);
                        copy_kv_slots_range(device,
                            &lo[layer_idx], &mut hi[0],
                            num_kv_heads, max_t, head_dim, 0, seq_len,
                        )?;
                    }
                }
            }

            let mut attn_out =
                GpuBuffer::zeros(device, dtype, &[seq_len, num_q_heads, head_dim])?;
            let mut scores =
                GpuBuffer::zeros(device, ScalarType::F32, &[seq_len, num_q_heads, max_t])?;
            g4::attn_prefill(
                device, dtype, &q_normed,
                &self.k_caches[layer_idx], &self.v_caches[layer_idx],
                &mut scores, &mut attn_out,
                seq_len, num_q_heads, num_kv_heads, head_dim, 0, max_t,
                sliding_window, 1.0,
            )?;

            let mut o = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::matvec_batched(
                device, dtype, &mut o, &attn_out, &w.o_proj,
                seq_len, q_dim, hidden_size, &mut self.counter,
            )?;

            let mut x2 = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::rms_norm_rows(
                device, dtype, &mut x2, &o, Some(&w.post_attn_norm),
                eps, seq_len, hidden_size,
            )?;
            let mut h_mid = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::add_residual(
                device, dtype, &mut h_mid, &residual, &x2, seq_len * hidden_size,
            )?;

            let residual2 = h_mid.clone_device()?;

            let mut x3 = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::rms_norm_rows(
                device, dtype, &mut x3, &h_mid, Some(&w.pre_ff_norm),
                eps, seq_len, hidden_size,
            )?;

            let mut gate = GpuBuffer::zeros(device, dtype, &[seq_len, w.intermediate_size])?;
            g4::matvec_batched(
                device, dtype, &mut gate, &x3, &w.gate_proj,
                seq_len, hidden_size, w.intermediate_size, &mut self.counter,
            )?;
            let mut up_buf = GpuBuffer::zeros(device, dtype, &[seq_len, w.intermediate_size])?;
            g4::matvec_batched(
                device, dtype, &mut up_buf, &x3, &w.up_proj,
                seq_len, hidden_size, w.intermediate_size, &mut self.counter,
            )?;
            let mut y = GpuBuffer::zeros(device, dtype, &[seq_len, w.intermediate_size])?;
            g4::gelu_tanh_gate_mul(
                device, dtype, &mut y, &gate, &up_buf, seq_len * w.intermediate_size,
            )?;

            let mut m = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::matvec_batched(
                device, dtype, &mut m, &y, &w.down_proj,
                seq_len, w.intermediate_size, hidden_size, &mut self.counter,
            )?;

            let mut x4 = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::rms_norm_rows(
                device, dtype, &mut x4, &m, Some(&w.post_ff_norm),
                eps, seq_len, hidden_size,
            )?;
            let mut h_pre_ple = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::add_residual(
                device, dtype, &mut h_pre_ple, &residual2, &x4, seq_len * hidden_size,
            )?;

            let mut pli_slice = GpuBuffer::zeros(device, dtype, &[seq_len, ple_hidden])?;
            g4::gather_layer_slice(
                device, dtype, &mut pli_slice, &pli,
                seq_len, num_layers, ple_hidden, layer_idx,
            )?;

            let mut gated = GpuBuffer::zeros(device, dtype, &[seq_len, ple_hidden])?;
            g4::matvec_batched(
                device, dtype, &mut gated, &h_pre_ple, &w.per_layer_input_gate_w,
                seq_len, hidden_size, ple_hidden, &mut self.counter,
            )?;
            let mut gated_act = GpuBuffer::zeros(device, dtype, &[seq_len, ple_hidden])?;
            g4::gelu_tanh_gate_mul(
                device, dtype, &mut gated_act, &gated, &pli_slice, seq_len * ple_hidden,
            )?;
            let mut projected = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::matvec_batched(
                device, dtype, &mut projected, &gated_act, &w.per_layer_projection_w,
                seq_len, ple_hidden, hidden_size, &mut self.counter,
            )?;
            let mut normed = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::rms_norm_rows(
                device, dtype, &mut normed, &projected, Some(&w.post_per_layer_input_norm_w),
                eps, seq_len, hidden_size,
            )?;
            let mut h_new = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::add_scaled_residual(
                device, dtype, &mut h_new, &h_pre_ple, &normed,
                w.layer_scalar, seq_len * hidden_size,
            )?;
            h_running = h_new;
        }

        // Last-position hidden → final norm + lm_head + softcap.
        let last_byte_off = (seq_len - 1) * hidden_size * 2;
        let mut last_hidden = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        unsafe {
            let src_ptr = h_running.offset_ptr(last_byte_off);
            gpu_hal::copy_d2d(device, last_hidden.as_mut_ptr(), src_ptr, hidden_size * 2)
                .map_err(|e| anyhow!("copy last hidden: {e}"))?;
        }
        let mut post_norm = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        g4::rms_norm(
            device, dtype, &mut post_norm, &last_hidden, Some(&self.final_norm_w),
            eps, hidden_size,
        )?;
        let mut logits_gpu = GpuBuffer::zeros(device, dtype, &[vocab_size])?;
        g4::matvec(
            device, dtype, &mut logits_gpu, &post_norm, &self.lm_head_w,
            hidden_size, vocab_size, &mut self.counter,
        )?;
        let mut logits_host = download_bf16(&logits_gpu)?;
        let cap = self.tcfg.final_logit_softcapping.unwrap_or(30.0) as f32;
        for v in logits_host.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
        let _ = self.device;
        Ok(logits_host)
    }

    /// Single decode step via the persistent megakernel. Writes new K/V slot
    /// at `pos` in every non-shared layer's cache (shared layers read the
    /// source's cache via descriptor aliasing). Returns softcapped logits.
    pub fn decode_step(&mut self, input_token_id: u32, pos: usize) -> Result<Vec<f32>> {
        if pos >= self.max_t {
            bail!("decode_step: pos {pos} >= max_t {}", self.max_t);
        }
        let device = self.device;
        let dtype = ScalarType::BF16;
        let hidden_size = self.tcfg.hidden_size;
        let eps = self.tcfg.rms_norm_eps as f32;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let num_layers = self.tcfg.num_hidden_layers;
        let vocab_size = self.tcfg.vocab_size;

        let h_in_host = load_scaled_embed_row(
            &self.loader,
            &format!("{}.embed_tokens.weight", self.weight_prefix),
            input_token_id,
            hidden_size,
        )?;
        let mut h_running = upload_bf16(device, &[hidden_size], &h_in_host)?;

        let pli_bytes = self.compute_per_layer_inputs_single(input_token_id)?;
        let expected_pli_bytes = num_layers * ple_hidden * 2;
        if pli_bytes.len() != expected_pli_bytes {
            bail!(
                "compute_per_layer_inputs returned {} bytes, expected {}",
                pli_bytes.len(), expected_pli_bytes
            );
        }
        let pli_gpu = GpuBuffer::from_host_bytes(
            device, dtype, &[num_layers, ple_hidden], &pli_bytes,
        )?;

        g4::persistent_decode(
            device, dtype,
            &self.layers_gpu,
            &mut h_running,
            &pli_gpu,
            &mut self.mega_workspace,
            &mut self.mega_matvec_counter,
            &mut self.mega_barrier_counter,
            &mut self.mega_barrier_flag,
            num_layers, hidden_size, ple_hidden, pos, eps, 1.0f32,
        )?;

        let mut post_norm = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        g4::rms_norm(
            device, dtype, &mut post_norm, &h_running, Some(&self.final_norm_w),
            eps, hidden_size,
        )?;
        let mut logits_gpu = GpuBuffer::zeros(device, dtype, &[vocab_size])?;
        g4::matvec(
            device, dtype, &mut logits_gpu, &post_norm, &self.lm_head_w,
            hidden_size, vocab_size, &mut self.counter,
        )?;
        let mut logits_host = download_bf16(&logits_gpu)?;
        let cap = self.tcfg.final_logit_softcapping.unwrap_or(30.0) as f32;
        for v in logits_host.iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
        Ok(logits_host)
    }

    /// Greedy sample — argmax.
    pub fn greedy_sample(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &x) in logits.iter().enumerate() {
            if x > best_val {
                best_val = x;
                best = i;
            }
        }
        best as u32
    }

    fn compute_per_layer_inputs_single(&mut self, token_id: u32) -> Result<Vec<u8>> {
        let device = self.device;
        let hidden_size = self.tcfg.hidden_size;
        let num_layers = self.tcfg.num_hidden_layers;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let eps = self.tcfg.rms_norm_eps as f32;
        let dtype = ScalarType::BF16;
        let total = num_layers * ple_hidden;

        let main_embed_host = load_scaled_embed_row(
            &self.loader,
            &format!("{}.embed_tokens.weight", self.weight_prefix),
            token_id,
            hidden_size,
        )?;
        let main_embed_gpu = upload_bf16(device, &[hidden_size], &main_embed_host)?;

        let mut proj = GpuBuffer::zeros(device, dtype, &[total])?;
        g4::matvec(
            device, dtype, &mut proj, &main_embed_gpu, &self.per_layer_model_projection_w,
            hidden_size, total, &mut self.counter,
        )?;

        let proj_scale = bf16::from_f32((hidden_size as f32).powf(-0.5)).to_f32();
        let mut proj_host = download_bf16(&proj)?;
        for v in proj_host.iter_mut() {
            *v *= proj_scale;
        }

        let proj_reshaped = upload_bf16(device, &[num_layers, ple_hidden], &proj_host)?;
        let mut proj_normed = GpuBuffer::zeros(device, dtype, &[num_layers, ple_hidden])?;
        g4::rms_norm_per_row(
            device, dtype, &mut proj_normed, &proj_reshaped,
            Some(&self.per_layer_projection_norm_w), eps, num_layers, ple_hidden,
        )?;
        let proj_normed_host = download_bf16(&proj_normed)?;

        let ple_raw = load_ple_raw_row(
            &self.loader,
            &format!("{}.embed_tokens_per_layer.weight", self.weight_prefix),
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

    fn compute_per_layer_inputs_batched(
        &mut self,
        token_ids_gpu: &GpuBuffer,
        prompt_token_ids: &[u32],
    ) -> Result<GpuBuffer> {
        let device = self.device;
        let dtype = ScalarType::BF16;
        let seq_len = prompt_token_ids.len();
        let hidden_size = self.tcfg.hidden_size;
        let num_layers = self.tcfg.num_hidden_layers;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let vocab_size = self.tcfg.vocab_size;
        let eps = self.tcfg.rms_norm_eps as f32;
        let total = num_layers * ple_hidden;

        let embed_scale = bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
        let mut main_embed_batch = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
        g4::embed_gather_scaled(
            device, dtype, &mut main_embed_batch, token_ids_gpu, &self.lm_head_w,
            seq_len, hidden_size, vocab_size, embed_scale,
        )?;

        let mut proj = GpuBuffer::zeros(device, dtype, &[seq_len, total])?;
        g4::matvec_batched(
            device, dtype, &mut proj, &main_embed_batch, &self.per_layer_model_projection_w,
            seq_len, hidden_size, total, &mut self.counter,
        )?;

        let proj_scale = bf16::from_f32((hidden_size as f32).powf(-0.5)).to_f32();
        g4::scalar_mul_inplace(device, dtype, &mut proj, proj_scale, seq_len * total)?;

        let mut proj_normed =
            GpuBuffer::zeros(device, dtype, &[seq_len, num_layers, ple_hidden])?;
        g4::rms_norm_rows(
            device, dtype, &mut proj_normed, &proj, Some(&self.per_layer_projection_norm_w),
            eps, seq_len * num_layers, ple_hidden,
        )?;

        let ple_raw_host = gather_ple_raw_batch(
            &self.loader,
            &format!("{}.embed_tokens_per_layer.weight", self.weight_prefix),
            prompt_token_ids,
            total,
            ple_hidden,
        )?;
        let ple_raw_gpu = upload_bf16(device, &[seq_len, num_layers, ple_hidden], &ple_raw_host)?;

        let combine_scale = bf16::from_f32(2.0f32.powf(-0.5)).to_f32();
        let mut pli = GpuBuffer::zeros(device, dtype, &[seq_len, num_layers, ple_hidden])?;
        g4::add_scaled_residual(
            device, dtype, &mut pli, &proj_normed, &ple_raw_gpu,
            combine_scale, seq_len * num_layers * ple_hidden,
        )?;
        Ok(pli)
    }
}
