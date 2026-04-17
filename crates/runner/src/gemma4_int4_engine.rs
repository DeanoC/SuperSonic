//! Gemma 4 INT4 decode engine — sibling to `Gemma4Engine`.
//!
//! Consumes the GPTQ bake produced by `oracle/bake_int4_gemma4.py` (stored
//! under `{model_dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq/`). Every
//! projection weight lives as a `(packed u8, BF16 scale, BF16 zero)` trio;
//! norms and `embed_tokens*` tables stay BF16. For each forward pass this
//! engine runs the same layer math as the BF16 primitive chain
//! (`gemma4_decode_validate`), just with `g4::matvec_int4` /
//! `g4::matvec_batched_int4` in place of `g4::matvec` / `g4::matvec_batched`
//! for the nine per-layer projections. The persistent megakernel is not
//! used here — adding INT4 paths to the megakernel is a follow-up
//! optimization per the same precedent set by the Qwen INT4 work.
//!
//! Public API mirrors `Gemma4Engine`: `load`, `prefill(&[u32]) -> logits`,
//! `decode_step(token, pos) -> logits`, `greedy_sample`. Intended to be a
//! drop-in for `run_gemma4` when `--int4` is set.

use std::ffi::{c_int, c_void};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ::gemma4::config::{AttnKind, Config, TextConfig};
use ::gemma4::weight_spec as g4_spec;
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::gemma4 as g4;
use kernel_ffi::gemma4::{Gemma4DecodeLayerDesc, Gemma4Int4ScaleDesc};
use model_store::BakedStore;

const INT4_GROUP_SIZE: usize = 128;

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

/// A single INT4-quantized projection weight.
/// `group_size` is fixed at 128 by the bake format.
struct Int4Weight {
    packed: GpuBuffer,
    scale: GpuBuffer,
    zero: GpuBuffer,
}

impl Int4Weight {
    fn load(store: &BakedStore, ordinal: usize, base_name: &str) -> Result<Self> {
        let packed = store
            .load_to_gpu(base_name, ordinal)
            .with_context(|| format!("load INT4 packed tensor {base_name}"))?;
        let scale_name = format!("{base_name}_int4_scale");
        let zero_name = format!("{base_name}_int4_zero");
        let scale = store
            .load_to_gpu(&scale_name, ordinal)
            .with_context(|| format!("load INT4 scale tensor {scale_name}"))?;
        let zero = store
            .load_to_gpu(&zero_name, ordinal)
            .with_context(|| format!("load INT4 zero tensor {zero_name}"))?;
        let packed_shape = packed.shape().to_vec();
        if packed_shape.len() != 2 {
            bail!("{base_name}: packed tensor rank {} != 2", packed_shape.len());
        }
        let out_dim = packed_shape[0];
        let in_dim = packed_shape[1] * 2;
        if in_dim % INT4_GROUP_SIZE != 0 || out_dim % INT4_GROUP_SIZE != 0 {
            bail!(
                "{base_name}: shape [{out_dim}, {in_dim}] not aligned to group_size={INT4_GROUP_SIZE}"
            );
        }
        let expected_scale = &[out_dim / INT4_GROUP_SIZE, in_dim / INT4_GROUP_SIZE];
        if scale.shape() != expected_scale || zero.shape() != expected_scale {
            bail!(
                "{base_name}: scale/zero shape mismatch (expected {:?}, got scale={:?} zero={:?})",
                expected_scale,
                scale.shape(),
                zero.shape()
            );
        }
        // out_dim / in_dim are validated against the shape; they're not
        // retained since each call site already knows the projection's shape.
        let _ = (in_dim, out_dim);
        Ok(Self { packed, scale, zero })
    }
}

struct Int4LayerWeights {
    kind: AttnKind,
    head_dim: usize,
    intermediate_size: usize,
    shared_kv: bool,
    kv_source: usize,
    layer_scalar: f32,

    input_norm: GpuBuffer,
    q_proj: Int4Weight,
    q_norm: GpuBuffer,
    k_proj: Option<Int4Weight>,
    v_proj: Option<Int4Weight>,
    k_norm: Option<GpuBuffer>,
    o_proj: Int4Weight,
    post_attn_norm: GpuBuffer,
    pre_ff_norm: GpuBuffer,
    post_ff_norm: GpuBuffer,
    gate_proj: Int4Weight,
    up_proj: Int4Weight,
    down_proj: Int4Weight,
    per_layer_input_gate: Int4Weight,
    per_layer_projection: Int4Weight,
    post_per_layer_input_norm: GpuBuffer,
}

fn load_layer_weights_int4(
    store: &BakedStore,
    ordinal: usize,
    tcfg: &TextConfig,
    weight_prefix: &str,
    layer_idx: usize,
) -> Result<Int4LayerWeights> {
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
        let name = want("layer_scalar")?;
        let buf = store
            .load_to_gpu(&name, ordinal)
            .with_context(|| format!("load {name}"))?;
        let bytes = buf.to_host_bytes()?;
        bf16_bytes_to_f32(&bytes)[0]
    };

    let (k_proj, v_proj, k_norm) = if shared_kv {
        (None, None, None)
    } else {
        (
            Some(Int4Weight::load(store, ordinal, &want("self_attn.k_proj.weight")?)?),
            Some(Int4Weight::load(store, ordinal, &want("self_attn.v_proj.weight")?)?),
            Some(store.load_to_gpu(&want("self_attn.k_norm.weight")?, ordinal)?),
        )
    };

    Ok(Int4LayerWeights {
        kind,
        head_dim,
        intermediate_size,
        shared_kv,
        kv_source,
        layer_scalar: layer_scalar_value,
        input_norm: store.load_to_gpu(&want("input_layernorm.weight")?, ordinal)?,
        q_proj: Int4Weight::load(store, ordinal, &want("self_attn.q_proj.weight")?)?,
        q_norm: store.load_to_gpu(&want("self_attn.q_norm.weight")?, ordinal)?,
        k_proj,
        v_proj,
        k_norm,
        o_proj: Int4Weight::load(store, ordinal, &want("self_attn.o_proj.weight")?)?,
        post_attn_norm: store.load_to_gpu(&want("post_attention_layernorm.weight")?, ordinal)?,
        pre_ff_norm: store.load_to_gpu(&want("pre_feedforward_layernorm.weight")?, ordinal)?,
        post_ff_norm: store.load_to_gpu(&want("post_feedforward_layernorm.weight")?, ordinal)?,
        gate_proj: Int4Weight::load(store, ordinal, &want("mlp.gate_proj.weight")?)?,
        up_proj: Int4Weight::load(store, ordinal, &want("mlp.up_proj.weight")?)?,
        down_proj: Int4Weight::load(store, ordinal, &want("mlp.down_proj.weight")?)?,
        per_layer_input_gate: Int4Weight::load(
            store, ordinal, &want("per_layer_input_gate.weight")?
        )?,
        per_layer_projection: Int4Weight::load(
            store, ordinal, &want("per_layer_projection.weight")?
        )?,
        post_per_layer_input_norm: store
            .load_to_gpu(&want("post_per_layer_input_norm.weight")?, ordinal)?,
    })
}

fn copy_kv_slot(
    device: usize,
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
        gpu_hal::copy_d2d(device, dst_ptr, src_ptr, row_bytes)
            .map_err(|e| anyhow!("copy_kv_slot: {e}"))?;
    }
    Ok(())
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

pub fn int4_bake_dir(model_dir: &Path) -> PathBuf {
    model_store::bake_dir_int4(model_dir)
}

pub fn int4_bake_ok(model_dir: &Path) -> bool {
    let bake = int4_bake_dir(model_dir);
    model_store::version_ok(&bake)
}

pub struct Gemma4Int4Engine {
    tcfg: TextConfig,
    store: BakedStore,
    weight_prefix: String,
    max_t: usize,
    #[allow(dead_code)]
    device: usize,

    // `store` outlives `embed_tokens` + `embed_tokens_per_layer` which we
    // mmap-row-slice per token at compute time (the tables are ~1.3 GiB and
    // ~5.6 GiB respectively on E4B).
    layers: Vec<Int4LayerWeights>,
    lm_head_w: GpuBuffer,
    final_norm_w: GpuBuffer,
    per_layer_model_projection_w: GpuBuffer,
    per_layer_projection_norm_w: GpuBuffer,

    sliding_cos: GpuBuffer,
    sliding_sin: GpuBuffer,
    full_cos: GpuBuffer,
    full_sin: GpuBuffer,

    k_caches: Vec<GpuBuffer>,
    v_caches: Vec<GpuBuffer>,

    counter: GpuBuffer,

    // Fused-kernel scratch. Single F32 workspace sized via
    // `persistent_decode_workspace_elems` (max of attn + mlp across layers).
    // Used by both the Step-29/30 two-launch path (still callable via
    // `decode_step_primitive`'s primitive-chain reference doesn't touch it)
    // and the Step-31 single-launch megakernel. Matvec/barrier counters are
    // cleared by each kernel launch so one pair covers every phase safely.
    fused_workspace: GpuBuffer,
    fused_matvec_counter: GpuBuffer,
    fused_barrier_counter: GpuBuffer,
    fused_barrier_flag: GpuBuffer,

    // Persistent-decode megakernel descriptor arrays — populated once at
    // `load()`. Shared-KV layers alias their source layer's K/V cache
    // pointers so the megakernel sees a single coherent cache buffer for
    // the source→shared dependency (identical aliasing as BF16 megakernel).
    layer_descs: Vec<Gemma4DecodeLayerDesc>,
    layers_gpu: GpuBuffer,
    int4_scale_descs: Vec<Gemma4Int4ScaleDesc>,
    int4_scales_gpu: GpuBuffer,
}

impl Gemma4Int4Engine {
    pub fn load(
        model_dir: &Path,
        weight_prefix: &str,
        max_t: usize,
        device: usize,
    ) -> Result<Self> {
        gpu_hal::set_device(device).map_err(|e| anyhow!("set_device: {e}"))?;
        let config: Config = ::gemma4::config::load_config(model_dir)
            .map_err(|e| anyhow!("load_config: {e}"))?;
        let tcfg = config.text_config;

        let bake = int4_bake_dir(model_dir);
        if !model_store::version_ok(&bake) {
            bail!(
                "No INT4 bake at {}. Run:\n  \
                 python oracle/bake_int4_gemma4.py --model-dir {}\n",
                bake.display(),
                model_dir.display()
            );
        }
        let store = BakedStore::open(&bake)
            .with_context(|| format!("open INT4 bake at {}", bake.display()))?;

        let num_layers = tcfg.num_hidden_layers;
        let dtype = ScalarType::BF16;

        let mut layers: Vec<Int4LayerWeights> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(load_layer_weights_int4(&store, device, &tcfg, weight_prefix, i)?);
        }

        let lm_head_w = store.load_to_gpu(
            &format!("{weight_prefix}.embed_tokens.weight"), device,
        )?;
        let final_norm_w = store.load_to_gpu(
            &format!("{weight_prefix}.norm.weight"), device,
        )?;
        let per_layer_model_projection_w = store.load_to_gpu(
            &format!("{weight_prefix}.per_layer_model_projection.weight"), device,
        )?;
        let per_layer_projection_norm_w = store.load_to_gpu(
            &format!("{weight_prefix}.per_layer_projection_norm.weight"), device,
        )?;

        // Verify the raw PLE + embed tables exist in the bake — we mmap-slice
        // them per token at compute time rather than uploading in full.
        let ple_name = format!("{weight_prefix}.embed_tokens_per_layer.weight");
        if store.raw_bytes(&ple_name).is_none() {
            bail!("{ple_name} missing from INT4 bake");
        }

        let num_kv_heads = tcfg.num_key_value_heads;
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

        let counter = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;

        // Workspace for the fused INT4 kernels — sized to the max across
        // layers via `persistent_decode_workspace_elems`, which takes the
        // max of (fused_attn_block_workspace_elems, fused_mlp_ple_workspace_elems).
        // Same buffer serves both the single-launch persistent-decode path
        // and the Step-29/30 per-layer fused calls (sequential within a layer).
        let num_q_heads = tcfg.num_attention_heads;
        let num_kv_heads = tcfg.num_key_value_heads;
        let head_dim_max = full_head_dim.max(sliding_head_dim);
        let intermediate_max = layers.iter().map(|l| l.intermediate_size).max().unwrap_or(0);
        let ple_hidden = tcfg.hidden_size_per_layer_input;
        let fused_workspace_elems = g4::persistent_decode_workspace_elems(
            tcfg.hidden_size, num_q_heads, num_kv_heads, head_dim_max, max_t,
            intermediate_max, ple_hidden,
        );
        let fused_workspace =
            GpuBuffer::zeros(device, ScalarType::F32, &[fused_workspace_elems])?;
        let fused_matvec_counter = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;
        let fused_barrier_counter = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;
        let fused_barrier_flag = GpuBuffer::zeros(device, ScalarType::U32, &[1])?;

        // Persistent-decode megakernel descriptor arrays. For each layer we
        // populate one Gemma4DecodeLayerDesc (norms + packed-INT4 weight
        // pointers + RoPE + KV cache) and one parallel Gemma4Int4ScaleDesc
        // (scale/zero tables + group_size). Shared-KV layers alias the
        // source layer's K/V cache pointer so a single write-read dependency
        // lives in one buffer and the megakernel finds the populated slot
        // without any intra-kernel replication.
        let mut layer_descs: Vec<Gemma4DecodeLayerDesc> = Vec::with_capacity(num_layers);
        let mut int4_scale_descs: Vec<Gemma4Int4ScaleDesc> = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let w = &layers[l];
            let kind_code: c_int = match w.kind {
                AttnKind::Sliding => 0,
                AttnKind::Full => 1,
            };
            let sliding_window_c: c_int = match w.kind {
                AttnKind::Sliding => tcfg.sliding_window as c_int,
                AttnKind::Full => 0,
            };
            let (cos_buf, sin_buf) = match w.kind {
                AttnKind::Sliding => (&sliding_cos, &sliding_sin),
                AttnKind::Full => (&full_cos, &full_sin),
            };
            let src_idx = if w.shared_kv { w.kv_source } else { l };
            let k_buf = &k_caches[src_idx];
            let v_buf = &v_caches[src_idx];

            let k_proj_ptr = w.k_proj.as_ref().map(|p| p.packed.as_ptr()).unwrap_or(std::ptr::null());
            let v_proj_ptr = w.v_proj.as_ref().map(|p| p.packed.as_ptr()).unwrap_or(std::ptr::null());
            let k_norm_ptr = w.k_norm.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
            let k_scale_ptr = w.k_proj.as_ref().map(|p| p.scale.as_ptr()).unwrap_or(std::ptr::null());
            let k_zero_ptr = w.k_proj.as_ref().map(|p| p.zero.as_ptr()).unwrap_or(std::ptr::null());
            let v_scale_ptr = w.v_proj.as_ref().map(|p| p.scale.as_ptr()).unwrap_or(std::ptr::null());
            let v_zero_ptr = w.v_proj.as_ref().map(|p| p.zero.as_ptr()).unwrap_or(std::ptr::null());

            layer_descs.push(Gemma4DecodeLayerDesc {
                layer_type: kind_code,
                shared_kv: if w.shared_kv { 1 } else { 0 },
                num_q_heads: num_q_heads as c_int,
                num_kv_heads: num_kv_heads as c_int,
                head_dim: w.head_dim as c_int,
                rotary_dim: w.head_dim as c_int,
                sliding_window: sliding_window_c,
                intermediate_size: w.intermediate_size as c_int,
                kv_max_t: max_t as c_int,
                layer_scalar: w.layer_scalar,
                input_norm_w: w.input_norm.as_ptr(),
                q_proj_w: w.q_proj.packed.as_ptr(),
                k_proj_w: k_proj_ptr,
                v_proj_w: v_proj_ptr,
                q_norm_w: w.q_norm.as_ptr(),
                k_norm_w: k_norm_ptr,
                o_proj_w: w.o_proj.packed.as_ptr(),
                post_attn_norm_w: w.post_attn_norm.as_ptr(),
                pre_ff_norm_w: w.pre_ff_norm.as_ptr(),
                gate_proj_w: w.gate_proj.packed.as_ptr(),
                up_proj_w: w.up_proj.packed.as_ptr(),
                down_proj_w: w.down_proj.packed.as_ptr(),
                post_ff_norm_w: w.post_ff_norm.as_ptr(),
                per_layer_input_gate_w: w.per_layer_input_gate.packed.as_ptr(),
                per_layer_projection_w: w.per_layer_projection.packed.as_ptr(),
                post_per_layer_input_norm_w: w.post_per_layer_input_norm.as_ptr(),
                cos_table: cos_buf.as_ptr(),
                sin_table: sin_buf.as_ptr(),
                kv_cache_k: k_buf.as_ptr() as *mut c_void,
                kv_cache_v: v_buf.as_ptr() as *mut c_void,
            });
            int4_scale_descs.push(Gemma4Int4ScaleDesc {
                q_proj_scale: w.q_proj.scale.as_ptr(),
                q_proj_zero: w.q_proj.zero.as_ptr(),
                k_proj_scale: k_scale_ptr,
                k_proj_zero: k_zero_ptr,
                v_proj_scale: v_scale_ptr,
                v_proj_zero: v_zero_ptr,
                o_proj_scale: w.o_proj.scale.as_ptr(),
                o_proj_zero: w.o_proj.zero.as_ptr(),
                gate_proj_scale: w.gate_proj.scale.as_ptr(),
                gate_proj_zero: w.gate_proj.zero.as_ptr(),
                up_proj_scale: w.up_proj.scale.as_ptr(),
                up_proj_zero: w.up_proj.zero.as_ptr(),
                down_proj_scale: w.down_proj.scale.as_ptr(),
                down_proj_zero: w.down_proj.zero.as_ptr(),
                per_layer_input_gate_scale: w.per_layer_input_gate.scale.as_ptr(),
                per_layer_input_gate_zero: w.per_layer_input_gate.zero.as_ptr(),
                per_layer_projection_scale: w.per_layer_projection.scale.as_ptr(),
                per_layer_projection_zero: w.per_layer_projection.zero.as_ptr(),
                group_size: INT4_GROUP_SIZE as c_int,
            });
        }
        let layers_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                layer_descs.as_ptr() as *const u8,
                layer_descs.len() * std::mem::size_of::<Gemma4DecodeLayerDesc>(),
            )
        };
        let layers_gpu = GpuBuffer::from_host_bytes(
            device, ScalarType::U8, &[layers_bytes.len()], layers_bytes,
        )?;
        let int4_scales_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                int4_scale_descs.as_ptr() as *const u8,
                int4_scale_descs.len() * std::mem::size_of::<Gemma4Int4ScaleDesc>(),
            )
        };
        let int4_scales_gpu = GpuBuffer::from_host_bytes(
            device, ScalarType::U8, &[int4_scales_bytes.len()], int4_scales_bytes,
        )?;

        Ok(Self {
            tcfg,
            store,
            weight_prefix: weight_prefix.to_string(),
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
            counter,
            fused_workspace,
            fused_matvec_counter,
            fused_barrier_counter,
            fused_barrier_flag,
            layer_descs,
            layers_gpu,
            int4_scale_descs,
            int4_scales_gpu,
        })
    }

    pub fn text_config(&self) -> &TextConfig {
        &self.tcfg
    }

    pub fn max_t(&self) -> usize {
        self.max_t
    }

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

    /// Row-slice a scaled embed vector for `token_id` off the INT4 bake's
    /// raw-BF16 embed_tokens table. Matches the unbaked path (scale =
    /// bf16(sqrt(hidden_size))).
    fn load_scaled_embed_row(&self, token_id: u32) -> Result<Vec<f32>> {
        let hidden_size = self.tcfg.hidden_size;
        let name = format!("{}.embed_tokens.weight", self.weight_prefix);
        let bytes = self
            .store
            .raw_bytes(&name)
            .ok_or_else(|| anyhow!("{name} missing from INT4 bake"))?;
        let vocab = bytes.len() / (hidden_size * 2);
        if (token_id as usize) >= vocab {
            bail!("token id {token_id} out of range (vocab={vocab})");
        }
        let row_bytes = hidden_size * 2;
        let off = token_id as usize * row_bytes;
        let row = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
        let scale = bf16::from_f32((hidden_size as f32).sqrt()).to_f32();
        Ok(row.iter().map(|v| v * scale).collect())
    }

    /// Row-slice `embed_tokens_per_layer[token_id]` off the raw PLE table,
    /// pre-multiplied by `sqrt(ple_hidden)` to match HF's
    /// `Gemma4TextScaledWordEmbedding` with embed_scale.
    fn load_ple_raw_row(&self, token_id: u32) -> Result<Vec<f32>> {
        let num_layers = self.tcfg.num_hidden_layers;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let row_elems = num_layers * ple_hidden;
        let row_bytes = row_elems * 2;
        let name = format!("{}.embed_tokens_per_layer.weight", self.weight_prefix);
        let bytes = self
            .store
            .raw_bytes(&name)
            .ok_or_else(|| anyhow!("{name} missing from INT4 bake"))?;
        let vocab = bytes.len() / row_bytes;
        if (token_id as usize) >= vocab {
            bail!("token id {token_id} out of range (PLE vocab={vocab})");
        }
        let off = token_id as usize * row_bytes;
        let raw = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
        let scale = (ple_hidden as f32).sqrt();
        Ok(raw.iter().map(|v| v * scale).collect())
    }

    fn gather_ple_raw_batch(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let num_layers = self.tcfg.num_hidden_layers;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let row_elems = num_layers * ple_hidden;
        let row_bytes = row_elems * 2;
        let name = format!("{}.embed_tokens_per_layer.weight", self.weight_prefix);
        let bytes = self
            .store
            .raw_bytes(&name)
            .ok_or_else(|| anyhow!("{name} missing from INT4 bake"))?;
        let vocab = bytes.len() / row_bytes;
        let scale = (ple_hidden as f32).sqrt();
        let mut out = Vec::with_capacity(token_ids.len() * row_elems);
        for &tok in token_ids {
            if (tok as usize) >= vocab {
                bail!("token id {tok} out of range (PLE vocab={vocab})");
            }
            let off = tok as usize * row_bytes;
            let row = bf16_bytes_to_f32(&bytes[off..off + row_bytes]);
            out.extend(row.iter().map(|v| v * scale));
        }
        Ok(out)
    }

    /// Compute `per_layer_inputs` for a single token — identical to
    /// `Gemma4Engine::compute_per_layer_inputs_single` but the model
    /// projection weight comes from the INT4 bake's raw BF16 entry.
    fn compute_per_layer_inputs_single(&mut self, token_id: u32) -> Result<Vec<u8>> {
        let device = self.device;
        let hidden_size = self.tcfg.hidden_size;
        let num_layers = self.tcfg.num_hidden_layers;
        let ple_hidden = self.tcfg.hidden_size_per_layer_input;
        let eps = self.tcfg.rms_norm_eps as f32;
        let dtype = ScalarType::BF16;
        let total = num_layers * ple_hidden;

        let main_embed_host = self.load_scaled_embed_row(token_id)?;
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

        let ple_raw = self.load_ple_raw_row(token_id)?;
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

        let ple_raw_host = self.gather_ple_raw_batch(prompt_token_ids)?;
        let ple_raw_gpu = upload_bf16(device, &[seq_len, num_layers, ple_hidden], &ple_raw_host)?;

        let combine_scale = bf16::from_f32(2.0f32.powf(-0.5)).to_f32();
        let mut pli = GpuBuffer::zeros(device, dtype, &[seq_len, num_layers, ple_hidden])?;
        g4::add_scaled_residual(
            device, dtype, &mut pli, &proj_normed, &ple_raw_gpu,
            combine_scale, seq_len * num_layers * ple_hidden,
        )?;
        Ok(pli)
    }

    /// Batched INT4 prefill; returns softcapped logits at the last position.
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
            g4::matvec_batched_int4(
                device, dtype, &mut q, &x,
                &w.q_proj.packed, &w.q_proj.scale, &w.q_proj.zero,
                seq_len, hidden_size, q_dim, INT4_GROUP_SIZE, &mut self.counter,
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
                g4::matvec_batched_int4(
                    device, dtype, &mut k, &x,
                    &k_proj.packed, &k_proj.scale, &k_proj.zero,
                    seq_len, hidden_size, kv_dim, INT4_GROUP_SIZE, &mut self.counter,
                )?;
                let mut v = GpuBuffer::zeros(device, dtype, &[seq_len, num_kv_heads, head_dim])?;
                g4::matvec_batched_int4(
                    device, dtype, &mut v, &x,
                    &v_proj.packed, &v_proj.scale, &v_proj.zero,
                    seq_len, hidden_size, kv_dim, INT4_GROUP_SIZE, &mut self.counter,
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
            g4::matvec_batched_int4(
                device, dtype, &mut o, &attn_out,
                &w.o_proj.packed, &w.o_proj.scale, &w.o_proj.zero,
                seq_len, q_dim, hidden_size, INT4_GROUP_SIZE, &mut self.counter,
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
            g4::matvec_batched_int4(
                device, dtype, &mut gate, &x3,
                &w.gate_proj.packed, &w.gate_proj.scale, &w.gate_proj.zero,
                seq_len, hidden_size, w.intermediate_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut up_buf = GpuBuffer::zeros(device, dtype, &[seq_len, w.intermediate_size])?;
            g4::matvec_batched_int4(
                device, dtype, &mut up_buf, &x3,
                &w.up_proj.packed, &w.up_proj.scale, &w.up_proj.zero,
                seq_len, hidden_size, w.intermediate_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut y = GpuBuffer::zeros(device, dtype, &[seq_len, w.intermediate_size])?;
            g4::gelu_tanh_gate_mul(
                device, dtype, &mut y, &gate, &up_buf, seq_len * w.intermediate_size,
            )?;

            let mut m = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::matvec_batched_int4(
                device, dtype, &mut m, &y,
                &w.down_proj.packed, &w.down_proj.scale, &w.down_proj.zero,
                seq_len, w.intermediate_size, hidden_size,
                INT4_GROUP_SIZE, &mut self.counter,
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
            g4::matvec_batched_int4(
                device, dtype, &mut gated, &h_pre_ple,
                &w.per_layer_input_gate.packed,
                &w.per_layer_input_gate.scale,
                &w.per_layer_input_gate.zero,
                seq_len, hidden_size, ple_hidden,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut gated_act = GpuBuffer::zeros(device, dtype, &[seq_len, ple_hidden])?;
            g4::gelu_tanh_gate_mul(
                device, dtype, &mut gated_act, &gated, &pli_slice, seq_len * ple_hidden,
            )?;
            let mut projected = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::matvec_batched_int4(
                device, dtype, &mut projected, &gated_act,
                &w.per_layer_projection.packed,
                &w.per_layer_projection.scale,
                &w.per_layer_projection.zero,
                seq_len, ple_hidden, hidden_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut normed = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::rms_norm_rows(
                device, dtype, &mut normed, &projected, Some(&w.post_per_layer_input_norm),
                eps, seq_len, hidden_size,
            )?;
            let mut h_new = GpuBuffer::zeros(device, dtype, &[seq_len, hidden_size])?;
            g4::add_scaled_residual(
                device, dtype, &mut h_new, &h_pre_ple, &normed,
                w.layer_scalar, seq_len * hidden_size,
            )?;
            h_running = h_new;
        }

        let last_byte_off = (seq_len - 1) * hidden_size * 2;
        let mut last_hidden = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        let src_ptr = h_running.offset_ptr(last_byte_off);
        gpu_hal::copy_d2d(device, last_hidden.as_mut_ptr(), src_ptr, hidden_size * 2)
            .map_err(|e| anyhow!("copy last hidden: {e}"))?;
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
        Ok(logits_host)
    }

    /// Single INT4 decode step — full 35-layer forward pass in ONE kernel
    /// launch via `g4::persistent_decode_int4`. Every matmul (Q/K/V/O/
    /// gate/up/down/per_layer_input_gate/per_layer_projection) is
    /// INT4-dequantized inline. Shared-KV layers see their source layer's
    /// cache via pointer aliasing in the descriptor array — no intra-kernel
    /// replication needed. The only pre-kernel work is the PLI compute
    /// (small matmul + norm over the token's row) and the lm-head epilogue
    /// runs after. Returns softcapped logits.
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

        let h_in_host = self.load_scaled_embed_row(input_token_id)?;
        let mut h_running = upload_bf16(device, &[hidden_size], &h_in_host)?;

        let pli_bytes = self.compute_per_layer_inputs_single(input_token_id)?;
        let pli_gpu = GpuBuffer::from_host_bytes(
            device, dtype, &[num_layers, ple_hidden], &pli_bytes,
        )?;

        g4::persistent_decode_int4(
            device, dtype,
            &self.layers_gpu,
            &self.int4_scales_gpu,
            &mut h_running,
            &pli_gpu,
            &mut self.fused_workspace,
            &mut self.fused_matvec_counter,
            &mut self.fused_barrier_counter,
            &mut self.fused_barrier_flag,
            num_layers, hidden_size, ple_hidden, pos, eps, 1.0f32,
        )?;

        let mut post_norm = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        g4::rms_norm(device, dtype, &mut post_norm, &h_running, Some(&self.final_norm_w), eps, hidden_size)?;
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

    /// Per-layer fused INT4 decode step (Step 29/30 path) — retained for
    /// benchmarking the megakernel's launch-overhead savings. Runs each
    /// layer as two fused launches (attn + mlp/ple) plus host-side PLI
    /// slicing. Uses the same K/V caches as `decode_step`; shared-KV layers
    /// get source writes replicated via D2D `copy_kv_slot` calls because
    /// the fused kernel doesn't alias pointers.
    #[allow(dead_code)]
    fn _decode_step_fused_per_layer(&mut self, input_token_id: u32, pos: usize) -> Result<Vec<f32>> {
        if pos >= self.max_t {
            bail!("decode_step: pos {pos} >= max_t {}", self.max_t);
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

        let h_in_host = self.load_scaled_embed_row(input_token_id)?;
        let mut h_running = upload_bf16(device, &[hidden_size], &h_in_host)?;

        let pli_bytes = self.compute_per_layer_inputs_single(input_token_id)?;

        for layer_idx in 0..num_layers {
            let w = &self.layers[layer_idx];
            let head_dim = w.head_dim;
            let rotary_dim = head_dim;
            let sliding_window = match w.kind {
                AttnKind::Sliding => self.tcfg.sliding_window as i32,
                AttnKind::Full => 0,
            };
            let (cos_table, sin_table) = match w.kind {
                AttnKind::Sliding => (&self.sliding_cos, &self.sliding_sin),
                AttnKind::Full => (&self.full_cos, &self.full_sin),
            };

            // Non-shared-KV source layers must publish their K/V to every
            // dependent shared layer *before* those layers execute their own
            // fused_attn_block_int4 (which reads the cache). Produce K/V into
            // this layer's cache first.
            if !w.shared_kv {
                let k_proj = w.k_proj.as_ref().expect("k_proj on non-shared layer");
                let v_proj = w.v_proj.as_ref().expect("v_proj on non-shared layer");
                let k_norm = w.k_norm.as_ref().expect("k_norm on non-shared layer");
                let kv_dim = num_kv_heads * head_dim;

                let mut x = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
                g4::rms_norm(device, dtype, &mut x, &h_running, Some(&w.input_norm), eps, hidden_size)?;

                let mut k = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::matvec_int4(
                    device, dtype, &mut k, &x,
                    &k_proj.packed, &k_proj.scale, &k_proj.zero,
                    hidden_size, kv_dim, INT4_GROUP_SIZE, &mut self.counter,
                )?;
                let mut v = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::matvec_int4(
                    device, dtype, &mut v, &x,
                    &v_proj.packed, &v_proj.scale, &v_proj.zero,
                    hidden_size, kv_dim, INT4_GROUP_SIZE, &mut self.counter,
                )?;

                let mut k_normed = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::rms_norm_per_row(
                    device, dtype, &mut k_normed, &k, Some(k_norm),
                    eps, num_kv_heads, head_dim,
                )?;
                let mut v_normed = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::rms_norm_per_row(
                    device, dtype, &mut v_normed, &v, None,
                    eps, num_kv_heads, head_dim,
                )?;

                g4::rope_decode(
                    device, dtype, &mut k_normed, cos_table, sin_table,
                    num_kv_heads, head_dim, rotary_dim, pos,
                )?;

                g4::kv_append(
                    device, dtype, &k_normed, &v_normed,
                    &mut self.k_caches[layer_idx], &mut self.v_caches[layer_idx],
                    num_kv_heads, head_dim, pos, max_t,
                )?;

                for shared_layer in (layer_idx + 1)..num_layers {
                    let s = &self.layers[shared_layer];
                    if s.shared_kv && s.kv_source == layer_idx {
                        let (lo, hi) = self.k_caches.split_at_mut(shared_layer);
                        copy_kv_slot(device, &lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                        let (lo, hi) = self.v_caches.split_at_mut(shared_layer);
                        copy_kv_slot(device, &lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                    }
                }
            }

            let (k_caches_src, v_caches_src) =
                (self.k_caches.as_mut_slice(), self.v_caches.as_mut_slice());
            let mut h_mid = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::fused_attn_block_int4(
                device, dtype,
                &h_running, &mut h_mid,
                &w.input_norm,
                &w.q_proj.packed, &w.q_proj.scale, &w.q_proj.zero,
                w.k_proj.as_ref().map(|p| &p.packed),
                w.k_proj.as_ref().map(|p| &p.scale),
                w.k_proj.as_ref().map(|p| &p.zero),
                w.v_proj.as_ref().map(|p| &p.packed),
                w.v_proj.as_ref().map(|p| &p.scale),
                w.v_proj.as_ref().map(|p| &p.zero),
                &w.q_norm,
                w.k_norm.as_ref(),
                &w.o_proj.packed, &w.o_proj.scale, &w.o_proj.zero,
                &w.post_attn_norm,
                cos_table, sin_table,
                &mut k_caches_src[layer_idx], &mut v_caches_src[layer_idx],
                &mut self.fused_workspace,
                &mut self.fused_matvec_counter,
                &mut self.fused_barrier_counter,
                &mut self.fused_barrier_flag,
                hidden_size, num_q_heads, num_kv_heads, head_dim, rotary_dim,
                sliding_window, pos, max_t,
                w.shared_kv,
                INT4_GROUP_SIZE,
                eps, 1.0,
            )?;

            let bytes_per_layer = ple_hidden * 2;
            let pli_off = layer_idx * bytes_per_layer;
            let pli_slice = &pli_bytes[pli_off..pli_off + bytes_per_layer];
            let per_layer_input_gpu =
                upload_bf16(device, &[ple_hidden], &bf16_bytes_to_f32(pli_slice))?;

            let mut h_new = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::fused_mlp_ple_int4(
                device, dtype,
                &h_mid, &mut h_new,
                &w.pre_ff_norm,
                &w.gate_proj.packed, &w.gate_proj.scale, &w.gate_proj.zero,
                &w.up_proj.packed, &w.up_proj.scale, &w.up_proj.zero,
                &w.down_proj.packed, &w.down_proj.scale, &w.down_proj.zero,
                &w.post_ff_norm,
                &per_layer_input_gpu,
                &w.per_layer_input_gate.packed,
                &w.per_layer_input_gate.scale,
                &w.per_layer_input_gate.zero,
                &w.per_layer_projection.packed,
                &w.per_layer_projection.scale,
                &w.per_layer_projection.zero,
                &w.post_per_layer_input_norm,
                &mut self.fused_workspace,
                &mut self.fused_matvec_counter,
                &mut self.fused_barrier_counter,
                &mut self.fused_barrier_flag,
                hidden_size, w.intermediate_size, ple_hidden,
                INT4_GROUP_SIZE,
                eps, w.layer_scalar,
            )?;
            h_running = h_new;
        }

        let mut post_norm = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        g4::rms_norm(device, dtype, &mut post_norm, &h_running, Some(&self.final_norm_w), eps, hidden_size)?;

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

    /// Reference INT4 decode step that runs the attention block as the
    /// pre-Step-28 primitive chain (10 launches per layer instead of one
    /// fused call). Used only by `gemma4_fused_int4_validate` to confirm
    /// that fusing launches changes nothing numerically. MLP+PLE is
    /// identical between the two paths, so any divergence is localized to
    /// the attention block.
    pub fn decode_step_primitive(
        &mut self, input_token_id: u32, pos: usize,
    ) -> Result<Vec<f32>> {
        if pos >= self.max_t {
            bail!("decode_step_primitive: pos {pos} >= max_t {}", self.max_t);
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

        let h_in_host = self.load_scaled_embed_row(input_token_id)?;
        let mut h_running = upload_bf16(device, &[hidden_size], &h_in_host)?;

        let pli_bytes = self.compute_per_layer_inputs_single(input_token_id)?;

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

            let mut x = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::rms_norm(device, dtype, &mut x, &h_running, Some(&w.input_norm), eps, hidden_size)?;

            let mut q = GpuBuffer::zeros(device, dtype, &[num_q_heads, head_dim])?;
            g4::matvec_int4(
                device, dtype, &mut q, &x,
                &w.q_proj.packed, &w.q_proj.scale, &w.q_proj.zero,
                hidden_size, q_dim, INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut q_normed = GpuBuffer::zeros(device, dtype, &[num_q_heads, head_dim])?;
            g4::rms_norm_per_row(
                device, dtype, &mut q_normed, &q, Some(&w.q_norm), eps, num_q_heads, head_dim,
            )?;
            g4::rope_decode(
                device, dtype, &mut q_normed, cos_table, sin_table,
                num_q_heads, head_dim, rotary_dim, pos,
            )?;

            if !w.shared_kv {
                let k_proj = w.k_proj.as_ref().expect("k_proj on non-shared layer");
                let v_proj = w.v_proj.as_ref().expect("v_proj on non-shared layer");
                let k_norm = w.k_norm.as_ref().expect("k_norm on non-shared layer");

                let mut k = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::matvec_int4(
                    device, dtype, &mut k, &x,
                    &k_proj.packed, &k_proj.scale, &k_proj.zero,
                    hidden_size, kv_dim, INT4_GROUP_SIZE, &mut self.counter,
                )?;
                let mut v = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::matvec_int4(
                    device, dtype, &mut v, &x,
                    &v_proj.packed, &v_proj.scale, &v_proj.zero,
                    hidden_size, kv_dim, INT4_GROUP_SIZE, &mut self.counter,
                )?;

                let mut k_normed = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::rms_norm_per_row(
                    device, dtype, &mut k_normed, &k, Some(k_norm),
                    eps, num_kv_heads, head_dim,
                )?;
                let mut v_normed = GpuBuffer::zeros(device, dtype, &[num_kv_heads, head_dim])?;
                g4::rms_norm_per_row(
                    device, dtype, &mut v_normed, &v, None,
                    eps, num_kv_heads, head_dim,
                )?;

                g4::rope_decode(
                    device, dtype, &mut k_normed, cos_table, sin_table,
                    num_kv_heads, head_dim, rotary_dim, pos,
                )?;

                g4::kv_append(
                    device, dtype, &k_normed, &v_normed,
                    &mut self.k_caches[layer_idx], &mut self.v_caches[layer_idx],
                    num_kv_heads, head_dim, pos, max_t,
                )?;

                for shared_layer in (layer_idx + 1)..num_layers {
                    let s = &self.layers[shared_layer];
                    if s.shared_kv && s.kv_source == layer_idx {
                        let (lo, hi) = self.k_caches.split_at_mut(shared_layer);
                        copy_kv_slot(device, &lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                        let (lo, hi) = self.v_caches.split_at_mut(shared_layer);
                        copy_kv_slot(device, &lo[layer_idx], &mut hi[0], num_kv_heads, max_t, head_dim, pos)?;
                    }
                }
            }

            let kv_len = pos + 1;
            let mut attn_out = GpuBuffer::zeros(device, dtype, &[num_q_heads, head_dim])?;
            let mut scores = GpuBuffer::zeros(device, ScalarType::F32, &[num_q_heads, max_t])?;
            g4::swa_attn_decode(
                device, dtype, &q_normed,
                &self.k_caches[layer_idx], &self.v_caches[layer_idx],
                &mut scores, &mut attn_out,
                num_q_heads, num_kv_heads, head_dim, kv_len, max_t, sliding_window, 1.0,
            )?;

            let attn_flat = {
                let bytes = attn_out.to_host_bytes()?;
                GpuBuffer::from_host_bytes(device, dtype, &[q_dim], &bytes)?
            };
            let mut o = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::matvec_int4(
                device, dtype, &mut o, &attn_flat,
                &w.o_proj.packed, &w.o_proj.scale, &w.o_proj.zero,
                q_dim, hidden_size, INT4_GROUP_SIZE, &mut self.counter,
            )?;

            let mut x2 = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::rms_norm(device, dtype, &mut x2, &o, Some(&w.post_attn_norm), eps, hidden_size)?;
            let residual_h = download_bf16(&residual)?;
            let x2_h = download_bf16(&x2)?;
            let h1_h: Vec<f32> = residual_h.iter().zip(x2_h.iter()).map(|(a, b)| a + b).collect();
            let h_mid = upload_bf16(device, &[hidden_size], &h1_h)?;

            let residual2 = h_mid.clone_device()?;

            let mut x3 = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::rms_norm(device, dtype, &mut x3, &h_mid, Some(&w.pre_ff_norm), eps, hidden_size)?;

            let mut gate = GpuBuffer::zeros(device, dtype, &[w.intermediate_size])?;
            g4::matvec_int4(
                device, dtype, &mut gate, &x3,
                &w.gate_proj.packed, &w.gate_proj.scale, &w.gate_proj.zero,
                hidden_size, w.intermediate_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut up_buf = GpuBuffer::zeros(device, dtype, &[w.intermediate_size])?;
            g4::matvec_int4(
                device, dtype, &mut up_buf, &x3,
                &w.up_proj.packed, &w.up_proj.scale, &w.up_proj.zero,
                hidden_size, w.intermediate_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut y = GpuBuffer::zeros(device, dtype, &[w.intermediate_size])?;
            g4::gelu_tanh_gate_mul(device, dtype, &mut y, &gate, &up_buf, w.intermediate_size)?;

            let mut m = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::matvec_int4(
                device, dtype, &mut m, &y,
                &w.down_proj.packed, &w.down_proj.scale, &w.down_proj.zero,
                w.intermediate_size, hidden_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;

            let mut x4 = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::rms_norm(device, dtype, &mut x4, &m, Some(&w.post_ff_norm), eps, hidden_size)?;
            let residual2_h = download_bf16(&residual2)?;
            let x4_h = download_bf16(&x4)?;
            let h_pre_ple: Vec<f32> =
                residual2_h.iter().zip(x4_h.iter()).map(|(a, b)| a + b).collect();

            let bytes_per_layer = ple_hidden * 2;
            let pli_off = layer_idx * bytes_per_layer;
            let pli_slice = &pli_bytes[pli_off..pli_off + bytes_per_layer];
            let per_layer_input_f32 = bf16_bytes_to_f32(pli_slice);
            let per_layer_input_gpu = upload_bf16(device, &[ple_hidden], &per_layer_input_f32)?;

            let ple_residual = upload_bf16(device, &[hidden_size], &h_pre_ple)?;
            let h_in_ple = ple_residual.clone_device()?;

            let mut gated = GpuBuffer::zeros(device, dtype, &[ple_hidden])?;
            g4::matvec_int4(
                device, dtype, &mut gated, &h_in_ple,
                &w.per_layer_input_gate.packed,
                &w.per_layer_input_gate.scale,
                &w.per_layer_input_gate.zero,
                hidden_size, ple_hidden,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut gated_act = GpuBuffer::zeros(device, dtype, &[ple_hidden])?;
            g4::gelu_tanh_gate_mul(
                device, dtype, &mut gated_act, &gated, &per_layer_input_gpu, ple_hidden,
            )?;
            let mut projected = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::matvec_int4(
                device, dtype, &mut projected, &gated_act,
                &w.per_layer_projection.packed,
                &w.per_layer_projection.scale,
                &w.per_layer_projection.zero,
                ple_hidden, hidden_size,
                INT4_GROUP_SIZE, &mut self.counter,
            )?;
            let mut normed = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
            g4::rms_norm(
                device, dtype, &mut normed, &projected, Some(&w.post_per_layer_input_norm),
                eps, hidden_size,
            )?;
            let ple_residual_h = download_bf16(&ple_residual)?;
            let normed_h = download_bf16(&normed)?;
            let h_post_ple: Vec<f32> = ple_residual_h
                .iter()
                .zip(normed_h.iter())
                .map(|(a, b)| (a + b) * w.layer_scalar)
                .collect();

            h_running = upload_bf16(device, &[hidden_size], &h_post_ple)?;
        }

        let mut post_norm = GpuBuffer::zeros(device, dtype, &[hidden_size])?;
        g4::rms_norm(device, dtype, &mut post_norm, &h_running, Some(&self.final_norm_w), eps, hidden_size)?;

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
}
