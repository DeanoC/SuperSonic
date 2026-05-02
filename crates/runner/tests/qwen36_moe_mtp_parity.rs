//! Phase 6.2c.2 parity test for the SuperSonic-side MTP layer forward.
//!
//! Drives [`runner::qwen36_moe_mtp::run_mtp_layer_step`] against the
//! Phase 6.2a Python oracle. Reads `steps[0].fused_bf16` (the pre-fusion
//! kernel's output, validated bit-exact in Phase 6.2c.1) as input,
//! produces the layer's post-residual output, and compares against the
//! oracle's `steps[0].attn_out_bf16` (which despite the name is the FULL
//! layer output — vLLM's HF `Qwen3_5MoeDecoderLayer` returns
//! `residual + attn + mlp`).
//!
//! The MTP MoE FFN is ~1.6 GiB BF16 — too large to round-trip through
//! the oracle JSON (the existing `prefusion_weights` block is already
//! 16 MiB). The test instead loads `mtp.*` tensors directly from the
//! model's safetensors at the path given by
//! `SUPERSONIC_QWEN36_MTP_MODEL_DIR`, exactly mirroring the Phase 6.2a
//! oracle's load path.
//!
//! Skipped silently when either env var isn't set so CI / non-HIP
//! machines stay green. To run locally:
//!
//! ```bash
//! .venv-bake/bin/python oracle/qwen36_moe_mtp_oracle.py \
//!     --model-dir /path/to/Qwen3.6-35B-A3B \
//!     --num-speculative-tokens 1 --seed 42 \
//!     --out /tmp/qwen36_mtp.json
//! SUPERSONIC_QWEN36_MTP_ORACLE_JSON=/tmp/qwen36_mtp.json \
//!   SUPERSONIC_QWEN36_MTP_MODEL_DIR=/path/to/Qwen3.6-35B-A3B \
//!   cargo test --release -p runner --test qwen36_moe_mtp_parity \
//!     -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use gpu_hal::{copy_d2h, is_backend_compiled, set_backend, Backend, GpuBuffer, ScalarType};
use memmap2::Mmap;
use runner::qwen36_moe_decode::{FullAttnKvCache, MtpLayerBuffers, MultiLayerGeom};
use runner::qwen36_moe_mtp::{alloc_mtp_forward_scratch, run_mtp_layer_step};
use safetensors::SafeTensors;
use serde_json::Value;

fn b64(input: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .expect("base64 decode")
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
            f32::from_bits(bits << 16)
        })
        .collect()
}

/// Inline sharded-safetensors loader. Mirrors `UnbakedLoader` in
/// `gemma4_engine.rs` (which is private). We only need to read 19
/// `mtp.*` tensors, all of them BF16.
struct SafetensorsShards {
    shards: Vec<Mmap>,
    /// tensor name → shard index in `shards`.
    index: BTreeMap<String, usize>,
}

impl SafetensorsShards {
    fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let raw: Value = serde_json::from_str(
            &std::fs::read_to_string(&index_path)
                .with_context(|| format!("read {}", index_path.display()))?,
        )?;
        let weight_map = raw["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("weight_map missing in {}", index_path.display()))?;

        // Collect distinct shard filenames in deterministic order.
        let mut filename_to_idx: BTreeMap<String, usize> = BTreeMap::new();
        let mut shard_files: Vec<String> = Vec::new();
        for v in weight_map.values() {
            let filename = v.as_str().unwrap_or("").to_string();
            if !filename_to_idx.contains_key(&filename) {
                filename_to_idx.insert(filename.clone(), shard_files.len());
                shard_files.push(filename);
            }
        }

        let mut shards: Vec<Mmap> = Vec::with_capacity(shard_files.len());
        for filename in &shard_files {
            let p = model_dir.join(filename);
            let f = File::open(&p)
                .with_context(|| format!("open shard {}", p.display()))?;
            shards.push(unsafe { Mmap::map(&f)? });
        }
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        for (name, fname) in weight_map {
            if let Some(&idx) = filename_to_idx.get(fname.as_str().unwrap_or("")) {
                index.insert(name.clone(), idx);
            }
        }
        Ok(Self { shards, index })
    }

    fn load_bf16_to_gpu(&self, ordinal: usize, name: &str) -> Result<GpuBuffer> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("safetensors index missing tensor: {name}"))?;
        let st = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = st.tensor(name)?;
        if view.dtype() != safetensors::Dtype::BF16 {
            return Err(anyhow!(
                "tensor {name} dtype {:?}, expected BF16",
                view.dtype()
            ));
        }
        Ok(GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            view.shape(),
            view.data(),
        )?)
    }
}

fn load_mtp_buffers_from_safetensors(
    shards: &SafetensorsShards,
    ordinal: usize,
    geom: &MultiLayerGeom,
    kv_max_t: usize,
) -> Result<MtpLayerBuffers> {
    let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
    let kv_cache = if kv_max_t > 0 {
        Some(FullAttnKvCache {
            k: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                .context("alloc mtp kv_cache_k")?,
            v: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                .context("alloc mtp kv_cache_v")?,
            kv_max_t: kv_max_t as i32,
        })
    } else {
        None
    };

    Ok(MtpLayerBuffers {
        pre_fc_norm_hidden_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.pre_fc_norm_hidden.weight")?,
        pre_fc_norm_embedding_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.pre_fc_norm_embedding.weight")?,
        fc_w: shards.load_bf16_to_gpu(ordinal, "mtp.fc.weight")?,
        norm_w: shards.load_bf16_to_gpu(ordinal, "mtp.norm.weight")?,
        input_norm_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.input_layernorm.weight")?,
        post_attn_norm_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.post_attention_layernorm.weight")?,
        q_proj_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.self_attn.q_proj.weight")?,
        k_proj_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.self_attn.k_proj.weight")?,
        v_proj_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.self_attn.v_proj.weight")?,
        o_proj_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.self_attn.o_proj.weight")?,
        q_norm_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.self_attn.q_norm.weight")?,
        k_norm_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.self_attn.k_norm.weight")?,
        gate_w: shards.load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.gate.weight")?,
        gate_up_proj_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.experts.gate_up_proj")?,
        down_proj_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.experts.down_proj")?,
        shared_gate_proj_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.shared_expert.gate_proj.weight")?,
        shared_up_proj_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.shared_expert.up_proj.weight")?,
        shared_down_proj_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.shared_expert.down_proj.weight")?,
        shared_expert_gate_w: shards
            .load_bf16_to_gpu(ordinal, "mtp.layers.0.mlp.shared_expert_gate.weight")?,
        kv_cache,
    })
}

fn parse_geom(json: &Value) -> MultiLayerGeom {
    let cfg = &json["config"];
    let head_dim = cfg["head_dim"].as_i64().unwrap() as i32;
    let partial = cfg["partial_rotary_factor"].as_f64().unwrap() as f32;
    let rotary_dim = (head_dim as f32 * partial) as i32;
    MultiLayerGeom {
        hidden: cfg["hidden"].as_i64().unwrap() as i32,
        vocab: cfg["vocab"].as_i64().unwrap() as i32,
        // num_layers / linear-attn fields aren't read by the MTP path; the
        // MoE FFN + full-attn fields below are what `run_mtp_layer_step`
        // actually consumes.
        num_layers: 1,
        rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,

        num_attention_heads: cfg["num_attention_heads"].as_i64().unwrap() as i32,
        num_kv_heads: cfg["num_kv_heads"].as_i64().unwrap() as i32,
        head_dim,
        rotary_dim,
        rope_theta: cfg["rope_theta"].as_f64().unwrap() as f32,

        // Linear-attn fields — unused by MTP, set to 0.
        num_k_heads: 0,
        num_v_heads: 0,
        head_k_dim: 0,
        head_v_dim: 0,
        conv_kernel_dim: 0,

        num_experts: cfg["num_experts"].as_i64().unwrap() as i32,
        moe_intermediate: cfg["moe_intermediate_size"].as_i64().unwrap() as i32,
        shared_intermediate: cfg["shared_expert_intermediate_size"].as_i64().unwrap() as i32,
        top_k: cfg["top_k"].as_i64().unwrap() as i32,
    }
}

#[test]
fn qwen36_moe_mtp_layer_forward_matches_oracle() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MTP_ORACLE_JSON") else {
        eprintln!(
            "skip: SUPERSONIC_QWEN36_MTP_ORACLE_JSON not set. Generate a \
             fixture with `python oracle/qwen36_moe_mtp_oracle.py --model-dir \
             <Qwen3.6-35B-A3B> --num-speculative-tokens 1 --out /tmp/qwen36_mtp.json` \
             and re-run with both env vars."
        );
        return;
    };
    let Ok(model_dir) = std::env::var("SUPERSONIC_QWEN36_MTP_MODEL_DIR") else {
        eprintln!(
            "skip: SUPERSONIC_QWEN36_MTP_MODEL_DIR not set. Point this at the \
             same model_dir the oracle used (the test loads `mtp.*` tensors \
             directly from safetensors — they're too big to round-trip through \
             the oracle JSON)."
        );
        return;
    };
    let model_dir = PathBuf::from(model_dir);

    let raw =
        std::fs::read_to_string(&json_path).expect("read mtp oracle json");
    let json: Value = serde_json::from_str(&raw).expect("mtp oracle json parse");
    assert_eq!(
        json["schema"].as_str().unwrap_or(""),
        "qwen36-moe-mtp-oracle-v1",
        "oracle JSON schema mismatch — regenerate with the Phase 6.2a oracle."
    );

    let geom = parse_geom(&json);
    let base_seq_len = json["base_seq_len"].as_i64().unwrap() as i32;
    let step0 = &json["steps"][0];
    let fused_bytes = b64(step0["fused_bf16"].as_str().unwrap());
    let want_bytes = b64(step0["attn_out_bf16"].as_str().unwrap());
    assert_eq!(fused_bytes.len(), (geom.hidden as usize) * 2);
    assert_eq!(want_bytes.len(), (geom.hidden as usize) * 2);

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    eprintln!("[mtp parity] loading mtp.* tensors from {} ...", model_dir.display());
    let shards = SafetensorsShards::open(&model_dir).expect("open safetensors");
    // Cache big enough to hold all draft steps if the test grew. K=1 here
    // is enough for the first step but we size for K=4 to give headroom.
    let kv_max_t = 4usize;
    let mut mtp = load_mtp_buffers_from_safetensors(&shards, ordinal, &geom, kv_max_t)
        .expect("load MtpLayerBuffers from safetensors");
    let mut scratch = alloc_mtp_forward_scratch(ordinal, &geom, kv_max_t)
        .expect("alloc mtp scratch");

    let fused_buf = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[geom.hidden as usize], &fused_bytes,
    ).expect("upload fused");
    let mut out_buf = GpuBuffer::zeros(
        ordinal, ScalarType::BF16, &[geom.hidden as usize],
    ).expect("alloc out");

    eprintln!("[mtp parity] running layer forward (position={base_seq_len}, cache_pos=0)...");
    run_mtp_layer_step(
        ordinal, &geom, &mut mtp,
        /* position */ base_seq_len,
        /* cache_pos */ 0,
        &fused_buf, &mut out_buf,
        &mut scratch,
    ).expect("run_mtp_layer_step");

    let mut got_bytes = vec![0u8; (geom.hidden as usize) * 2];
    copy_d2h(
        ordinal, got_bytes.as_mut_ptr() as *mut _,
        out_buf.as_ptr(), got_bytes.len(),
    ).expect("d2h out");

    let got = bf16_bytes_to_f32(&got_bytes);
    let want = bf16_bytes_to_f32(&want_bytes);
    let n = got.len();
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut got_sq = 0.0f64;
    let mut want_sq = 0.0f64;
    let mut exact = 0usize;
    for i in 0..n {
        let d = (got[i] - want[i]).abs();
        if d == 0.0 { exact += 1; }
        max_abs = max_abs.max(d);
        sum_abs += d;
        dot += got[i] as f64 * want[i] as f64;
        got_sq += (got[i] as f64).powi(2);
        want_sq += (want[i] as f64).powi(2);
    }
    let cos_sim = dot / (got_sq.sqrt() * want_sq.sqrt() + 1e-30);
    let mean_abs = sum_abs / n as f32;
    eprintln!(
        "[mtp parity attn_out] n={n} exact={exact} max_abs={max_abs:.5e} \
         mean_abs={mean_abs:.5e} cos_sim={cos_sim:.7}"
    );

    // Same envelope as the multi-layer parity test: cos_sim ≥ 0.999, max
    // abs ≤ 0.05 BF16 (the per-block tests' ULP envelope at hidden=2048
    // matvec scale).
    assert!(
        cos_sim >= 0.999,
        "MTP layer cos_sim {cos_sim:.7} below floor 0.999 — likely a \
         kernel/weights/cache_pos mismatch, not BF16 noise"
    );
    assert!(
        max_abs <= 0.05,
        "MTP layer max_abs {max_abs:.5e} exceeds 0.05 envelope"
    );
}
