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
//! 16 MiB). The test loads `mtp.*` tensors from the INT4 bake's raw BF16
//! MTP pass-through tensors at the path given by
//! `SUPERSONIC_QWEN36_MTP_MODEL_DIR`, matching the runtime path.
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
use model_store::BakedStore;
use runner::qwen36_moe_decode::{FullAttnKvCache, MtpLayerBuffers, MultiLayerGeom};
use runner::qwen36_moe_mtp::{
    alloc_mtp_chain_scratch, alloc_mtp_forward_scratch, run_mtp_draft_chain,
    run_mtp_draft_step, run_mtp_layer_step,
};
use runner::qwen36_moe_speculative::{
    run_speculative_decode_step, run_speculative_decode_step_batched, SpeculativeStepResult,
};
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

fn load_mtp_buffers_from_bake(
    store: &BakedStore,
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
        pre_fc_norm_hidden_w: store.load_to_gpu("mtp.pre_fc_norm_hidden.weight", ordinal)?,
        pre_fc_norm_embedding_w: store.load_to_gpu("mtp.pre_fc_norm_embedding.weight", ordinal)?,
        fc_w: store.load_to_gpu("mtp.fc.weight", ordinal)?,
        norm_w: store.load_to_gpu("mtp.norm.weight", ordinal)?,
        input_norm_w: store.load_to_gpu("mtp.layers.0.input_layernorm.weight", ordinal)?,
        post_attn_norm_w: store
            .load_to_gpu("mtp.layers.0.post_attention_layernorm.weight", ordinal)?,
        q_proj_w: store.load_to_gpu("mtp.layers.0.self_attn.q_proj.weight", ordinal)?,
        k_proj_w: store.load_to_gpu("mtp.layers.0.self_attn.k_proj.weight", ordinal)?,
        v_proj_w: store.load_to_gpu("mtp.layers.0.self_attn.v_proj.weight", ordinal)?,
        o_proj_w: store.load_to_gpu("mtp.layers.0.self_attn.o_proj.weight", ordinal)?,
        q_norm_w: store.load_to_gpu("mtp.layers.0.self_attn.q_norm.weight", ordinal)?,
        k_norm_w: store.load_to_gpu("mtp.layers.0.self_attn.k_norm.weight", ordinal)?,
        gate_w: store.load_to_gpu("mtp.layers.0.mlp.gate.weight", ordinal)?,
        gate_up_proj_w: store.load_to_gpu("mtp.layers.0.mlp.experts.gate_up_proj", ordinal)?,
        down_proj_w: store.load_to_gpu("mtp.layers.0.mlp.experts.down_proj", ordinal)?,
        shared_gate_proj_w: store
            .load_to_gpu("mtp.layers.0.mlp.shared_expert.gate_proj.weight", ordinal)?,
        shared_up_proj_w: store
            .load_to_gpu("mtp.layers.0.mlp.shared_expert.up_proj.weight", ordinal)?,
        shared_down_proj_w: store
            .load_to_gpu("mtp.layers.0.mlp.shared_expert.down_proj.weight", ordinal)?,
        shared_expert_gate_w: store
            .load_to_gpu("mtp.layers.0.mlp.shared_expert_gate.weight", ordinal)?,
        kv_cache,
    })
}

fn open_mtp_bake(model_dir: &Path) -> Result<BakedStore> {
    let bake_dir = model_store::bake_dir_int4(model_dir);
    BakedStore::open(&bake_dir)
        .with_context(|| format!("open INT4 bake at {}", bake_dir.display()))
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

    eprintln!(
        "[mtp parity] loading mtp.* tensors from INT4 bake under {} ...",
        model_dir.display()
    );
    let store = open_mtp_bake(&model_dir).expect("open INT4 bake");
    // Cache big enough to hold all draft steps if the test grew. K=1 here
    // is enough for the first step but we size for K=4 to give headroom.
    let kv_max_t = 4usize;
    let mut mtp = load_mtp_buffers_from_bake(&store, ordinal, &geom, kv_max_t)
        .expect("load MtpLayerBuffers from bake");
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

    // BF16 parity envelope: cos_sim ≥ 0.998 (industry-standard for
    // hidden-sized BF16 comparisons; 2048-element vectors don't
    // average out per-element rounding noise as much as the 248k-
    // element logits comparisons do), max_abs ≤ 1.0 (1 BF16 ULP at
    // magnitude 64 is 0.5; real-prefill state can hit those
    // magnitudes — synthetic noise stays well under). cos_sim is
    // the meaningful divergence signal; max_abs is a guard rail
    // against catastrophic mismatches.
    //
    // Observed across the local fixture set (synthetic K=3 + the 3
    // real-prefill fixtures from issue #88):
    //   synthetic:           cos_sim 0.9999947, max_abs 0.0078
    //   quick_brown_fox:     cos_sim 0.9994+,    max_abs ≤ 0.5
    //   factorial:           cos_sim 0.9994+,    max_abs ≤ 0.5
    //   once_upon:           cos_sim 0.9989+,    max_abs ≤ 0.6
    assert_parity("mtp.attn_out", &got_bytes, &want_bytes, 1.0, 0.998);
}

/// BF16-vs-BF16 parity helper: cosine similarity ≥ `cos_sim_floor` and
/// max-abs ≤ `max_abs_tol`. Same envelope as the multilayer parity test.
fn assert_parity(
    label: &str,
    got_bytes: &[u8],
    want_bytes: &[u8],
    max_abs_tol: f32,
    cos_sim_floor: f64,
) {
    assert_eq!(got_bytes.len(), want_bytes.len(), "{label}: byte length mismatch");
    let got = bf16_bytes_to_f32(got_bytes);
    let want = bf16_bytes_to_f32(want_bytes);
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
        "[mtp parity {label}] n={n} exact={exact} max_abs={max_abs:.5e} \
         mean_abs={mean_abs:.5e} cos_sim={cos_sim:.7}"
    );
    assert!(
        max_abs <= max_abs_tol,
        "{label}: max_abs {max_abs:.5e} exceeds {max_abs_tol:.5e}"
    );
    assert!(
        cos_sim >= cos_sim_floor,
        "{label}: cos_sim {cos_sim:.7} below floor {cos_sim_floor}"
    );
}

/// Phase 6.2c.3 end-to-end parity for the draft-step tail (post-norm +
/// tied lm_head + argmax). Validates `h_post`, `logits`, and the
/// greedy `draft_token_id` against the oracle's step-0 dump.
///
/// Loads the tied `lm_head.weight` (~970 MiB BF16) from the model_dir
/// in addition to the `mtp.*` tensors the layer-forward test already
/// loads — total GPU footprint ~2.6 GiB, fits on the 24 GiB local box.
#[test]
fn qwen36_moe_mtp_draft_step_matches_oracle() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MTP_ORACLE_JSON") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_ORACLE_JSON not set");
        return;
    };
    let Ok(model_dir) = std::env::var("SUPERSONIC_QWEN36_MTP_MODEL_DIR") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_MODEL_DIR not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);

    let raw = std::fs::read_to_string(&json_path).expect("read mtp oracle json");
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
    let want_h_post = b64(step0["h_post_bf16"].as_str().unwrap());
    let want_logits = b64(step0["logits_bf16"].as_str().unwrap());
    let want_draft = step0["draft_token_id"].as_u64().unwrap() as u32;

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    eprintln!(
        "[mtp draft] loading mtp.* from INT4 bake + tied lm_head.weight from safetensors under {} ...",
        model_dir.display()
    );
    let store = open_mtp_bake(&model_dir).expect("open INT4 bake");
    let shards = SafetensorsShards::open(&model_dir).expect("open safetensors");
    let kv_max_t = 4usize;
    let mut mtp = load_mtp_buffers_from_bake(&store, ordinal, &geom, kv_max_t)
        .expect("load mtp buffers");
    let lm_head_w = shards
        .load_bf16_to_gpu(ordinal, "lm_head.weight")
        .expect("load lm_head.weight");

    let mut scratch = alloc_mtp_forward_scratch(ordinal, &geom, kv_max_t)
        .expect("alloc mtp scratch");

    let fused_buf = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[geom.hidden as usize], &fused_bytes,
    ).expect("upload fused");
    let mut out_buf = GpuBuffer::zeros(
        ordinal, ScalarType::BF16, &[geom.hidden as usize],
    ).expect("alloc out");
    let mut h_post_buf = GpuBuffer::zeros(
        ordinal, ScalarType::BF16, &[geom.hidden as usize],
    ).expect("alloc h_post");

    eprintln!(
        "[mtp draft] running draft step (position={base_seq_len}, cache_pos=0)..."
    );
    let result = run_mtp_draft_step(
        ordinal, &geom, &mut mtp,
        /* position */ base_seq_len,
        /* cache_pos */ 0,
        &lm_head_w,
        &fused_buf, &mut out_buf, &mut h_post_buf,
        &mut scratch,
    ).expect("run_mtp_draft_step");

    // h_post + logits envelopes are loose (max_abs ≤ 1.0) because
    // real-prefill state produces larger magnitudes than synthetic
    // noise — 1 BF16 ULP at magnitude 64 is 0.5. cos_sim catches
    // kernel divergence; max_abs is a guard rail.
    //
    // The two cos_sim floors differ by tensor size:
    //   h_post  is `[hidden=2048]`   → 0.998 floor (small-tensor noise:
    //     once_upon's per-element BF16 rounding lands at cos_sim 0.9988)
    //   logits  is `[vocab=248320]`  → 0.999 floor (per-element rounding
    //     averages out across 121× more elements; all fixtures hit
    //     ≥ 0.9994)
    assert_parity(
        "mtp.h_post",
        &result.h_post_bytes, &want_h_post,
        /* max_abs */ 1.0, /* cos_sim_floor */ 0.998,
    );
    assert_parity(
        "mtp.logits",
        &result.logits_bytes, &want_logits,
        /* max_abs */ 1.0, /* cos_sim_floor */ 0.999,
    );
    if result.draft_token_id == want_draft {
        eprintln!("[mtp draft] draft_token_id={want_draft} (greedy argmax matches oracle)");
    } else {
        // Near-tied logits under BF16 + F32-accum noise can flip the
        // argmax even at 4-nines cos_sim. Log loudly but don't fail —
        // matches the relaxed lm_head test's behavior.
        eprintln!(
            "[mtp draft] draft_token_id={} differs from oracle's {} — \
             check the gap between top-1 and top-2 logits; expected when \
             they're within ~1 BF16 ULP",
            result.draft_token_id, want_draft
        );
    }
}

/// Phase 6.3a parity for the recurrent K-step MTP draft chain. Generates
/// `K = num_speculative_tokens` draft tokens from the same `(h_base_step0,
/// base_next_token_id)` seed the oracle uses, and compares the sequence
/// against the oracle's `draft_token_ids` list.
///
/// Loads `embed_tokens.weight` in addition to `lm_head.weight` and the
/// `mtp.*` tensors. Total GPU footprint ~3.6 GiB on the local fixture.
#[test]
fn qwen36_moe_mtp_draft_chain_matches_oracle() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MTP_ORACLE_JSON") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_ORACLE_JSON not set");
        return;
    };
    let Ok(model_dir) = std::env::var("SUPERSONIC_QWEN36_MTP_MODEL_DIR") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_MODEL_DIR not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);

    let raw = std::fs::read_to_string(&json_path).expect("read mtp oracle json");
    let json: Value = serde_json::from_str(&raw).expect("mtp oracle json parse");
    let geom = parse_geom(&json);
    let base_seq_len = json["base_seq_len"].as_i64().unwrap() as i32;
    let base_next_token_id = json["base_next_token_id"].as_i64().unwrap() as u32;
    let h_base_bytes = b64(json["h_base_step0_bf16"].as_str().unwrap());
    assert_eq!(h_base_bytes.len(), (geom.hidden as usize) * 2);

    let want_drafts: Vec<u32> = json["draft_token_ids"]
        .as_array()
        .expect("draft_token_ids")
        .iter()
        .map(|v| v.as_u64().expect("draft_token_id is integer") as u32)
        .collect();
    let k = want_drafts.len();
    assert!(k >= 1, "oracle has no draft tokens");
    eprintln!("[mtp chain] K={k} target tokens: {want_drafts:?}");

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    eprintln!(
        "[mtp chain] loading mtp.* from INT4 bake + embed_tokens + tied lm_head from safetensors under {} ...",
        model_dir.display()
    );
    let store = open_mtp_bake(&model_dir).expect("open INT4 bake");
    let shards = SafetensorsShards::open(&model_dir).expect("open safetensors");
    let mut mtp = load_mtp_buffers_from_bake(&store, ordinal, &geom, k.max(1))
        .expect("load mtp buffers");
    let embed_w = shards
        .load_bf16_to_gpu(ordinal, "model.language_model.embed_tokens.weight")
        .expect("load embed_tokens.weight");
    let lm_head_w = shards
        .load_bf16_to_gpu(ordinal, "lm_head.weight")
        .expect("load lm_head.weight");

    let mut forward_scratch = alloc_mtp_forward_scratch(ordinal, &geom, k.max(1))
        .expect("alloc mtp forward scratch");
    let mut chain_scratch = alloc_mtp_chain_scratch(ordinal, &geom)
        .expect("alloc mtp chain scratch");

    let h_base_buf = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[geom.hidden as usize], &h_base_bytes,
    ).expect("upload h_base");

    eprintln!(
        "[mtp chain] running {k}-step recurrence (base_position={base_seq_len}, \
         first_token_id={base_next_token_id})..."
    );
    let records = run_mtp_draft_chain(
        ordinal, &geom, &mut mtp, base_seq_len,
        &h_base_buf, base_next_token_id, k,
        &embed_w, &lm_head_w,
        &mut forward_scratch, &mut chain_scratch,
    ).expect("run_mtp_draft_chain");

    let got_drafts: Vec<u32> = records.iter().map(|r| r.draft_token_id).collect();
    eprintln!("[mtp chain] got: {got_drafts:?}  want: {want_drafts:?}");

    // Token-level parity. The chain MUST reproduce the oracle's draft
    // sequence bit-for-bit on the test fixture: argmax-flip cascading
    // is the failure mode this test exists to catch (a single flipped
    // token at step k makes step k+1's `e_in = embed[wrong_token]`,
    // which derails the rest of the chain). On the local fixture the
    // kernel is stable enough that BF16 rounding doesn't flip any
    // top-1 across K=3 steps; if a future fixture lands on a near-tied
    // top-1 and legitimately flips, that's a real signal worth
    // investigating before relaxing — Phase 6.2c.3's single-step test
    // tolerates the same flip class because at that level there's no
    // cascade to worry about.
    assert_eq!(records.len(), k, "chain returned wrong number of records");
    if let Some(idx) = got_drafts.iter()
        .zip(want_drafts.iter())
        .position(|(g, w)| g != w)
    {
        panic!(
            "MTP chain diverges at step {idx}: got {} want {} \
             (full got={got_drafts:?} want={want_drafts:?}). A flipped \
             argmax at step {idx} cascades through the rest of the \
             chain because `e_in[step{}]` reads `embed_tokens[\
             wrong_token]`. Investigate the per-step logit gap before \
             relaxing.",
            got_drafts[idx], want_drafts[idx], idx + 1
        );
    }

    // Per-step logits parity. Token agreement is enforced above, so
    // every step's `e_in` matches the oracle and the per-step logits
    // should hold cos_sim ≥ 0.999 (vocab=248320 averages out per-
    // element BF16 rounding). The max-abs envelope is loose: peak
    // logit magnitudes from real-prefill state can land at 30-64
    // (vs ~8-16 for synthetic noise), where 1 BF16 ULP is 0.25-0.5.
    // cos_sim is the meaningful divergence signal — max_abs is the
    // guard rail against catastrophic mismatches.
    //
    // Observed across the local fixtures (synthetic K=3 + #88
    // real-prefill set): max_abs ≤ 0.778, cos_sim ≥ 0.9992.
    for (i, want_step) in json["steps"].as_array().expect("steps").iter().enumerate() {
        if i >= k { break; }
        let want_logits = b64(want_step["logits_bf16"].as_str().unwrap());
        assert_parity(
            &format!("mtp.chain.step{i}.logits"),
            &records[i].logits_bytes, &want_logits,
            /* max_abs */ 1.0, /* cos_sim_floor */ 0.999,
        );
    }
}

/// Phase 6.3c integration test: validate `run_speculative_decode_step`
/// orchestration end-to-end using the real MTP chain on the loaded
/// `mtp.*` weights, plus a controlled mock `base_step` closure that
/// returns canned predictions. The mock is what makes this testable
/// on the local 24 GiB box without needing the full ~17 GiB INT4 base
/// model loaded — we just need to confirm the driver's verify loop +
/// accept-prefix logic produces the right `emitted_tokens` for known
/// mock-prediction patterns.
///
/// Three passes cover the three accept-prefix outcomes:
///   - Pass 1: mock agrees with all drafts → full-accept + bonus.
///   - Pass 2: mock disagrees on draft index 1 → partial-accept.
///   - Pass 3: mock disagrees immediately → zero-accept.
#[test]
fn qwen36_moe_speculative_driver_orchestration() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MTP_ORACLE_JSON") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_ORACLE_JSON not set");
        return;
    };
    let Ok(model_dir) = std::env::var("SUPERSONIC_QWEN36_MTP_MODEL_DIR") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_MODEL_DIR not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);

    let raw = std::fs::read_to_string(&json_path).expect("read mtp oracle json");
    let json: Value = serde_json::from_str(&raw).expect("mtp oracle json parse");
    let geom = parse_geom(&json);
    let base_seq_len = json["base_seq_len"].as_i64().unwrap() as i32;
    let base_next_token_id = json["base_next_token_id"].as_i64().unwrap() as u32;
    let h_base_bytes = b64(json["h_base_step0_bf16"].as_str().unwrap());
    let want_drafts: Vec<u32> = json["draft_token_ids"]
        .as_array().expect("draft_token_ids")
        .iter().map(|v| v.as_u64().expect("u64") as u32).collect();
    let k = want_drafts.len();
    eprintln!("[spec] K={k} oracle drafts={want_drafts:?}");

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    eprintln!("[spec] loading mtp.* from INT4 bake + embed_tokens + tied lm_head from safetensors ...");
    let store = open_mtp_bake(&model_dir).expect("open INT4 bake");
    let shards = SafetensorsShards::open(&model_dir).expect("open safetensors");
    let mut mtp = load_mtp_buffers_from_bake(&store, ordinal, &geom, k.max(1))
        .expect("load mtp buffers");
    let embed_w = shards
        .load_bf16_to_gpu(ordinal, "model.language_model.embed_tokens.weight")
        .expect("load embed_tokens.weight");
    let lm_head_w = shards
        .load_bf16_to_gpu(ordinal, "lm_head.weight")
        .expect("load lm_head.weight");

    let mut forward_scratch = alloc_mtp_forward_scratch(ordinal, &geom, k.max(1))
        .expect("alloc fwd scratch");
    let mut chain_scratch = alloc_mtp_chain_scratch(ordinal, &geom)
        .expect("alloc chain scratch");

    let synth_fh = vec![0u8; (geom.hidden as usize) * 2];

    // ---- Pass 1: full-accept ----
    let bonus_token_pass1: u32 = 999;
    let mut step_idx = 0usize;
    let want_drafts_p1 = want_drafts.clone();
    let synth_fh_p1 = synth_fh.clone();
    let base_step_p1 = move |_pos: i32, _input: u32|
        -> anyhow::Result<(u32, Vec<u8>)>
    {
        let predicted = if step_idx < want_drafts_p1.len() {
            want_drafts_p1[step_idx]
        } else {
            bonus_token_pass1
        };
        step_idx += 1;
        Ok((predicted, synth_fh_p1.clone()))
    };
    eprintln!("[spec] pass 1 (full-accept)");
    let r1 = run_speculative_decode_step(
        ordinal, &geom, &mut mtp,
        &mut forward_scratch, &mut chain_scratch,
        &embed_w, &lm_head_w,
        &h_base_bytes, base_next_token_id, base_seq_len, k,
        base_step_p1,
    ).expect("pass 1 run");
    let mut want1 = want_drafts.clone();
    want1.push(bonus_token_pass1);
    assert_eq!(r1.emitted_tokens, want1);
    assert_eq!(r1.n_accepted, k);
    eprintln!("[spec] pass 1: emitted={:?} n_accepted={}", r1.emitted_tokens, r1.n_accepted);

    // ---- Pass 2: partial-accept (k>=2 only) ----
    if k >= 2 {
        let bad_pred: u32 = 12345;
        let mut step_idx = 0usize;
        let first_match = want_drafts[0];
        let synth_fh_p2 = synth_fh.clone();
        let base_step_p2 = move |_pos: i32, _input: u32|
            -> anyhow::Result<(u32, Vec<u8>)>
        {
            let predicted = if step_idx == 0 { first_match } else { bad_pred };
            step_idx += 1;
            Ok((predicted, synth_fh_p2.clone()))
        };
        eprintln!("[spec] pass 2 (reject at k=1)");
        let r2 = run_speculative_decode_step(
            ordinal, &geom, &mut mtp,
            &mut forward_scratch, &mut chain_scratch,
            &embed_w, &lm_head_w,
            &h_base_bytes, base_next_token_id, base_seq_len, k,
            base_step_p2,
        ).expect("pass 2 run");
        assert_eq!(r2.emitted_tokens, vec![want_drafts[0], bad_pred]);
        assert_eq!(r2.n_accepted, 1);
        eprintln!("[spec] pass 2: emitted={:?} n_accepted={}", r2.emitted_tokens, r2.n_accepted);
    } else {
        eprintln!("[spec] pass 2 skipped (K={k} too small)");
    }

    // ---- Pass 3: zero-accept (immediate reject) ----
    let bad_pred: u32 = 67890;
    let synth_fh_p3 = synth_fh.clone();
    let base_step_p3 = move |_pos: i32, _input: u32|
        -> anyhow::Result<(u32, Vec<u8>)>
    {
        Ok((bad_pred, synth_fh_p3.clone()))
    };
    eprintln!("[spec] pass 3 (reject at k=0)");
    let r3 = run_speculative_decode_step(
        ordinal, &geom, &mut mtp,
        &mut forward_scratch, &mut chain_scratch,
        &embed_w, &lm_head_w,
        &h_base_bytes, base_next_token_id, base_seq_len, k,
        base_step_p3,
    ).expect("pass 3 run");
    assert_eq!(r3.emitted_tokens, vec![bad_pred]);
    assert_eq!(r3.n_accepted, 0);
    eprintln!("[spec] pass 3: emitted={:?} n_accepted={}", r3.emitted_tokens, r3.n_accepted);

    // The driver should produce a final-hidden buffer of the right size
    // regardless of which pass ran.
    assert_eq!(r3.final_hidden_bytes.len(), (geom.hidden as usize) * 2);
    let _ = SpeculativeStepResult { ..r3 };

    // ---- Pass 4: K=0 fallback (no speculation; emit one base token) ----
    //
    // Contract: `emitted_tokens` must always be non-empty (the
    // engine relies on `position += emitted.len()` for forward
    // progress). When K=0 the driver still runs one base step and
    // emits its prediction — equivalent to plain greedy decode.
    let k0_pred: u32 = 4242;
    let synth_fh_p4 = synth_fh.clone();
    let mut p4_calls = 0usize;
    let base_step_p4 = move |_pos: i32, _input: u32|
        -> anyhow::Result<(u32, Vec<u8>)>
    {
        p4_calls += 1;
        assert_eq!(p4_calls, 1, "K=0 must run exactly one base step");
        Ok((k0_pred, synth_fh_p4.clone()))
    };
    eprintln!("[spec] pass 4 (K=0 fallback)");
    let r4 = run_speculative_decode_step(
        ordinal, &geom, &mut mtp,
        &mut forward_scratch, &mut chain_scratch,
        &embed_w, &lm_head_w,
        &h_base_bytes, base_next_token_id, base_seq_len, /* num_drafts */ 0,
        base_step_p4,
    ).expect("pass 4 run");
    assert_eq!(r4.emitted_tokens, vec![k0_pred],
        "K=0 must still emit exactly one token (the contract is non-empty)");
    assert_eq!(r4.n_accepted, 0);
    assert_eq!(r4.final_hidden_bytes.len(), (geom.hidden as usize) * 2);
    eprintln!("[spec] pass 4: emitted={:?} n_accepted={}", r4.emitted_tokens, r4.n_accepted);
}

/// Phase 6.4b orchestration test for the batched closure variant.
/// Same accept-prefix logic as the per-step driver, but the closure
/// is called ONCE with K+1 (position, input) pairs and returns K+1
/// predictions in one go. Mocks the closure with canned predictions
/// to cover full-accept, partial-accept, and zero-accept paths.
///
/// IMPORTANT: this orchestration test uses a mock base closure so
/// the linear-attn-state-pollution issue (real chains advancing
/// state past the rejection point) does NOT show here — that's a
/// real-engine concern Phase 6.4c will address. The orchestration
/// itself (accept-prefix walk, final_hidden_bytes selection) is
/// what's under test.
#[test]
fn qwen36_moe_speculative_driver_orchestration_batched() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MTP_ORACLE_JSON") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_ORACLE_JSON not set");
        return;
    };
    let Ok(model_dir) = std::env::var("SUPERSONIC_QWEN36_MTP_MODEL_DIR") else {
        eprintln!("skip: SUPERSONIC_QWEN36_MTP_MODEL_DIR not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);

    let raw = std::fs::read_to_string(&json_path).expect("read mtp oracle json");
    let json: Value = serde_json::from_str(&raw).expect("mtp oracle json parse");
    let geom = parse_geom(&json);
    let base_seq_len = json["base_seq_len"].as_i64().unwrap() as i32;
    let base_next_token_id = json["base_next_token_id"].as_i64().unwrap() as u32;
    let h_base_bytes = b64(json["h_base_step0_bf16"].as_str().unwrap());
    let want_drafts: Vec<u32> = json["draft_token_ids"]
        .as_array().expect("draft_token_ids")
        .iter().map(|v| v.as_u64().expect("u64") as u32).collect();
    let k = want_drafts.len();
    eprintln!("[spec batched] K={k} oracle drafts={want_drafts:?}");

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    let store = open_mtp_bake(&model_dir).expect("open INT4 bake");
    let shards = SafetensorsShards::open(&model_dir).expect("open safetensors");
    let mut mtp = load_mtp_buffers_from_bake(&store, ordinal, &geom, k.max(1))
        .expect("load mtp buffers");
    let embed_w = shards
        .load_bf16_to_gpu(ordinal, "model.language_model.embed_tokens.weight")
        .expect("load embed_tokens.weight");
    let lm_head_w = shards
        .load_bf16_to_gpu(ordinal, "lm_head.weight")
        .expect("load lm_head.weight");
    let mut forward_scratch = alloc_mtp_forward_scratch(ordinal, &geom, k.max(1))
        .expect("alloc fwd scratch");
    let mut chain_scratch = alloc_mtp_chain_scratch(ordinal, &geom)
        .expect("alloc chain scratch");

    let synth_fh = vec![0u8; (geom.hidden as usize) * 2];

    // ---- Pass 1: full-accept (K matches + bonus) ----
    let bonus_token: u32 = 999;
    let want_drafts_p1 = want_drafts.clone();
    let synth_fh_p1 = synth_fh.clone();
    let base_step_batched_p1 = move |inputs: &[(i32, u32)]|
        -> anyhow::Result<Vec<(u32, Vec<u8>)>>
    {
        assert_eq!(inputs.len(), k + 1, "expected K+1 inputs");
        // Predictions: drafts[0..K] match, then bonus for K-th.
        let mut out: Vec<(u32, Vec<u8>)> = Vec::with_capacity(k + 1);
        for i in 0..k {
            out.push((want_drafts_p1[i], synth_fh_p1.clone()));
        }
        out.push((bonus_token, synth_fh_p1.clone()));
        Ok(out)
    };
    let r1 = run_speculative_decode_step_batched(
        ordinal, &geom, &mut mtp,
        &mut forward_scratch, &mut chain_scratch,
        &embed_w, &lm_head_w,
        &h_base_bytes, base_next_token_id, base_seq_len, k,
        base_step_batched_p1,
    ).expect("pass 1 batched");
    let mut want1 = want_drafts.clone();
    want1.push(bonus_token);
    assert_eq!(r1.emitted_tokens, want1, "full-accept emit");
    assert_eq!(r1.n_accepted, k);
    eprintln!("[spec batched] pass 1: emitted={:?} n_accepted={}",
        r1.emitted_tokens, r1.n_accepted);

    // ---- Pass 2: partial-accept (reject at k=1, K≥2) ----
    if k >= 2 {
        let bad_pred: u32 = 12345;
        let first_match = want_drafts[0];
        let synth_fh_p2 = synth_fh.clone();
        let base_step_batched_p2 = move |inputs: &[(i32, u32)]|
            -> anyhow::Result<Vec<(u32, Vec<u8>)>>
        {
            assert_eq!(inputs.len(), k + 1);
            let mut out: Vec<(u32, Vec<u8>)> = Vec::with_capacity(k + 1);
            // Index 0 matches drafts[0]
            out.push((first_match, synth_fh_p2.clone()));
            // Index 1 mismatches drafts[1] → reject.
            out.push((bad_pred, synth_fh_p2.clone()));
            // Remaining filler (would not be inspected by accept-prefix
            // logic but must exist to satisfy the contract).
            for _ in 2..(k + 1) {
                out.push((0u32, synth_fh_p2.clone()));
            }
            Ok(out)
        };
        let r2 = run_speculative_decode_step_batched(
            ordinal, &geom, &mut mtp,
            &mut forward_scratch, &mut chain_scratch,
            &embed_w, &lm_head_w,
            &h_base_bytes, base_next_token_id, base_seq_len, k,
            base_step_batched_p2,
        ).expect("pass 2 batched");
        assert_eq!(r2.emitted_tokens, vec![want_drafts[0], bad_pred],
            "partial-accept emit drafts[0] + corrected");
        assert_eq!(r2.n_accepted, 1);
        eprintln!("[spec batched] pass 2: emitted={:?} n_accepted={}",
            r2.emitted_tokens, r2.n_accepted);
    }

    // ---- Pass 3: zero-accept (immediate reject at k=0) ----
    let bad_pred: u32 = 67890;
    let synth_fh_p3 = synth_fh.clone();
    let base_step_batched_p3 = move |inputs: &[(i32, u32)]|
        -> anyhow::Result<Vec<(u32, Vec<u8>)>>
    {
        assert_eq!(inputs.len(), k + 1);
        let mut out: Vec<(u32, Vec<u8>)> = Vec::with_capacity(k + 1);
        out.push((bad_pred, synth_fh_p3.clone())); // immediate reject
        for _ in 1..(k + 1) {
            out.push((0u32, synth_fh_p3.clone()));
        }
        Ok(out)
    };
    let r3 = run_speculative_decode_step_batched(
        ordinal, &geom, &mut mtp,
        &mut forward_scratch, &mut chain_scratch,
        &embed_w, &lm_head_w,
        &h_base_bytes, base_next_token_id, base_seq_len, k,
        base_step_batched_p3,
    ).expect("pass 3 batched");
    assert_eq!(r3.emitted_tokens, vec![bad_pred]);
    assert_eq!(r3.n_accepted, 0);
    eprintln!("[spec batched] pass 3: emitted={:?} n_accepted={}",
        r3.emitted_tokens, r3.n_accepted);

    // ---- Pass 4: K=0 fallback ----
    let k0_pred: u32 = 4242;
    let synth_fh_p4 = synth_fh.clone();
    let base_step_batched_p4 = move |inputs: &[(i32, u32)]|
        -> anyhow::Result<Vec<(u32, Vec<u8>)>>
    {
        assert_eq!(inputs.len(), 1, "K=0 fallback runs exactly one base step");
        Ok(vec![(k0_pred, synth_fh_p4.clone())])
    };
    let r4 = run_speculative_decode_step_batched(
        ordinal, &geom, &mut mtp,
        &mut forward_scratch, &mut chain_scratch,
        &embed_w, &lm_head_w,
        &h_base_bytes, base_next_token_id, base_seq_len, /* K */ 0,
        base_step_batched_p4,
    ).expect("pass 4 batched");
    assert_eq!(r4.emitted_tokens, vec![k0_pred]);
    assert_eq!(r4.n_accepted, 0);
    eprintln!("[spec batched] pass 4 (K=0): emitted={:?} n_accepted={}",
        r4.emitted_tokens, r4.n_accepted);
}
