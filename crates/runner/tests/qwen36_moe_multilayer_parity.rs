//! PR 4c step 2 multi-layer parity test for Qwen3.6-MoE.
//!
//! Loads the multi-layer Python oracle's JSON payload (produced by
//! `oracle/qwen36_moe_multilayer_oracle.py`), uploads each layer's BF16
//! weights + initial linear-attn state to the GPU, runs the chained decode
//! via [`runner::qwen36_moe_decode::run_chained_decode`], applies the host-
//! side final RMSnorm + lm_head, and compares against the oracle's
//! `intermediates_per_layer` + `final_hidden` + `logits` (cos_sim ≥ 0.999
//! per the PR 4c acceptance criteria; same ≤0.05 max-abs envelope as the
//! per-block tests).
//!
//! Skipped silently when `SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON` isn't
//! set so CI / non-HIP machines stay green. To run locally:
//!
//! ```bash
//! ~/venvs/rocm/bin/python oracle/qwen36_moe_multilayer_oracle.py \
//!     --mode synthetic --num-layers 4 --out /tmp/qwen36_ml.json
//! SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON=/tmp/qwen36_ml.json \
//!   cargo test --release -p runner --test qwen36_moe_multilayer_parity \
//!     -- --nocapture
//! ```
//!
//! Only runs when the HIP backend is compiled (PR 4c is HIP-only —
//! `kernels/qwen36_moe.hip` is the only compiled implementation, per
//! `~/.claude/.../memory/hardware_hip_only.md` and CLAUDE.md).
//! The `supersonic_backend_hip` rustc cfg is set by `gpu-hal` and
//! `kernel-ffi` build scripts but doesn't propagate to the `runner` crate's
//! integration tests; we gate at runtime via [`gpu_hal::is_backend_compiled`]
//! so this file always builds and skips cleanly when HIP isn't available.

use base64::Engine;
use gpu_hal::{is_backend_compiled, set_backend, Backend, GpuBuffer, ScalarType};
use runner::qwen36_moe_decode::{
    bf16_bytes_to_f32, host_final_norm_lm_head, is_full_attn_layer, run_chained_decode,
    AttnLayerBuffers, FfnInt4Sidecars, FfnLayerBuffers, FullAttnInt4Sidecars, LayerBuffers,
    LinearAttnInt4Sidecars, MultiLayerGeom,
};
use serde_json::Value;

fn b64(input: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .expect("base64 decode")
}

fn b64_field(obj: &Value, name: &str) -> Vec<u8> {
    let s = obj
        .get(name)
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("oracle JSON missing field {name}"));
    b64(s)
}

fn parse_geom(json: &Value) -> MultiLayerGeom {
    let cfg = &json["config"];
    let attn = &cfg["attn"];
    let lin = &cfg["lin"];
    let ffn = &cfg["ffn"];
    MultiLayerGeom {
        hidden: cfg["hidden"].as_i64().unwrap() as i32,
        vocab: cfg["vocab"].as_i64().unwrap() as i32,
        num_layers: json["num_layers"].as_i64().unwrap() as i32,
        rms_norm_eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        num_attention_heads: attn["num_attention_heads"].as_i64().unwrap() as i32,
        num_kv_heads: attn["num_kv_heads"].as_i64().unwrap() as i32,
        head_dim: attn["head_dim"].as_i64().unwrap() as i32,
        rotary_dim: attn["rotary_dim"].as_i64().unwrap() as i32,
        rope_theta: attn["rope_theta"].as_f64().unwrap() as f32,
        num_k_heads: lin["num_k_heads"].as_i64().unwrap() as i32,
        num_v_heads: lin["num_v_heads"].as_i64().unwrap() as i32,
        head_k_dim: lin["head_k_dim"].as_i64().unwrap() as i32,
        head_v_dim: lin["head_v_dim"].as_i64().unwrap() as i32,
        conv_kernel_dim: lin["conv_kernel_dim"].as_i64().unwrap() as i32,
        num_experts: ffn["num_experts"].as_i64().unwrap() as i32,
        moe_intermediate: ffn["moe_intermediate"].as_i64().unwrap() as i32,
        shared_intermediate: ffn["shared_intermediate"].as_i64().unwrap() as i32,
        top_k: ffn["top_k"].as_i64().unwrap() as i32,
    }
}

fn upload_bf16(ordinal: usize, shape: &[usize], bytes: &[u8], label: &str) -> GpuBuffer {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, shape, bytes)
        .unwrap_or_else(|e| panic!("upload {label}: {e}"))
}

fn upload_f32(ordinal: usize, shape: &[usize], bytes: &[u8], label: &str) -> GpuBuffer {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, shape, bytes)
        .unwrap_or_else(|e| panic!("upload {label}: {e}"))
}

fn upload_u8(ordinal: usize, shape: &[usize], bytes: &[u8], label: &str) -> GpuBuffer {
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, shape, bytes)
        .unwrap_or_else(|e| panic!("upload {label}: {e}"))
}

/// Helper: pull (packed_u8, scale_bf16, zero_bf16) byte streams for one
/// INT4-quantized tensor out of `int4_weights_per_layer[li].{attn|ffn}.<name>`.
fn decode_int4_sidecar(block: &Value, name: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let blk = &block[name];
    let packed = b64(blk["packed"].as_str()
        .unwrap_or_else(|| panic!("missing int4 {name}.packed")));
    let scale = b64(blk["scale"].as_str()
        .unwrap_or_else(|| panic!("missing int4 {name}.scale")));
    let zero = b64(blk["zero"].as_str()
        .unwrap_or_else(|| panic!("missing int4 {name}.zero")));
    (packed, scale, zero)
}

fn build_full_attn_layer(
    ordinal: usize,
    geom: &MultiLayerGeom,
    weights: &Value,
    int4_block: Option<&Value>,
    group_size: i32,
) -> AttnLayerBuffers {
    let hidden = geom.hidden as usize;
    let h = geom.num_attention_heads as usize;
    let hkv = geom.num_kv_heads as usize;
    let d = geom.head_dim as usize;

    // In INT4 mode the projection weight buffers carry packed nibbles
    // ([out, in/2] u8) instead of BF16 reconstructions. Sidecars come from
    // the parallel `int4_weights_per_layer[li].attn` block.
    let (q_proj_w, k_proj_w, v_proj_w, o_proj_w, int4) = if let Some(blk) = int4_block {
        let (qp, qs, qz) = decode_int4_sidecar(blk, "q_proj_w");
        let (kp, ks, kz) = decode_int4_sidecar(blk, "k_proj_w");
        let (vp, vs, vz) = decode_int4_sidecar(blk, "v_proj_w");
        let (op, os, oz) = decode_int4_sidecar(blk, "o_proj_w");
        let q_proj_w = upload_u8(ordinal, &[2 * h * d, hidden / 2], &qp, "q_proj packed");
        let k_proj_w = upload_u8(ordinal, &[hkv * d, hidden / 2], &kp, "k_proj packed");
        let v_proj_w = upload_u8(ordinal, &[hkv * d, hidden / 2], &vp, "v_proj packed");
        let o_proj_w = upload_u8(ordinal, &[hidden, h * d / 2], &op, "o_proj packed");
        let int4 = FullAttnInt4Sidecars {
            group_size,
            q_proj_scale: upload_bf16(ordinal, &[qs.len() / 2], &qs, "q scale"),
            q_proj_zero:  upload_bf16(ordinal, &[qz.len() / 2], &qz, "q zero"),
            k_proj_scale: upload_bf16(ordinal, &[ks.len() / 2], &ks, "k scale"),
            k_proj_zero:  upload_bf16(ordinal, &[kz.len() / 2], &kz, "k zero"),
            v_proj_scale: upload_bf16(ordinal, &[vs.len() / 2], &vs, "v scale"),
            v_proj_zero:  upload_bf16(ordinal, &[vz.len() / 2], &vz, "v zero"),
            o_proj_scale: upload_bf16(ordinal, &[os.len() / 2], &os, "o scale"),
            o_proj_zero:  upload_bf16(ordinal, &[oz.len() / 2], &oz, "o zero"),
        };
        (q_proj_w, k_proj_w, v_proj_w, o_proj_w, Some(int4))
    } else {
        (
            upload_bf16(ordinal, &[2 * h * d, hidden], &b64_field(weights, "q_proj_w"), "q_proj_w"),
            upload_bf16(ordinal, &[hkv * d, hidden], &b64_field(weights, "k_proj_w"), "k_proj_w"),
            upload_bf16(ordinal, &[hkv * d, hidden], &b64_field(weights, "v_proj_w"), "v_proj_w"),
            upload_bf16(ordinal, &[hidden, h * d], &b64_field(weights, "o_proj_w"), "o_proj_w"),
            None,
        )
    };

    AttnLayerBuffers::Full {
        input_norm_w: upload_bf16(ordinal, &[hidden], &b64_field(weights, "input_norm_w"), "input_norm_w"),
        q_proj_w,
        k_proj_w,
        v_proj_w,
        q_norm_w: upload_bf16(ordinal, &[d], &b64_field(weights, "q_norm_w"), "q_norm_w"),
        k_norm_w: upload_bf16(ordinal, &[d], &b64_field(weights, "k_norm_w"), "k_norm_w"),
        o_proj_w,
        int4,
    }
}

fn build_linear_attn_layer(
    ordinal: usize,
    geom: &MultiLayerGeom,
    weights: &Value,
    int4_block: Option<&Value>,
    group_size: i32,
) -> AttnLayerBuffers {
    let hidden = geom.hidden as usize;
    let k = geom.num_k_heads as usize;
    let v = geom.num_v_heads as usize;
    let kd = geom.head_k_dim as usize;
    let vd = geom.head_v_dim as usize;
    let kernel = geom.conv_kernel_dim as usize;
    let key_dim = k * kd;
    let val_dim = v * vd;
    let qkv_dim = 2 * key_dim + val_dim;
    let state_elems = v * kd * vd;

    let conv1d_bias = weights
        .get("conv1d_bias")
        .and_then(|v| v.as_str())
        .map(|s| upload_bf16(ordinal, &[qkv_dim], &b64(s), "conv1d_bias"));

    let (in_proj_qkv_w, in_proj_z_w, out_proj_w, int4) = if let Some(blk) = int4_block {
        let (qp, qs, qz) = decode_int4_sidecar(blk, "in_proj_qkv_w");
        let (zp, zs, zz) = decode_int4_sidecar(blk, "in_proj_z_w");
        let (op, os, oz) = decode_int4_sidecar(blk, "out_proj_w");
        let in_proj_qkv_w = upload_u8(ordinal, &[qkv_dim, hidden / 2], &qp, "in_proj_qkv packed");
        let in_proj_z_w = upload_u8(ordinal, &[val_dim, hidden / 2], &zp, "in_proj_z packed");
        let out_proj_w = upload_u8(ordinal, &[hidden, val_dim / 2], &op, "out_proj packed");
        let int4 = LinearAttnInt4Sidecars {
            group_size,
            in_proj_qkv_scale: upload_bf16(ordinal, &[qs.len() / 2], &qs, "in_proj_qkv scale"),
            in_proj_qkv_zero:  upload_bf16(ordinal, &[qz.len() / 2], &qz, "in_proj_qkv zero"),
            in_proj_z_scale:   upload_bf16(ordinal, &[zs.len() / 2], &zs, "in_proj_z scale"),
            in_proj_z_zero:    upload_bf16(ordinal, &[zz.len() / 2], &zz, "in_proj_z zero"),
            out_proj_scale:    upload_bf16(ordinal, &[os.len() / 2], &os, "out_proj scale"),
            out_proj_zero:     upload_bf16(ordinal, &[oz.len() / 2], &oz, "out_proj zero"),
        };
        (in_proj_qkv_w, in_proj_z_w, out_proj_w, Some(int4))
    } else {
        (
            upload_bf16(ordinal, &[qkv_dim, hidden], &b64_field(weights, "in_proj_qkv_w"), "in_proj_qkv_w"),
            upload_bf16(ordinal, &[val_dim, hidden], &b64_field(weights, "in_proj_z_w"), "in_proj_z_w"),
            upload_bf16(ordinal, &[hidden, val_dim], &b64_field(weights, "out_proj_w"), "out_proj_w"),
            None,
        )
    };

    AttnLayerBuffers::Linear {
        input_norm_w: upload_bf16(ordinal, &[hidden], &b64_field(weights, "input_norm_w"), "input_norm_w"),
        in_proj_qkv_w,
        in_proj_z_w,
        in_proj_a_w: upload_bf16(ordinal, &[v, hidden], &b64_field(weights, "in_proj_a_w"), "in_proj_a_w"),
        in_proj_b_w: upload_bf16(ordinal, &[v, hidden], &b64_field(weights, "in_proj_b_w"), "in_proj_b_w"),
        // The kernel's depthwise conv1d expects the channel-major layout the
        // bake produces: `[qkv_dim, kernel]`. The oracle stores it in the
        // squeezed shape; the BF16 byte stream is identical either way.
        conv1d_w: upload_bf16(ordinal, &[qkv_dim, kernel], &b64_field(weights, "conv1d_w"), "conv1d_w"),
        conv1d_bias,
        dt_bias: upload_bf16(ordinal, &[v], &b64_field(weights, "dt_bias"), "dt_bias"),
        a_log: upload_bf16(ordinal, &[v], &b64_field(weights, "a_log"), "a_log"),
        norm_w: upload_bf16(ordinal, &[vd], &b64_field(weights, "norm_w"), "norm_w"),
        out_proj_w,
        conv_state: upload_bf16(
            ordinal,
            &[qkv_dim, kernel - 1],
            &b64_field(weights, "conv_state_before"),
            "conv_state_before",
        ),
        // Recurrent state is F32 (production keeps it F32 across decode steps).
        recurrent_state: upload_f32(
            ordinal,
            &[state_elems],
            &b64_field(weights, "recurrent_state_before"),
            "recurrent_state_before",
        ),
        int4,
    }
}

fn build_ffn_layer(
    ordinal: usize,
    geom: &MultiLayerGeom,
    weights: &Value,
    int4_block: Option<&Value>,
    group_size: i32,
) -> FfnLayerBuffers {
    let hidden = geom.hidden as usize;
    let e = geom.num_experts as usize;
    let i_dim = geom.moe_intermediate as usize;
    let is_dim = geom.shared_intermediate as usize;

    let (gate_up_proj_w, down_proj_w, shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w, int4) =
        if let Some(blk) = int4_block {
            let (gp, gs, gz) = decode_int4_sidecar(blk, "gate_up_proj_w");
            let (dp, ds, dz) = decode_int4_sidecar(blk, "down_proj_w");
            let (sgp, sgs, sgz) = decode_int4_sidecar(blk, "shared_gate_proj_w");
            let (sup, sus, suz) = decode_int4_sidecar(blk, "shared_up_proj_w");
            let (sdp, sds, sdz) = decode_int4_sidecar(blk, "shared_down_proj_w");
            // Fused-expert tensors are 3D `[E, out, in]`; packed is
            // `[E, out, in/2]` u8.
            let gate_up_proj_w = upload_u8(ordinal, &[e, 2 * i_dim, hidden / 2], &gp, "gate_up packed");
            let down_proj_w = upload_u8(ordinal, &[e, hidden, i_dim / 2], &dp, "down_proj packed");
            let shared_gate_proj_w = upload_u8(ordinal, &[is_dim, hidden / 2], &sgp, "sgp packed");
            let shared_up_proj_w = upload_u8(ordinal, &[is_dim, hidden / 2], &sup, "sup packed");
            let shared_down_proj_w = upload_u8(ordinal, &[hidden, is_dim / 2], &sdp, "sdp packed");
            let int4 = FfnInt4Sidecars {
                group_size,
                gate_up_proj_scale: upload_bf16(ordinal, &[gs.len() / 2], &gs, "gate_up scale"),
                gate_up_proj_zero:  upload_bf16(ordinal, &[gz.len() / 2], &gz, "gate_up zero"),
                down_proj_scale:    upload_bf16(ordinal, &[ds.len() / 2], &ds, "down_proj scale"),
                down_proj_zero:     upload_bf16(ordinal, &[dz.len() / 2], &dz, "down_proj zero"),
                shared_gate_proj_scale: upload_bf16(ordinal, &[sgs.len() / 2], &sgs, "sgp scale"),
                shared_gate_proj_zero:  upload_bf16(ordinal, &[sgz.len() / 2], &sgz, "sgp zero"),
                shared_up_proj_scale:   upload_bf16(ordinal, &[sus.len() / 2], &sus, "sup scale"),
                shared_up_proj_zero:    upload_bf16(ordinal, &[suz.len() / 2], &suz, "sup zero"),
                shared_down_proj_scale: upload_bf16(ordinal, &[sds.len() / 2], &sds, "sdp scale"),
                shared_down_proj_zero:  upload_bf16(ordinal, &[sdz.len() / 2], &sdz, "sdp zero"),
            };
            (gate_up_proj_w, down_proj_w, shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w, Some(int4))
        } else {
            (
                upload_bf16(ordinal, &[e, 2 * i_dim, hidden], &b64_field(weights, "gate_up_proj_w"), "gate_up_proj_w"),
                upload_bf16(ordinal, &[e, hidden, i_dim], &b64_field(weights, "down_proj_w"), "down_proj_w"),
                upload_bf16(ordinal, &[is_dim, hidden], &b64_field(weights, "shared_gate_proj_w"), "shared_gate_proj_w"),
                upload_bf16(ordinal, &[is_dim, hidden], &b64_field(weights, "shared_up_proj_w"), "shared_up_proj_w"),
                upload_bf16(ordinal, &[hidden, is_dim], &b64_field(weights, "shared_down_proj_w"), "shared_down_proj_w"),
                None,
            )
        };

    FfnLayerBuffers {
        post_attn_norm_w: upload_bf16(ordinal, &[hidden], &b64_field(weights, "post_attn_norm_w"), "post_attn_norm_w"),
        gate_w: upload_bf16(ordinal, &[e, hidden], &b64_field(weights, "gate_w"), "gate_w"),
        gate_up_proj_w,
        down_proj_w,
        shared_gate_proj_w,
        shared_up_proj_w,
        shared_down_proj_w,
        shared_expert_gate_w: upload_bf16(
            ordinal,
            &[1, hidden],
            &b64_field(weights, "shared_expert_gate_w"),
            "shared_expert_gate_w",
        ),
        int4,
    }
}

/// Identical envelope to the per-block parity tests: cos_sim against the
/// oracle's BF16 buffer, plus a max |delta| tolerance. Per the plan, the
/// final logits parity gate is `cos_sim ≥ 0.999`. Per-layer hiddens get a
/// tighter `0.9999` floor since they're individual residuals (not a
/// reduction over `vocab` lanes).
fn assert_parity_bf16(label: &str, got: &[u8], want: &[u8], max_abs_tol: f32, cos_sim_floor: f64) {
    assert_eq!(got.len(), want.len(), "{label}: byte length mismatch");
    let g = bf16_bytes_to_f32(got);
    let w = bf16_bytes_to_f32(want);
    let n = g.len();
    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    let mut dot = 0f64;
    let mut g_sq = 0f64;
    let mut w_sq = 0f64;
    let mut exact = 0usize;
    for i in 0..n {
        let d = (g[i] - w[i]).abs();
        if d == 0.0 {
            exact += 1;
        }
        max_abs = max_abs.max(d);
        sum_abs += d;
        dot += g[i] as f64 * w[i] as f64;
        g_sq += (g[i] as f64).powi(2);
        w_sq += (w[i] as f64).powi(2);
    }
    let cos_sim = dot / (g_sq.sqrt() * w_sq.sqrt() + 1e-30);
    let mean_abs = sum_abs / n as f32;
    eprintln!(
        "[parity {label}] n={n} exact={exact} max_abs={max_abs:.5e} \
         mean_abs={mean_abs:.5e} cos_sim={cos_sim:.7}"
    );
    assert!(
        max_abs <= max_abs_tol,
        "{label}: max_abs {max_abs} exceeds tolerance {max_abs_tol}"
    );
    assert!(
        cos_sim >= cos_sim_floor,
        "{label}: cos_sim {cos_sim:.7} below floor {cos_sim_floor}"
    );
}

#[test]
fn multilayer_chained_decode_matches_oracle() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!(
            "skip: HIP backend not compiled — multi-layer parity test only \
             exercises the HIP kernels (CUDA/Metal aren't wired)."
        );
        return;
    }
    let Ok(json_path) = std::env::var("SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON") else {
        eprintln!(
            "skip: SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON not set. Generate with \
             `~/venvs/rocm/bin/python oracle/qwen36_moe_multilayer_oracle.py \
             --mode synthetic --num-layers 4 --out /tmp/qwen36_ml.json`."
        );
        return;
    };
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("read multi-layer oracle json {json_path}: {e}"));
    let json: Value = serde_json::from_str(&raw).expect("parse multi-layer oracle json");
    let schema = json["schema"].as_str().unwrap_or("");
    let int4_mode = match schema {
        "qwen36-moe-oracle-multilayer-v1" => false,
        "qwen36-moe-oracle-multilayer-int4-v1" => true,
        other => panic!("unsupported multi-layer schema: {other}"),
    };
    assert_eq!(
        json["dtype"].as_str().unwrap_or(""),
        "bf16",
        "multi-layer parity requires bf16 dtype"
    );

    let geom = parse_geom(&json);
    let position = json["position"].as_i64().unwrap_or(0) as i32;
    let int4_group_size = if int4_mode {
        json["config"]["int4_group_size"].as_i64().unwrap_or(128) as i32
    } else {
        0
    };

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    // Per-layer weight + state buffers. One LayerBuffers per transformer
    // layer; the order in `weights_per_layer` matches `layers[i].layer_idx`.
    let weights_per_layer = json["weights_per_layer"]
        .as_array()
        .expect("oracle JSON missing weights_per_layer (regenerate without --no-emit-weights)");
    assert_eq!(
        weights_per_layer.len(),
        geom.num_layers as usize,
        "weights_per_layer length mismatch"
    );

    // INT4 sidecar block — present iff the oracle was run with --int4.
    let int4_per_layer: Option<&Vec<Value>> = if int4_mode {
        Some(json["int4_weights_per_layer"]
            .as_array()
            .expect("INT4 oracle missing int4_weights_per_layer (regenerate with --int4 and without --no-emit-weights)"))
    } else {
        None
    };

    let mut layers: Vec<LayerBuffers> = Vec::with_capacity(geom.num_layers as usize);
    for (li, layer_json) in weights_per_layer.iter().enumerate() {
        let attn_w = &layer_json["attn"];
        let ffn_w = &layer_json["ffn"];
        let attn_int4 = int4_per_layer.map(|v| &v[li]["attn"]);
        let ffn_int4 = int4_per_layer.map(|v| &v[li]["ffn"]);
        let attn = if is_full_attn_layer(li as i32) {
            build_full_attn_layer(ordinal, &geom, attn_w, attn_int4, int4_group_size)
        } else {
            build_linear_attn_layer(ordinal, &geom, attn_w, attn_int4, int4_group_size)
        };
        let ffn = build_ffn_layer(ordinal, &geom, ffn_w, ffn_int4, int4_group_size);
        layers.push(LayerBuffers { attn, ffn });
    }

    let initial_hidden = b64_field(&json, "input_hidden");
    let final_norm_w = b64_field(&json, "final_norm_w");
    let lm_head_w = b64_field(&json, "lm_head_w");
    let oracle_logits = b64_field(&json, "logits");

    let outputs = run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, position)
        .expect("chained decode");

    // Per-layer parity. Tighter floor (0.9999) — no vocab-wide reduction
    // to absorb noise.
    // BF16 rounding noise compounds through residuals — every layer's matvec
    // adds ~1 ULP per output channel, and N-layer chains see roughly N×
    // the single-block max_abs and a few-times-N drop in cos_sim. The
    // PR 4c acceptance criterion (cos_sim ≥ 0.999 on the final logits)
    // is the real structural-correctness gate; per-layer envelopes are
    // outlier sanity checks scaled with depth so the 4th residual being
    // 4× noisier than the 1st doesn't false-positive.
    let max_abs_envelope = |li: usize| -> f32 {
        // Single-block stage-5 tests use 0.05; allow that much per layer
        // of compounding (capped at 0.5 so a real divergence still trips).
        (0.05 * (li as f32 + 1.0) * 1.5).min(0.5)
    };
    // Drop per-layer cos_sim floor by ~5e-5 per layer of depth — matches
    // the observed drift on the synthetic 4-layer fixture (0.9999937 at
    // layer 0 down to ~0.99988 at layer 3) with headroom. Still tight
    // enough that any real bug (e.g. wrong layer kind, swapped weight
    // pointer) trips it by orders of magnitude.
    let cos_sim_floor = |li: usize| -> f64 {
        (0.9999 - 5e-5 * li as f64).max(0.999)
    };
    let inters = json["intermediates_per_layer"].as_array().expect("intermediates_per_layer array");
    assert_eq!(inters.len(), geom.num_layers as usize, "intermediates length mismatch");
    for (li, item) in inters.iter().enumerate() {
        let want_attn = b64_field(item, "output_after_attn");
        let want_ffn = b64_field(item, "output_after_ffn");
        let envelope = max_abs_envelope(li);
        let floor = cos_sim_floor(li);
        assert_parity_bf16(
            &format!("layer {li} output_after_attn"),
            &outputs.per_layer_attn_out[li],
            &want_attn,
            envelope,
            floor,
        );
        assert_parity_bf16(
            &format!("layer {li} output_after_ffn"),
            &outputs.per_layer_ffn_out[li],
            &want_ffn,
            envelope,
            floor,
        );
    }

    // The kernel-side residual is covered by the per-layer last-FFN check
    // above (oracle's `final_hidden` is POST-RMSnorm so it isn't directly
    // comparable to `outputs.final_hidden_bytes`). Host-side final RMSnorm
    // + lm_head against that residual produces logits to compare; the
    // cos_sim floor is the PR 4c acceptance criterion (≥ 0.999 over the
    // vocab logits).
    let logits = host_final_norm_lm_head(
        &outputs.final_hidden_bytes,
        &final_norm_w,
        &lm_head_w,
        geom.hidden as usize,
        geom.vocab as usize,
        geom.rms_norm_eps,
    );
    assert_parity_bf16("logits", &logits, &oracle_logits, 0.5, 0.999);
}
