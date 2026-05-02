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
use kernel_ffi::qwen36_moe::{
    persistent_decode_launch, Qwen36MoeDecodeLayerDesc, Qwen36MoeInt4ScaleDesc,
    Qwen36MoePersistentGeom,
};
use runner::qwen36_moe_decode::{
    bf16_bytes_to_f32, ffn_workspace_floats, full_attn_workspace_floats, host_final_norm_lm_head,
    is_full_attn_layer, run_chained_decode, AttnLayerBuffers, FfnInt4Sidecars, FfnLayerBuffers,
    FullAttnInt4Sidecars, LayerBuffers, LinearAttnInt4Sidecars, MultiLayerGeom,
};
use runner::qwen36_moe_state::{restore_linear_attn_state, save_linear_attn_state};
use serde_json::Value;
use std::ffi::c_void;
use std::os::raw::c_int;

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
        // Parity test runs at position=0; KV cache stays disabled so the
        // kernel uses the back-compat kv_len=1 self-attention path.
        kv_cache: None,
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

// =============================================================================
// Phase 3e: persistent decode megakernel parity test.
//
// Drives `kernel_ffi::qwen36_moe::persistent_decode_launch` with the same
// fixtures the chained-decode test uses, then asserts the final hidden
// matches the chained path nearly bit-for-bit. The two paths run the
// IDENTICAL `__device__` phase functions (extracted in Phase 3a-3d) — only
// the launch orchestration differs (1 cooperative launch vs 80 step
// launches, with `reset_counters_16` between phases inside the
// megakernel). So the comparison floor is very tight (cos_sim ≥ 0.99999,
// max_abs ≤ 1e-3).
// =============================================================================

/// Build the parallel `Qwen36MoeDecodeLayerDesc` array from the live
/// `Vec<LayerBuffers>`. All weight pointers come straight from the
/// GpuBuffers; null pointers indicate "tensor stays BF16" (the persistent
/// kernel reads INT4 sidecars from the parallel `Qwen36MoeInt4ScaleDesc`
/// array — this struct only carries the `*_w` slot pointers).
fn build_layer_descs(layers: &mut [LayerBuffers]) -> Vec<Qwen36MoeDecodeLayerDesc> {
    use std::ptr;
    let mut descs = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter_mut().enumerate() {
        let mut d = Qwen36MoeDecodeLayerDesc::default();
        d.layer_idx = li as c_int;
        d.is_full_attention = if l.is_full_attn() { 1 } else { 0 };
        match &mut l.attn {
            AttnLayerBuffers::Full {
                input_norm_w,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                q_norm_w,
                k_norm_w,
                o_proj_w,
                kv_cache,
                ..
            } => {
                d.input_norm_w = input_norm_w.as_ptr() as *const c_void;
                d.q_proj_w = q_proj_w.as_ptr() as *const c_void;
                d.k_proj_w = k_proj_w.as_ptr() as *const c_void;
                d.v_proj_w = v_proj_w.as_ptr() as *const c_void;
                d.q_norm_w = q_norm_w.as_ptr() as *const c_void;
                d.k_norm_w = k_norm_w.as_ptr() as *const c_void;
                d.o_proj_w = o_proj_w.as_ptr() as *const c_void;
                if let Some(c) = kv_cache.as_mut() {
                    d.kv_cache_k = c.k.as_mut_ptr();
                    d.kv_cache_v = c.v.as_mut_ptr();
                    d.kv_max_t = c.kv_max_t;
                }
            }
            AttnLayerBuffers::Linear {
                input_norm_w,
                in_proj_qkv_w,
                in_proj_z_w,
                in_proj_a_w,
                in_proj_b_w,
                conv1d_w,
                dt_bias,
                a_log,
                norm_w,
                out_proj_w,
                conv_state,
                recurrent_state,
                ..
            } => {
                d.input_norm_w = input_norm_w.as_ptr() as *const c_void;
                d.linear_in_proj_qkv_w = in_proj_qkv_w.as_ptr() as *const c_void;
                d.linear_in_proj_z_w = in_proj_z_w.as_ptr() as *const c_void;
                d.linear_in_proj_a_w = in_proj_a_w.as_ptr() as *const c_void;
                d.linear_in_proj_b_w = in_proj_b_w.as_ptr() as *const c_void;
                d.linear_conv1d_w = conv1d_w.as_ptr() as *const c_void;
                d.linear_dt_bias = dt_bias.as_ptr() as *const c_void;
                d.linear_a_log_exp = a_log.as_ptr() as *const c_void;
                d.linear_norm_w = norm_w.as_ptr() as *const c_void;
                d.linear_out_proj_w = out_proj_w.as_ptr() as *const c_void;
                d.linear_conv_state = conv_state.as_mut_ptr();
                d.linear_recurrent_state = recurrent_state.as_mut_ptr();
            }
        }
        // FFN block.
        d.post_attn_norm_w = l.ffn.post_attn_norm_w.as_ptr() as *const c_void;
        d.router_w = l.ffn.gate_w.as_ptr() as *const c_void;
        d.experts_gate_up_w = l.ffn.gate_up_proj_w.as_ptr() as *const c_void;
        d.experts_down_w = l.ffn.down_proj_w.as_ptr() as *const c_void;
        d.shared_expert_gate_proj_w = l.ffn.shared_gate_proj_w.as_ptr() as *const c_void;
        d.shared_expert_up_proj_w = l.ffn.shared_up_proj_w.as_ptr() as *const c_void;
        d.shared_expert_down_proj_w = l.ffn.shared_down_proj_w.as_ptr() as *const c_void;
        d.shared_expert_gate_w = l.ffn.shared_expert_gate_w.as_ptr() as *const c_void;
        // Geometry fields are duplicated across layers to match the Rust
        // descriptor struct (the kernel uses the launch-level geometry
        // constants instead — these are reserved for sanity checks).
        let _ = ptr::null::<c_void>();
        descs.push(d);
    }
    descs
}

/// Build the parallel `Qwen36MoeInt4ScaleDesc` array. Returns `None` when
/// none of the layers carry INT4 sidecars (the BF16 fixture path).
fn build_int4_descs(layers: &[LayerBuffers]) -> Option<Vec<Qwen36MoeInt4ScaleDesc>> {
    let any_int4 = layers.iter().any(|l| {
        let attn_q = match &l.attn {
            AttnLayerBuffers::Full { int4, .. } => int4.is_some(),
            AttnLayerBuffers::Linear { int4, .. } => int4.is_some(),
        };
        attn_q || l.ffn.int4.is_some()
    });
    if !any_int4 {
        return None;
    }
    let mut int4 = Vec::with_capacity(layers.len());
    for l in layers.iter() {
        let mut d = Qwen36MoeInt4ScaleDesc::default();
        match &l.attn {
            AttnLayerBuffers::Full { int4: Some(s), .. } => {
                d.q_proj_scale = s.q_proj_scale.as_ptr() as *const c_void;
                d.q_proj_zero = s.q_proj_zero.as_ptr() as *const c_void;
                d.k_proj_scale = s.k_proj_scale.as_ptr() as *const c_void;
                d.k_proj_zero = s.k_proj_zero.as_ptr() as *const c_void;
                d.v_proj_scale = s.v_proj_scale.as_ptr() as *const c_void;
                d.v_proj_zero = s.v_proj_zero.as_ptr() as *const c_void;
                d.o_proj_scale = s.o_proj_scale.as_ptr() as *const c_void;
                d.o_proj_zero = s.o_proj_zero.as_ptr() as *const c_void;
                d.group_size = s.group_size;
            }
            AttnLayerBuffers::Linear { int4: Some(s), .. } => {
                d.linear_in_proj_qkv_scale = s.in_proj_qkv_scale.as_ptr() as *const c_void;
                d.linear_in_proj_qkv_zero = s.in_proj_qkv_zero.as_ptr() as *const c_void;
                d.linear_in_proj_z_scale = s.in_proj_z_scale.as_ptr() as *const c_void;
                d.linear_in_proj_z_zero = s.in_proj_z_zero.as_ptr() as *const c_void;
                d.linear_out_proj_scale = s.out_proj_scale.as_ptr() as *const c_void;
                d.linear_out_proj_zero = s.out_proj_zero.as_ptr() as *const c_void;
                d.group_size = s.group_size;
            }
            _ => {}
        }
        if let Some(s) = &l.ffn.int4 {
            d.experts_gate_up_scale = s.gate_up_proj_scale.as_ptr() as *const c_void;
            d.experts_gate_up_zero = s.gate_up_proj_zero.as_ptr() as *const c_void;
            d.experts_down_scale = s.down_proj_scale.as_ptr() as *const c_void;
            d.experts_down_zero = s.down_proj_zero.as_ptr() as *const c_void;
            d.shared_expert_gate_proj_scale = s.shared_gate_proj_scale.as_ptr() as *const c_void;
            d.shared_expert_gate_proj_zero = s.shared_gate_proj_zero.as_ptr() as *const c_void;
            d.shared_expert_up_proj_scale = s.shared_up_proj_scale.as_ptr() as *const c_void;
            d.shared_expert_up_proj_zero = s.shared_up_proj_zero.as_ptr() as *const c_void;
            d.shared_expert_down_proj_scale = s.shared_down_proj_scale.as_ptr() as *const c_void;
            d.shared_expert_down_proj_zero = s.shared_down_proj_zero.as_ptr() as *const c_void;
            d.group_size = s.group_size;
        }
        int4.push(d);
    }
    Some(int4)
}

/// Upload the descriptor Vec to a device buffer as opaque U8 bytes. Same
/// pattern as the existing stub-launch test (`hip_stub_launch_walks_descriptor_array`).
fn upload_descs<T: Sized>(ordinal: usize, descs: &[T], label: &str) -> GpuBuffer {
    let per = std::mem::size_of::<T>();
    let mut bytes = Vec::with_capacity(per * descs.len());
    for d in descs {
        let p = d as *const T as *const u8;
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(p, per) });
    }
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[bytes.len()], &bytes)
        .unwrap_or_else(|e| panic!("upload {label}: {e}"))
}

#[test]
fn multilayer_persistent_decode_matches_chained() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
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
    let raw = std::fs::read_to_string(&json_path).expect("read multi-layer oracle json");
    let json: Value = serde_json::from_str(&raw).expect("parse multi-layer oracle json");
    let schema = json["schema"].as_str().unwrap_or("");
    let int4_mode = match schema {
        "qwen36-moe-oracle-multilayer-v1" => false,
        "qwen36-moe-oracle-multilayer-int4-v1" => true,
        other => panic!("unsupported multi-layer schema: {other}"),
    };

    let geom = parse_geom(&json);
    let position = json["position"].as_i64().unwrap_or(0) as i32;
    let int4_group_size = if int4_mode {
        json["config"]["int4_group_size"].as_i64().unwrap_or(128) as i32
    } else {
        0
    };

    set_backend(Backend::Hip);
    let ordinal = 0usize;

    let weights_per_layer = json["weights_per_layer"]
        .as_array()
        .expect("oracle JSON missing weights_per_layer");
    let int4_per_layer: Option<&Vec<Value>> = if int4_mode {
        Some(
            json["int4_weights_per_layer"]
                .as_array()
                .expect("INT4 oracle missing int4_weights_per_layer"),
        )
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

    // Snapshot the linear-attn state so we can reset between the chained
    // and persistent runs (linear-attn mutates conv_state +
    // recurrent_state per token).
    let snapshot =
        save_linear_attn_state(ordinal, &layers).expect("save_linear_attn_state");

    // ---- Chained-path reference ----
    let chained = run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, position)
        .expect("chained decode");

    // Restore linear state to the pre-chained values so the persistent
    // run sees the same starting point.
    restore_linear_attn_state(ordinal, &mut layers, &snapshot)
        .expect("restore_linear_attn_state");

    // ---- Persistent megakernel run ----
    let descs = build_layer_descs(&mut layers);
    let int4_descs = build_int4_descs(&layers);
    let descs_dev = upload_descs(ordinal, &descs, "layer descriptors");
    let int4_dev = int4_descs
        .as_ref()
        .map(|v| upload_descs(ordinal, v, "int4 scale descriptors"));

    let hidden = geom.hidden as usize;
    let mut hidden_ping = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[hidden],
        &initial_hidden,
    )
    .expect("alloc hidden_ping");
    let mut hidden_pong =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("alloc hidden_pong");

    let ws_floats = full_attn_workspace_floats(&geom).max(ffn_workspace_floats(&geom));
    let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ws_floats])
        .expect("alloc workspace");
    let mut ffn_topk_idx_scratch =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
            .expect("alloc ffn_topk_idx_scratch");
    let mut sync_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc sync_buf");

    let pgeom = Qwen36MoePersistentGeom {
        hidden: geom.hidden,
        num_heads: geom.num_attention_heads,
        num_kv_heads: geom.num_kv_heads,
        head_dim: geom.head_dim,
        rotary_dim: geom.rotary_dim,
        num_k_heads: geom.num_k_heads,
        num_v_heads: geom.num_v_heads,
        head_k_dim: geom.head_k_dim,
        head_v_dim: geom.head_v_dim,
        conv_kernel_dim: geom.conv_kernel_dim,
        num_experts: geom.num_experts,
        moe_intermediate: geom.moe_intermediate,
        shared_intermediate: geom.shared_intermediate,
        top_k: geom.top_k,
        rope_theta: geom.rope_theta,
        rms_norm_eps: geom.rms_norm_eps,
    };

    persistent_decode_launch(
        ordinal,
        ScalarType::BF16,
        pgeom,
        position,
        &descs_dev,
        int4_dev.as_ref(),
        geom.num_layers as usize,
        &mut hidden_ping,
        &mut hidden_pong,
        &mut workspace,
        &mut ffn_topk_idx_scratch,
        &mut sync_buf,
    )
    .expect("persistent_decode_launch");

    // After even num_layers (the synthetic fixture's 4 layers, the prod
    // 35B-A3B's 40), the final hidden lands back in `hidden_ping`.
    let persistent_final = hidden_ping
        .to_host_bytes()
        .expect("download persistent final hidden");

    // The persistent kernel runs the IDENTICAL `__device__` phase
    // functions (full_attn_phase / linear_attn_phase / ffn_phase) the
    // chained step kernels run — only the launch orchestration differs
    // (one cooperative launch + grid_barrier between phases vs. 80
    // separate step launches). So the comparison should be bit-exact.
    // Local 7900 XTX bring-up: 256/256 elements match, max_abs=0,
    // cos_sim=1.0 on both BF16 and INT4 fixtures.
    assert_parity_bf16(
        "persistent vs chained final_hidden",
        &persistent_final,
        &chained.final_hidden_bytes,
        1e-3,
        0.99999,
    );
}
