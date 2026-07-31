//! PR 4c step 2 multi-layer parity test for Qwen3.6-MoE.
//!
//! Loads the multi-layer Python oracle's JSON payload (produced by
//! `oracle/qwen36_moe_multilayer_oracle.py`), uploads each layer's BF16
//! weights + initial linear-attn state to the GPU, runs the chained decode
//! via [`supersonic_runtime::qwen36_moe::decode::run_chained_decode`], applies
//! the host-side final RMSnorm + lm_head, and compares against the oracle's
//! `intermediates_per_layer` + `final_hidden` + `logits`. Local kernel
//! stages retain their per-block envelopes. Chained handoffs use frozen
//! per-layer cumulative budgets, and final logits retain cos_sim ≥ 0.999.
//!
//! The qualification fixture is tracked at
//! `oracle/fixtures/qwen36_moe_multilayer_int4_v1.json` and is mandatory.
//! `SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON` may override it for diagnostic
//! runs, but the normal qualification command needs no environment variable:
//!
//! ```bash
//! cargo test --release -p runner --test qwen36_moe_multilayer_parity \
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
use gpu_hal::{copy_d2h, is_backend_compiled, set_backend, Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::qwen36_moe::{
    ffn_step_launch, Qwen36MoeFfnStepInt4, Qwen36MoeFfnStepParams, Qwen36MoeFfnStepWeights,
};
use runner::qwen36_moe_logits::{bf16_bytes_to_f32, host_final_norm_lm_head};
use runner::qwen36_moe_state::{restore_linear_attn_state, save_linear_attn_state};
use serde_json::Value;
use std::ffi::c_void;
use std::path::PathBuf;
use supersonic_runtime::qwen36_moe::chain::{run_chain_step, Qwen36ChainStep};
use supersonic_runtime::qwen36_moe::decode::{
    ffn_output_elems, ffn_workspace_floats, run_chained_decode, Qwen36ExecutionOptions,
};
use supersonic_runtime::qwen36_moe::layer_loader::Qwen36WeightMode;
use supersonic_runtime::qwen36_moe::layers::LoadedQwen36Layers;
use supersonic_runtime::qwen36_moe::persistent_decode::{LmHeadFold, CACHE_POS_INHERIT};
use supersonic_runtime::qwen36_moe::types::{
    is_full_attn_layer, AttnLayerBuffers, FfnInt4Sidecars, FfnLayerBuffers, FullAttnInt4Sidecars,
    LayerBuffers, LinearAttnInt4Sidecars, MultiLayerGeom, PositionPair, ResidentWeight,
};

const TRACKED_MULTILAYER_ORACLE: &str = "../../oracle/fixtures/qwen36_moe_multilayer_int4_v1.json";

#[derive(Clone, Copy, Debug)]
struct NumericalBudget {
    max_abs: f32,
    cosine_floor: f64,
}

#[derive(Clone, Copy, Debug)]
enum HandoffBoundary {
    Attention,
    Ffn,
}

// Candidate-independent cumulative budgets frozen from the accepted Round 3
// qualification run. Maxima were rounded outward to nearby binary fractions
// and cosine floors downward. They are constants rather than values derived
// from the candidate under test, so additional accumulation still fails at
// the first material handoff where it exceeds the observed envelope.
//
// Observed attention max_abs:
//   [0.0234375, 0.15625, 0.2578125, 0.40625]
// Observed FFN max_abs:
//   [0.109375, 0.23828125, 0.4375, 0.4765625]
const ATTENTION_HANDOFF_BUDGETS: [NumericalBudget; 4] = [
    NumericalBudget {
        max_abs: 0.03125,
        cosine_floor: 0.9999,
    },
    NumericalBudget {
        max_abs: 0.1875,
        cosine_floor: 0.99975,
    },
    NumericalBudget {
        max_abs: 0.3125,
        cosine_floor: 0.9996,
    },
    NumericalBudget {
        max_abs: 0.4375,
        cosine_floor: 0.99945,
    },
];
const FFN_HANDOFF_BUDGETS: [NumericalBudget; 4] = [
    NumericalBudget {
        max_abs: 0.125,
        cosine_floor: 0.9998,
    },
    NumericalBudget {
        max_abs: 0.25,
        cosine_floor: 0.9997,
    },
    NumericalBudget {
        max_abs: 0.5,
        cosine_floor: 0.9995,
    },
    NumericalBudget {
        max_abs: 0.5,
        cosine_floor: 0.9994,
    },
];
const EXACT_INPUT_LOGITS_BUDGET: NumericalBudget = NumericalBudget {
    max_abs: 0.0625,
    cosine_floor: 0.999,
};
const CHAINED_LOGITS_BUDGET: NumericalBudget = NumericalBudget {
    max_abs: 0.25,
    cosine_floor: 0.999,
};

fn tracked_multilayer_oracle_path() -> PathBuf {
    std::env::var_os("SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TRACKED_MULTILAYER_ORACLE)
        })
}

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
    let packed = b64(blk["packed"]
        .as_str()
        .unwrap_or_else(|| panic!("missing int4 {name}.packed")));
    let scale = b64(blk["scale"]
        .as_str()
        .unwrap_or_else(|| panic!("missing int4 {name}.scale")));
    let zero = b64(blk["zero"]
        .as_str()
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
            q_proj_type: 4,
            q_proj_scale: upload_bf16(ordinal, &[qs.len() / 2], &qs, "q scale"),
            q_proj_zero: upload_bf16(ordinal, &[qz.len() / 2], &qz, "q zero"),
            k_proj_type: 4,
            k_proj_scale: upload_bf16(ordinal, &[ks.len() / 2], &ks, "k scale"),
            k_proj_zero: upload_bf16(ordinal, &[kz.len() / 2], &kz, "k zero"),
            v_proj_type: 4,
            v_proj_scale: upload_bf16(ordinal, &[vs.len() / 2], &vs, "v scale"),
            v_proj_zero: upload_bf16(ordinal, &[vz.len() / 2], &vz, "v zero"),
            o_proj_type: 4,
            o_proj_scale: upload_bf16(ordinal, &[os.len() / 2], &os, "o scale"),
            o_proj_zero: upload_bf16(ordinal, &[oz.len() / 2], &oz, "o zero"),
        };
        (q_proj_w, k_proj_w, v_proj_w, o_proj_w, Some(int4))
    } else {
        (
            upload_bf16(
                ordinal,
                &[2 * h * d, hidden],
                &b64_field(weights, "q_proj_w"),
                "q_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[hkv * d, hidden],
                &b64_field(weights, "k_proj_w"),
                "k_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[hkv * d, hidden],
                &b64_field(weights, "v_proj_w"),
                "v_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[hidden, h * d],
                &b64_field(weights, "o_proj_w"),
                "o_proj_w",
            ),
            None,
        )
    };

    AttnLayerBuffers::Full {
        input_norm_w: upload_bf16(
            ordinal,
            &[hidden],
            &b64_field(weights, "input_norm_w"),
            "input_norm_w",
        ),
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
            in_proj_qkv_type: 4,
            in_proj_qkv_scale: upload_bf16(ordinal, &[qs.len() / 2], &qs, "in_proj_qkv scale"),
            in_proj_qkv_zero: upload_bf16(ordinal, &[qz.len() / 2], &qz, "in_proj_qkv zero"),
            in_proj_z_type: 4,
            in_proj_z_scale: upload_bf16(ordinal, &[zs.len() / 2], &zs, "in_proj_z scale"),
            in_proj_z_zero: upload_bf16(ordinal, &[zz.len() / 2], &zz, "in_proj_z zero"),
            out_proj_type: 4,
            out_proj_scale: upload_bf16(ordinal, &[os.len() / 2], &os, "out_proj scale"),
            out_proj_zero: upload_bf16(ordinal, &[oz.len() / 2], &oz, "out_proj zero"),
        };
        (in_proj_qkv_w, in_proj_z_w, out_proj_w, Some(int4))
    } else {
        (
            upload_bf16(
                ordinal,
                &[qkv_dim, hidden],
                &b64_field(weights, "in_proj_qkv_w"),
                "in_proj_qkv_w",
            ),
            upload_bf16(
                ordinal,
                &[val_dim, hidden],
                &b64_field(weights, "in_proj_z_w"),
                "in_proj_z_w",
            ),
            upload_bf16(
                ordinal,
                &[hidden, val_dim],
                &b64_field(weights, "out_proj_w"),
                "out_proj_w",
            ),
            None,
        )
    };

    AttnLayerBuffers::Linear {
        input_norm_w: upload_bf16(
            ordinal,
            &[hidden],
            &b64_field(weights, "input_norm_w"),
            "input_norm_w",
        ),
        in_proj_qkv_w,
        in_proj_z_w,
        in_proj_a_w: upload_bf16(
            ordinal,
            &[v, hidden],
            &b64_field(weights, "in_proj_a_w"),
            "in_proj_a_w",
        ),
        in_proj_b_w: upload_bf16(
            ordinal,
            &[v, hidden],
            &b64_field(weights, "in_proj_b_w"),
            "in_proj_b_w",
        ),
        // The kernel's depthwise conv1d expects the channel-major layout the
        // bake produces: `[qkv_dim, kernel]`. The oracle stores it in the
        // squeezed shape; the BF16 byte stream is identical either way.
        conv1d_w: upload_bf16(
            ordinal,
            &[qkv_dim, kernel],
            &b64_field(weights, "conv1d_w"),
            "conv1d_w",
        ),
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

    let (
        gate_up_proj_w,
        down_proj_w,
        shared_gate_proj_w,
        shared_up_proj_w,
        shared_down_proj_w,
        int4,
    ) = if let Some(blk) = int4_block {
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
            gate_up_proj_type: 4,
            gate_up_proj_scale: upload_bf16(ordinal, &[gs.len() / 2], &gs, "gate_up scale"),
            gate_up_proj_zero: upload_bf16(ordinal, &[gz.len() / 2], &gz, "gate_up zero"),
            down_proj_type: 4,
            down_proj_scale: upload_bf16(ordinal, &[ds.len() / 2], &ds, "down_proj scale"),
            down_proj_zero: upload_bf16(ordinal, &[dz.len() / 2], &dz, "down_proj zero"),
            shared_gate_proj_type: 4,
            shared_gate_proj_scale: upload_bf16(ordinal, &[sgs.len() / 2], &sgs, "sgp scale"),
            shared_gate_proj_zero: upload_bf16(ordinal, &[sgz.len() / 2], &sgz, "sgp zero"),
            shared_up_proj_type: 4,
            shared_up_proj_scale: upload_bf16(ordinal, &[sus.len() / 2], &sus, "sup scale"),
            shared_up_proj_zero: upload_bf16(ordinal, &[suz.len() / 2], &suz, "sup zero"),
            shared_down_proj_type: 4,
            shared_down_proj_scale: upload_bf16(ordinal, &[sds.len() / 2], &sds, "sdp scale"),
            shared_down_proj_zero: upload_bf16(ordinal, &[sdz.len() / 2], &sdz, "sdp zero"),
        };
        (
            gate_up_proj_w,
            down_proj_w,
            shared_gate_proj_w,
            shared_up_proj_w,
            shared_down_proj_w,
            Some(int4),
        )
    } else {
        (
            upload_bf16(
                ordinal,
                &[e, 2 * i_dim, hidden],
                &b64_field(weights, "gate_up_proj_w"),
                "gate_up_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[e, hidden, i_dim],
                &b64_field(weights, "down_proj_w"),
                "down_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[is_dim, hidden],
                &b64_field(weights, "shared_gate_proj_w"),
                "shared_gate_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[is_dim, hidden],
                &b64_field(weights, "shared_up_proj_w"),
                "shared_up_proj_w",
            ),
            upload_bf16(
                ordinal,
                &[hidden, is_dim],
                &b64_field(weights, "shared_down_proj_w"),
                "shared_down_proj_w",
            ),
            None,
        )
    };

    FfnLayerBuffers {
        post_attn_norm_w: upload_bf16(
            ordinal,
            &[hidden],
            &b64_field(weights, "post_attn_norm_w"),
            "post_attn_norm_w",
        ),
        gate_w: upload_bf16(
            ordinal,
            &[e, hidden],
            &b64_field(weights, "gate_w"),
            "gate_w",
        ),
        gate_up_proj_w: ResidentWeight::Dense(gate_up_proj_w),
        down_proj_w: ResidentWeight::Dense(down_proj_w),
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

fn build_layers(
    ordinal: usize,
    geom: &MultiLayerGeom,
    weights_per_layer: &[Value],
    int4_per_layer: Option<&Vec<Value>>,
    int4_group_size: i32,
) -> Vec<LayerBuffers> {
    weights_per_layer
        .iter()
        .enumerate()
        .map(|(li, layer_json)| {
            let attn_w = &layer_json["attn"];
            let ffn_w = &layer_json["ffn"];
            let attn_int4 = int4_per_layer.map(|layers| &layers[li]["attn"]);
            let ffn_int4 = int4_per_layer.map(|layers| &layers[li]["ffn"]);
            let attn = if is_full_attn_layer(li as i32) {
                build_full_attn_layer(ordinal, geom, attn_w, attn_int4, int4_group_size)
            } else {
                build_linear_attn_layer(ordinal, geom, attn_w, attn_int4, int4_group_size)
            };
            let ffn = build_ffn_layer(ordinal, geom, ffn_w, ffn_int4, int4_group_size);
            LayerBuffers { attn, ffn }
        })
        .collect()
}

fn run_isolated_attention(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layer: &mut LayerBuffers,
    input_hidden: &[u8],
    position: i32,
) -> Vec<u8> {
    let mut single_geom = *geom;
    single_geom.num_layers = 1;
    run_chained_decode(
        ordinal,
        &single_geom,
        std::slice::from_mut(layer),
        input_hidden,
        position,
    )
    .expect("isolated attention chain")
    .per_layer_attn_out
    .into_iter()
    .next()
    .expect("isolated attention output")
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

fn handoff_budget(layer_idx: usize, boundary: HandoffBoundary) -> NumericalBudget {
    let budgets = match boundary {
        HandoffBoundary::Attention => &ATTENTION_HANDOFF_BUDGETS,
        HandoffBoundary::Ffn => &FFN_HANDOFF_BUDGETS,
    };
    *budgets
        .get(layer_idx)
        .unwrap_or_else(|| panic!("missing {boundary:?} budget for layer {layer_idx}"))
}

fn assert_chained_handoff_parity(
    label: &str,
    layer_idx: usize,
    boundary: HandoffBoundary,
    got: &[u8],
    want: &[u8],
) {
    let budget = handoff_budget(layer_idx, boundary);
    assert_parity_bf16(label, got, want, budget.max_abs, budget.cosine_floor);
}

fn max_abs_delta_bf16(lhs: &[u8], rhs: &[u8]) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "BF16 delta byte length mismatch");
    bf16_bytes_to_f32(lhs)
        .into_iter()
        .zip(bf16_bytes_to_f32(rhs))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn max_bf16_ulp(bytes: &[u8]) -> f32 {
    bf16_bytes_to_f32(bytes)
        .into_iter()
        .map(|value| {
            let magnitude = value.abs();
            if magnitude == 0.0 {
                2.0f32.powi(-133)
            } else {
                2.0f32.powi(magnitude.log2().floor() as i32 - 7)
            }
        })
        .fold(0.0, f32::max)
}

fn corrupt_one_lane_from_adjacent(
    handoff: &mut [u8],
    reference: &[u8],
    rejection_threshold: f32,
) -> (usize, usize, f32, f32) {
    assert_eq!(handoff.len(), reference.len(), "BF16 byte length mismatch");
    let actual = bf16_bytes_to_f32(handoff);
    let oracle = bf16_bytes_to_f32(reference);
    let candidate = (0..actual.len())
        .filter_map(|target| {
            let source = (target + 1) % actual.len();
            let lane_displacement = (actual[source] - actual[target]).abs();
            let resulting_error = (actual[source] - oracle[target]).abs();
            (lane_displacement >= 0.125
                && lane_displacement <= 0.5
                && resulting_error > rejection_threshold)
                .then_some((target, source, lane_displacement, resulting_error))
        })
        .min_by(|left, right| left.3.total_cmp(&right.3))
        .expect("runtime handoff has no bounded adjacent-lane corruption candidate");
    let source_bits = bf16::from_f32(actual[candidate.1]).to_bits().to_le_bytes();
    handoff[candidate.0 * 2..candidate.0 * 2 + 2].copy_from_slice(&source_bits);
    candidate
}

#[test]
fn tracked_multilayer_oracle_is_the_mandatory_default() {
    let path = tracked_multilayer_oracle_path();
    assert!(
        path.is_file(),
        "tracked multi-layer oracle fixture is missing: {}",
        path.display()
    );
    let raw = std::fs::read_to_string(&path).expect("read tracked multi-layer oracle");
    let json: Value = serde_json::from_str(&raw).expect("parse tracked multi-layer oracle");
    assert_eq!(
        json["schema"], "qwen36-moe-oracle-multilayer-int4-v1",
        "qualification default must be the independent INT4 oracle"
    );
    assert_eq!(json["mode"], "synthetic");
    assert_eq!(json["state"], "fresh");
    assert_eq!(json["num_layers"], 4);
}

#[test]
fn chained_gate_rejects_single_lane_packing_scale_perturbation() {
    let oracle = std::iter::repeat_n(0x3f80u16, 256)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut corrupted_handoff = oracle.clone();
    corrupted_handoff[73 * 2..73 * 2 + 2].copy_from_slice(&0x3fa0u16.to_le_bytes());

    let rejected = std::panic::catch_unwind(|| {
        assert_chained_handoff_parity(
            "single-lane packing-scale perturbation",
            0,
            HandoffBoundary::Ffn,
            &corrupted_handoff,
            &oracle,
        );
    });
    assert!(
        rejected.is_err(),
        "single-lane packing-scale perturbation passed the chained gate"
    );
}

fn i32_bytes_to_vec(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte i32")))
        .collect()
}

fn run_isolated_ffn_stage(
    ordinal: usize,
    geom: &MultiLayerGeom,
    layer_idx: usize,
    ffn: &FfnLayerBuffers,
    input_hidden: &[u8],
    stage: i32,
) -> (Vec<u8>, Vec<i32>) {
    let hidden = geom.hidden as usize;
    let input = upload_bf16(ordinal, &[hidden], input_hidden, "isolated FFN input");
    let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[ffn_output_elems(geom)])
        .expect("alloc isolated FFN output");
    let mut output_idx = GpuBuffer::zeros(ordinal, ScalarType::U32, &[geom.top_k as usize])
        .expect("alloc isolated FFN route output");
    let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ffn_workspace_floats(geom)])
        .expect("alloc isolated FFN workspace");
    let mut sync_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U8, &[96]).expect("alloc isolated FFN sync");
    let weights = Qwen36MoeFfnStepWeights {
        input_hidden: input.as_ptr(),
        post_attn_norm_w: ffn.post_attn_norm_w.as_ptr(),
        gate_w: ffn.gate_w.as_ptr(),
        gate_up_proj_w: ffn.gate_up_proj_w.as_ptr(),
        down_proj_w: ffn.down_proj_w.as_ptr(),
        shared_gate_proj_w: ffn.shared_gate_proj_w.as_ptr(),
        shared_up_proj_w: ffn.shared_up_proj_w.as_ptr(),
        shared_down_proj_w: ffn.shared_down_proj_w.as_ptr(),
        shared_expert_gate_w: ffn.shared_expert_gate_w.as_ptr(),
    };
    let int4 = ffn
        .int4
        .as_ref()
        .map_or_else(Qwen36MoeFfnStepInt4::disabled, |sidecars| {
            Qwen36MoeFfnStepInt4 {
                group_size: sidecars.group_size,
                gate_up_proj_type: sidecars.gate_up_proj_type,
                gate_up_proj_scale: sidecars.gate_up_proj_scale.as_ptr(),
                gate_up_proj_zero: sidecars.gate_up_proj_zero.as_ptr(),
                down_proj_type: sidecars.down_proj_type,
                down_proj_scale: sidecars.down_proj_scale.as_ptr(),
                down_proj_zero: sidecars.down_proj_zero.as_ptr(),
                shared_gate_proj_type: sidecars.shared_gate_proj_type,
                shared_gate_proj_scale: sidecars.shared_gate_proj_scale.as_ptr(),
                shared_gate_proj_zero: sidecars.shared_gate_proj_zero.as_ptr(),
                shared_up_proj_type: sidecars.shared_up_proj_type,
                shared_up_proj_scale: sidecars.shared_up_proj_scale.as_ptr(),
                shared_up_proj_zero: sidecars.shared_up_proj_zero.as_ptr(),
                shared_down_proj_type: sidecars.shared_down_proj_type,
                shared_down_proj_scale: sidecars.shared_down_proj_scale.as_ptr(),
                shared_down_proj_zero: sidecars.shared_down_proj_zero.as_ptr(),
            }
        });
    ffn_step_launch(
        ordinal,
        ScalarType::BF16,
        Qwen36MoeFfnStepParams {
            stage,
            layer_idx: layer_idx as i32,
            hidden: geom.hidden,
            num_experts: geom.num_experts,
            moe_intermediate: geom.moe_intermediate,
            shared_intermediate: geom.shared_intermediate,
            top_k: geom.top_k,
            rms_norm_eps: geom.rms_norm_eps,
        },
        &weights,
        &int4,
        &mut output,
        &mut output_idx,
        &mut workspace,
        &mut sync_buf,
    )
    .unwrap_or_else(|error| panic!("isolated FFN layer {layer_idx} stage {stage}: {error}"));

    let output_bytes = output
        .to_host_bytes()
        .expect("download isolated FFN output");
    let idx_bytes = output_idx
        .to_host_bytes()
        .expect("download isolated FFN routes");
    (output_bytes, i32_bytes_to_vec(&idx_bytes))
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
    let json_path = tracked_multilayer_oracle_path();
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("read multi-layer oracle json {}: {e}", json_path.display()));
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

    let mut layers = build_layers(
        ordinal,
        &geom,
        weights_per_layer,
        int4_per_layer,
        int4_group_size,
    );
    let mut exact_input_attn_layers = build_layers(
        ordinal,
        &geom,
        weights_per_layer,
        int4_per_layer,
        int4_group_size,
    );
    let mut chained_input_attn_layers = build_layers(
        ordinal,
        &geom,
        weights_per_layer,
        int4_per_layer,
        int4_group_size,
    );

    let initial_hidden = b64_field(&json, "input_hidden");
    let final_norm_w = b64_field(&json, "final_norm_w");
    let lm_head_w = b64_field(&json, "lm_head_w");
    let oracle_logits = b64_field(&json, "logits");

    let outputs = run_chained_decode(ordinal, &geom, &mut layers, &initial_hidden, position)
        .expect("chained decode");

    // Each local block is gated against an exact-input oracle. The chained
    // checks then use triangle-inequality max/L2 bounds from that local
    // error plus the measured propagation of the preceding residual.
    let inters = json["intermediates_per_layer"]
        .as_array()
        .expect("intermediates_per_layer array");
    assert_eq!(
        inters.len(),
        geom.num_layers as usize,
        "intermediates length mismatch"
    );
    let mut last_exact_input_residual = None;
    for (li, item) in inters.iter().enumerate() {
        let oracle_layer_input = if li == 0 {
            initial_hidden.clone()
        } else {
            b64_field(&inters[li - 1], "output_after_ffn")
        };
        let want_attn = b64_field(item, "output_after_attn");
        let want_ffn = b64_field(item, "output_after_ffn");
        let want_topk_idx = i32_bytes_to_vec(&b64_field(item, "ffn_topk_idx"));
        let want_topk_weights = b64_field(item, "ffn_topk_weights");
        let want_shared = b64_field(item, "ffn_shared_out");
        let want_expert_stack = b64_field(item, "ffn_expert_stack");
        let want_routed = b64_field(item, "ffn_moe_out");
        let want_exact_input_residual = b64_field(item, "ffn_output_hidden_exact_input");

        let (route_weights, got_topk_idx) =
            run_isolated_ffn_stage(ordinal, &geom, li, &layers[li].ffn, &want_attn, 1);
        assert_eq!(
            got_topk_idx, want_topk_idx,
            "layer {li} isolated FFN route indices differ"
        );
        assert_parity_bf16(
            &format!("layer {li} isolated FFN route weights"),
            &route_weights[..geom.top_k as usize * 2],
            &want_topk_weights,
            0.01,
            0.9999,
        );
        let (chained_route_weights, got_chained_topk_idx) = run_isolated_ffn_stage(
            ordinal,
            &geom,
            li,
            &layers[li].ffn,
            &outputs.per_layer_attn_out[li],
            1,
        );
        assert_eq!(
            got_chained_topk_idx, want_topk_idx,
            "layer {li} chained-input FFN route indices differ"
        );
        let chained_route_weight_delta = max_abs_delta_bf16(
            &chained_route_weights[..geom.top_k as usize * 2],
            &route_weights[..geom.top_k as usize * 2],
        );
        eprintln!(
            "[parity layer {li} FFN route propagation] max_abs={chained_route_weight_delta:.5e}"
        );
        let (got_shared, _) =
            run_isolated_ffn_stage(ordinal, &geom, li, &layers[li].ffn, &want_attn, 2);
        assert_parity_bf16(
            &format!("layer {li} isolated FFN shared"),
            &got_shared,
            &want_shared,
            0.05,
            0.999,
        );
        let isolated_shared_error = max_abs_delta_bf16(&got_shared, &want_shared);
        let (got_expert0, _) =
            run_isolated_ffn_stage(ordinal, &geom, li, &layers[li].ffn, &want_attn, 3);
        assert_parity_bf16(
            &format!("layer {li} isolated FFN expert 0"),
            &got_expert0,
            &want_expert_stack[..geom.hidden as usize * 2],
            0.05,
            0.999,
        );
        let (got_routed, _) =
            run_isolated_ffn_stage(ordinal, &geom, li, &layers[li].ffn, &want_attn, 4);
        assert_parity_bf16(
            &format!("layer {li} isolated FFN routed"),
            &got_routed,
            &want_routed,
            0.05,
            0.999,
        );
        let isolated_routed_error = max_abs_delta_bf16(&got_routed, &want_routed);
        let (got_exact_input_residual, _) =
            run_isolated_ffn_stage(ordinal, &geom, li, &layers[li].ffn, &want_attn, 5);
        let isolated_residual_bound = isolated_shared_error
            + isolated_routed_error
            + 0.5 * max_bf16_ulp(&got_shared)
            + 0.5 * max_bf16_ulp(&got_routed)
            + max_bf16_ulp(&want_exact_input_residual);
        assert_parity_bf16(
            &format!("layer {li} isolated FFN residual"),
            &got_exact_input_residual,
            &want_exact_input_residual,
            isolated_residual_bound,
            0.999,
        );
        let (got_chained_input_residual, _) = run_isolated_ffn_stage(
            ordinal,
            &geom,
            li,
            &layers[li].ffn,
            &outputs.per_layer_attn_out[li],
            5,
        );
        assert_eq!(
            got_chained_input_residual, outputs.per_layer_ffn_out[li],
            "layer {li} chained FFN output is not reproducible from its captured input"
        );
        let exact_input_kernel_error =
            max_abs_delta_bf16(&got_exact_input_residual, &want_exact_input_residual);
        let propagated_input_delta =
            max_abs_delta_bf16(&got_chained_input_residual, &got_exact_input_residual);
        let ffn_budget = handoff_budget(li, HandoffBoundary::Ffn);
        eprintln!(
            "[parity layer {li} FFN diagnostics] exact_input_kernel_error={exact_input_kernel_error:.5e} \
             propagated_input_delta={propagated_input_delta:.5e} \
             fixed_max_abs={:.5e} fixed_cos_floor={:.7}",
            ffn_budget.max_abs, ffn_budget.cosine_floor
        );

        let actual_layer_input = if li == 0 {
            initial_hidden.clone()
        } else {
            outputs.per_layer_ffn_out[li - 1].clone()
        };
        let got_exact_input_attn = run_isolated_attention(
            ordinal,
            &geom,
            &mut exact_input_attn_layers[li],
            &oracle_layer_input,
            position,
        );
        let got_chained_input_attn = run_isolated_attention(
            ordinal,
            &geom,
            &mut chained_input_attn_layers[li],
            &actual_layer_input,
            position,
        );
        assert_eq!(
            got_chained_input_attn, outputs.per_layer_attn_out[li],
            "layer {li} chained attention output is not reproducible from its captured input"
        );
        let exact_input_attn_error = max_abs_delta_bf16(&got_exact_input_attn, &want_attn);
        let propagated_attn_input_delta =
            max_abs_delta_bf16(&got_chained_input_attn, &got_exact_input_attn);
        let attention_budget = handoff_budget(li, HandoffBoundary::Attention);
        eprintln!(
            "[parity layer {li} attention diagnostics] exact_input_kernel_error={exact_input_attn_error:.5e} \
             propagated_input_delta={propagated_attn_input_delta:.5e} \
             fixed_max_abs={:.5e} fixed_cos_floor={:.7}",
            attention_budget.max_abs, attention_budget.cosine_floor
        );
        assert_chained_handoff_parity(
            &format!("layer {li} output_after_attn"),
            li,
            HandoffBoundary::Attention,
            &outputs.per_layer_attn_out[li],
            &want_attn,
        );
        assert_chained_handoff_parity(
            &format!("layer {li} output_after_ffn"),
            li,
            HandoffBoundary::Ffn,
            &outputs.per_layer_ffn_out[li],
            &want_ffn,
        );
        if li == 0 {
            let mut corrupted_handoff = outputs.per_layer_ffn_out[li].clone();
            let (target, source, lane_displacement, resulting_error) =
                corrupt_one_lane_from_adjacent(
                    &mut corrupted_handoff,
                    &want_ffn,
                    ffn_budget.max_abs,
                );
            let rejected = std::panic::catch_unwind(|| {
                assert_chained_handoff_parity(
                    "layer 0 actual FFN handoff with adjacent-lane indexing fault",
                    li,
                    HandoffBoundary::Ffn,
                    &corrupted_handoff,
                    &want_ffn,
                );
            });
            assert!(
                rejected.is_err(),
                "one-lane runtime handoff corruption passed the frozen layer-0 FFN budget"
            );
            eprintln!(
                "[parity corruption negative] layer={li} boundary=ffn target_lane={target} \
                 source_lane={source} lane_displacement={lane_displacement:.5e} \
                 resulting_oracle_error={resulting_error:.5e} rejected=true"
            );
        }
        last_exact_input_residual = Some(got_exact_input_residual);
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
    let exact_input_logits = host_final_norm_lm_head(
        &last_exact_input_residual.expect("last exact-input residual"),
        &final_norm_w,
        &lm_head_w,
        geom.hidden as usize,
        geom.vocab as usize,
        geom.rms_norm_eps,
    );
    assert_parity_bf16(
        "exact-input logits",
        &exact_input_logits,
        &oracle_logits,
        EXACT_INPUT_LOGITS_BUDGET.max_abs,
        EXACT_INPUT_LOGITS_BUDGET.cosine_floor,
    );
    let exact_input_logit_error = max_abs_delta_bf16(&exact_input_logits, &oracle_logits);
    let propagated_logit_delta = max_abs_delta_bf16(&logits, &exact_input_logits);
    eprintln!(
        "[parity logits diagnostics] exact_input_kernel_error={exact_input_logit_error:.5e} \
         propagated_input_delta={propagated_logit_delta:.5e} \
         fixed_max_abs={:.5e} fixed_cos_floor={:.7}",
        CHAINED_LOGITS_BUDGET.max_abs, CHAINED_LOGITS_BUDGET.cosine_floor
    );
    assert_parity_bf16(
        "chained logits",
        &logits,
        &oracle_logits,
        CHAINED_LOGITS_BUDGET.max_abs,
        CHAINED_LOGITS_BUDGET.cosine_floor,
    );
}

// =============================================================================
// Phase 3e: persistent decode megakernel parity test.
//
// Drives the production runtime layer owner and chain dispatcher with the
// same fixtures the chained-decode test uses, then asserts the final hidden
// and folded lm-head logits match the chained path and oracle. The two paths run the
// IDENTICAL `__device__` phase functions (extracted in Phase 3a-3d) — only
// the launch orchestration differs (1 cooperative launch vs 80 step
// launches, with `reset_counters_16` between phases inside the
// megakernel). So the comparison floor is very tight (cos_sim ≥ 0.99999,
// max_abs ≤ 1e-3).
//
// This test also validates the production `LoadedQwen36Layers` descriptor
// ownership and `run_chain_step` dispatch used by the engine.
// =============================================================================

#[test]
fn multilayer_persistent_decode_matches_chained() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    let json_path = tracked_multilayer_oracle_path();
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
    let final_norm_w = b64_field(&json, "final_norm_w");
    let lm_head_w = b64_field(&json, "lm_head_w");
    let weight_mode = if int4_mode {
        Qwen36WeightMode::Int4
    } else {
        Qwen36WeightMode::Bf16
    };
    let mut loaded = LoadedQwen36Layers::dense(layers, weight_mode);

    // Snapshot the linear-attn state so we can reset between the chained
    // and persistent runs (linear-attn mutates conv_state +
    // recurrent_state per token).
    let snapshot =
        save_linear_attn_state(ordinal, loaded.layers()).expect("save_linear_attn_state");
    let execution = Qwen36ExecutionOptions::default();

    // ---- Runtime-owned chained-path reference ----
    let chained = run_chain_step(Qwen36ChainStep {
        ordinal,
        geom: &geom,
        loaded_layers: &mut loaded,
        initial_hidden: &initial_hidden,
        position: PositionPair::dense(position),
        step: 0,
        accurate_stage_timings: false,
        fold: None,
        download_final_hidden: true,
        expert_prefetch: None,
        execution: &execution,
    })
    .expect("runtime chained decode")
    .outputs;

    // Restore linear state to the pre-chained values so the persistent
    // run sees the same starting point.
    restore_linear_attn_state(
        ordinal,
        loaded
            .layers_mut_before_persistent()
            .expect("mutable layers before persistent descriptors"),
        &snapshot,
    )
    .expect("restore_linear_attn_state");

    if !int4_mode {
        let err = loaded
            .enable_persistent(ordinal, &geom)
            .expect_err("BF16 persistent execution must be rejected before mutation");
        assert!(err.to_string().contains("does not support Bf16"));
        return;
    }

    // ---- Persistent megakernel + folded lm-head via runtime dispatch ----
    loaded
        .enable_persistent(ordinal, &geom)
        .expect("enable runtime persistent decode");
    let final_norm_w_buf = upload_bf16(
        ordinal,
        &[geom.hidden as usize],
        &final_norm_w,
        "final norm",
    );
    let lm_head_w_buf = upload_bf16(
        ordinal,
        &[geom.vocab as usize, geom.hidden as usize],
        &lm_head_w,
        "lm head",
    );
    let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geom.vocab as usize])
        .expect("alloc folded logits");
    let persistent = run_chain_step(Qwen36ChainStep {
        ordinal,
        geom: &geom,
        loaded_layers: &mut loaded,
        initial_hidden: &initial_hidden,
        position: PositionPair::dense(position),
        step: 0,
        accurate_stage_timings: false,
        fold: Some(LmHeadFold {
            final_norm_w: &final_norm_w_buf,
            lm_head_w: &lm_head_w_buf,
            logits_out: Some(&mut logits_buf),
            top1_out: None,
            vocab: geom.vocab,
        }),
        download_final_hidden: true,
        expert_prefetch: None,
        execution: &execution,
    })
    .expect("runtime persistent decode");
    assert!(persistent.lm_head_folded);
    assert!(!persistent.lm_head_folded_top1);
    let persistent_final = persistent.outputs.final_hidden_bytes;
    let mut persistent_logits = vec![0u8; geom.vocab as usize * 2];
    copy_d2h(
        ordinal,
        persistent_logits.as_mut_ptr() as *mut c_void,
        logits_buf.as_ptr(),
        persistent_logits.len(),
    )
    .expect("download folded lm-head logits");
    let chained_logits = host_final_norm_lm_head(
        &chained.final_hidden_bytes,
        &final_norm_w,
        &lm_head_w,
        geom.hidden as usize,
        geom.vocab as usize,
        geom.rms_norm_eps,
    );

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
    assert_parity_bf16(
        "persistent folded lm-head vs chained host lm-head",
        &persistent_logits,
        &chained_logits,
        0.05,
        0.99999,
    );

    // Segmented persistent is the sparse-VMM orchestration: each layer runs
    // attention + router top-k, returns to the host for remap, then resumes
    // that layer's FFN. With a no-op remap callback it must match the same
    // chained reference.
    let segmented_outputs = unsafe {
        loaded.with_experimental_parts(|layers, scratch| {
            restore_linear_attn_state(ordinal, layers, &snapshot)
                .expect("restore_linear_attn_state before segmented persistent");
            scratch
                .expect("persistent scratch")
                .run_sparse_with_expert_prefetch(
                    ordinal,
                    &initial_hidden,
                    position,
                    CACHE_POS_INHERIT,
                    |_phase, _layer, _topk| Ok(()),
                )
        })
    }
    .expect("experimental segmented persistent comparison");
    assert_parity_bf16(
        "segmented persistent sparse vs chained final_hidden",
        &segmented_outputs.final_hidden_bytes,
        &chained.final_hidden_bytes,
        1e-3,
        0.99999,
    );
}
