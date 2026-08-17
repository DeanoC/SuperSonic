//! Incremental Qwen3.8 GQH GGUF crawl. Each test is one rung; later rungs
//! assume earlier ones still pass.

use std::path::PathBuf;

use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
use kernel_ffi::gqh::{self, RUNG_GQH2_H, RUNG_GQH3};
use model_store::gguf::GgufFile;
use model_store::gqh::GqhRung;
use qwen35::gguf_ingest::{check_mapping, load_text_config};
use qwen35::weights::{
    infer_lowbit_type, is_gqh_qtype, matmul_gqh, LayerKind, Qwen35Weights, LOWBIT_GQH2_H,
    LOWBIT_GQH3, LOWBIT_GGML_Q2_K, LOWBIT_GGML_Q4_K, LOWBIT_GGML_Q5_K, LOWBIT_GGML_Q6_K,
    LOWBIT_GGML_Q8_0,
};

fn gguf_path() -> Option<PathBuf> {
    let path = PathBuf::from("/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf");
    path.is_file().then_some(path)
}

fn hf_dir() -> PathBuf {
    PathBuf::from("/data/models/Qwen3.8-27B")
}

#[test]
fn rung2_hf_config_matches_gguf_geometry() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    let config = load_text_config(&hf_dir()).expect("hf config");
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.linear_num_value_heads, 48);
    assert_eq!(config.linear_value_dim(), 6144);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.num_full_attention_layers(), 16);

    let file = GgufFile::open(&path).expect("open gguf");
    assert_eq!(file.kv("general.architecture"), Some("qwen35"));
    let mapped = check_mapping(&file, &config).expect("role map");
    assert!(mapped.len() > 800, "mapped {}", mapped.len());
    assert!(file.tensor("blk.64.nextn.eh_proj.weight").is_some());
}

#[test]
fn rung3_q2k_embed_row_is_finite() {
    let Some(path) = gguf_path() else {
        return;
    };
    let file = GgufFile::open(&path).expect("open gguf");
    let embed = file.tensor("token_embd.weight").expect("embed");
    assert_eq!(embed.tensor_type, 10);
    let cols = embed.dims[0];
    let rows = embed.dims[1];
    let packed = file.tensor_bytes("token_embd.weight").expect("bytes");
    let row_bytes = model_store::q2k::row_bytes(cols).expect("row bytes");
    assert_eq!(packed.len(), row_bytes * rows);

    for token in [0usize, 1, 248044] {
        let mut out = vec![0.0f32; cols];
        model_store::q2k::decode_row(
            &packed[token * row_bytes..(token + 1) * row_bytes],
            cols,
            &mut out,
        )
        .unwrap_or_else(|e| panic!("token {token}: {e}"));
        assert!(
            out.iter().all(|v| v.is_finite()),
            "token {token} produced a non-finite embed"
        );
        let energy: f32 = out.iter().map(|v| v * v).sum();
        assert!(energy > 0.0, "token {token} embed is all zeros");
    }
}

#[test]
fn rung5_upload_packed_mapped_tensors_to_device0() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    set_backend(Backend::Hip);
    if kernel_ffi::query_gpu_info(0).is_err() {
        eprintln!("skip: no HIP device 0");
        return;
    }
    let config = load_text_config(&hf_dir()).expect("hf config");
    let file = GgufFile::open(&path).expect("open gguf");
    let mapped = check_mapping(&file, &config).expect("role map");

    let mut buffers = Vec::new();
    let mut bytes = 0usize;
    for item in &mapped {
        let tensor = file.tensor(&item.gguf_name).expect("tensor");
        let data = file.tensor_bytes(&item.gguf_name).expect("bytes");
        let (dtype, shape) = if tensor.tensor_type == 0 {
            (ScalarType::F32, tensor.dims.clone())
        } else {
            (ScalarType::U8, vec![data.len()])
        };
        let buf = GpuBuffer::from_host_bytes(0, dtype, &shape, data)
            .unwrap_or_else(|e| panic!("upload {}: {e}", item.gguf_name));
        bytes += buf.len_bytes();
        buffers.push(buf);
    }
    gpu_hal::sync(0).expect("sync");
    println!(
        "uploaded {} mapped tensors, {:.2} GiB packed on device 0",
        buffers.len(),
        bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    assert!(bytes > 8 * 1024 * 1024 * 1024, "expected >8 GiB packed");
    assert!(bytes < 16 * 1024 * 1024 * 1024, "packed upload exceeded 16 GiB");
    drop(buffers);
}

fn require_hip() -> Option<usize> {
    set_backend(Backend::Hip);
    kernel_ffi::query_gpu_info(0).ok().map(|_| 0)
}

#[test]
fn rung6_load_gguf_weights() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&hf_dir()).expect("hf config");
    let weights = Qwen35Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    assert_eq!(weights.layers.len(), 64);
    assert_eq!(weights.embed_tokens.dtype(), ScalarType::BF16);
    assert_eq!(weights.embed_tokens.shape(), &[248320, 5120]);
    assert_eq!(weights.lm_head.dtype(), ScalarType::U8);
    assert_eq!(
        infer_lowbit_type(weights.lm_head.as_ref(), 5120, false),
        LOWBIT_GQH2_H
    );
    assert!(weights.gqh_header("lm_head.weight").is_some());

    let l0 = &weights.layers[0];
    assert!(matches!(l0.kind, LayerKind::Linear));
    let lin = l0.linear.as_ref().expect("linear 0");
    assert_eq!(lin.qkv_proj_w.shape()[0], 10240);
    assert_eq!(lin.z_proj_w.shape()[0], 6144);
    assert_eq!(lin.conv1d_w.shape(), &[10240, 1, 4]);
    let gate_ty = infer_lowbit_type(&l0.gate_proj_w, 5120, false);
    assert!(
        gate_ty == LOWBIT_GQH3 || gate_ty == LOWBIT_GQH2_H,
        "layer0 gate qtype {gate_ty}"
    );
    assert!(weights.gqh_header("layers.0.mlp.gate_proj").is_some());

    let l3 = &weights.layers[3];
    assert!(matches!(l3.kind, LayerKind::Full));
    let full = l3.full.as_ref().expect("full 3");
    assert_eq!(full.q_proj_w.shape()[0], 12288);
    assert_eq!(full.k_proj_w.shape()[0], 1024);
    println!(
        "load_gguf: 64 layers, {} GQH headers, embed {:?}, lm_head {:?}",
        weights.gqh_headers.len(),
        weights.embed_tokens.shape(),
        weights.lm_head.shape()
    );
}

#[test]
fn rung7_loaded_ffn_up_gqh_matvec() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&hf_dir()).expect("hf config");
    let file = GgufFile::open(&path).expect("open");
    let weights = Qwen35Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let header = weights
        .gqh_header("layers.0.mlp.up_proj")
        .cloned()
        .expect("up header");
    let tensor = file.tensor("blk.0.ffn_up.weight").expect("up tensor");
    let rung = GqhRung::from_ggml_type(tensor.tensor_type).expect("gqh");
    let cols = tensor.dims[0];
    let rows = 8usize;
    let packed_all = file.tensor_bytes("blk.0.ffn_up.weight").expect("bytes");
    let row_bytes = packed_all.len() / tensor.dims[1];
    let packed = &packed_all[..row_bytes * rows];
    let mut cpu_w = vec![0.0f32; rows * cols];
    for r in 0..rows {
        model_store::gqh::decode_row(
            rung,
            &packed[r * row_bytes..(r + 1) * row_bytes],
            cols,
            Some(header.clone()),
            &mut cpu_w[r * cols..(r + 1) * cols],
        )
        .expect("cpu decode");
    }
    let x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let mut want = vec![0.0f32; rows];
    for r in 0..rows {
        let mut lane = [0.0f32; 32];
        for sb in 0..(cols / 256) {
            for lane_i in 0..32 {
                let j0 = lane_i * 8;
                for t in 0..8 {
                    let j = sb * 256 + j0 + t;
                    lane[lane_i] += cpu_w[r * cols + j] * x[j];
                }
            }
        }
        let mut accs = lane;
        let mut off = 16;
        while off > 0 {
            for i in 0..off {
                accs[i] += accs[i + off];
            }
            off >>= 1;
        }
        want[r] = accs[0];
    }

    let row_bytes_w = weights.layers[0].up_proj_w.shape()[1];
    let mut y = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("y");
    let x_buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[cols], &{
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
    })
    .expect("x");
    // Slice is the first `rows` packed rows of the already-uploaded weight.
    let packed_prefix =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[rows, row_bytes_w], packed)
            .expect("prefix");
    gqh::matvec(
        ordinal,
        match rung {
            GqhRung::Gqh3 => RUNG_GQH3,
            GqhRung::Gqh2H => RUNG_GQH2_H,
            GqhRung::Gqh2C => gqh::RUNG_GQH2_C,
        },
        &packed_prefix,
        &x_buf,
        &mut y,
        cols,
        rows,
        1,
        cols,
        rows,
        header.tensor_scale,
        header.grid_code,
    )
    .expect("hip matvec");
    let got_bytes = y.to_host_bytes().expect("d2h");
    let got: Vec<f32> = got_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "ffn_up matvec [{i}] got {g} want {w}"
        );
    }
}

#[test]
fn rung8_embed_row_then_prefill_gqh_qkv() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&hf_dir()).expect("hf config");
    let weights = Qwen35Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let hidden = config.hidden_size;
    let token = 9707usize; // arbitrary in-vocab token for gather+matvec
    let mut hidden_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("hidden");
    gpu_hal::copy_d2d(
        ordinal,
        hidden_bf16.as_mut_ptr(),
        weights.embed_tokens.offset_ptr(token * hidden * 2),
        hidden * 2,
    )
    .expect("embed gather");

    let header = weights
        .gqh_header("layers.0.linear_attn.in_proj_qkv")
        .cloned()
        .expect("qkv header");
    let qkv = &weights.layers[0].linear.as_ref().unwrap().qkv_proj_w;
    let n = qkv.shape()[0];
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n]).expect("qkv out");
    let rung = infer_lowbit_type(qkv, hidden, false);
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_gqh(
        ordinal,
        n,
        hidden,
        &hidden_bf16,
        qkv,
        header.tensor_scale,
        header.grid_code,
        match rung {
            LOWBIT_GQH3 => RUNG_GQH3,
            LOWBIT_GQH2_H => RUNG_GQH2_H,
            other => panic!("qkv not GQH: {other}"),
        },
        &mut out,
    )
    .expect("prefill gqh qkv");
    let bytes = out.to_host_bytes().expect("d2h");
    let vals: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| {
            half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()
        })
        .collect();
    assert_eq!(vals.len(), n);
    assert!(vals.iter().all(|v| v.is_finite()), "qkv out not finite");
    let energy: f32 = vals.iter().map(|v| v * v).sum();
    assert!(energy > 0.0, "qkv out is all zeros");
    println!(
        "rung8: token={token} qkv n={n} energy={energy:.4} first3={:?}",
        &vals[..3]
    );
}

fn qtype_name(qtype: i32) -> &'static str {
    match qtype {
        LOWBIT_GGML_Q8_0 => "Q8_0",
        LOWBIT_GGML_Q2_K => "Q2_K",
        LOWBIT_GGML_Q4_K => "Q4_K",
        LOWBIT_GGML_Q5_K => "Q5_K",
        LOWBIT_GGML_Q6_K => "Q6_K",
        LOWBIT_GQH3 => "GQH3",
        LOWBIT_GQH2_H => "GQH2_H",
        110 => "GQH2_C",
        other => {
            let _ = other;
            "unknown"
        }
    }
}

fn bf16_vals(buf: &GpuBuffer) -> Vec<f32> {
    let bytes = buf.to_host_bytes().expect("d2h");
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn summarize(vals: &[f32], label: &str) {
    let nans = vals.iter().filter(|v| v.is_nan()).count();
    let infs = vals.iter().filter(|v| v.is_infinite()).count();
    let energy: f32 = vals.iter().filter(|v| v.is_finite()).map(|v| v * v).sum();
    let max_abs = vals
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    println!(
        "{label}: n={} nan={nans} inf={infs} energy={energy:.4} max_abs={max_abs:.4}",
        vals.len()
    );
}

fn assert_finite_energy(vals: &[f32], label: &str) {
    assert!(!vals.is_empty(), "{label} empty");
    summarize(vals, label);
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "{label} produced a non-finite value"
    );
    let energy: f32 = vals.iter().map(|v| v * v).sum();
    assert!(energy > 0.0, "{label} is all zeros");
}

fn gather_embed(ordinal: usize, weights: &Qwen35Weights, token: usize) -> GpuBuffer {
    let hidden = weights.config.hidden_size;
    let mut hidden_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("hidden");
    gpu_hal::copy_d2d(
        ordinal,
        hidden_bf16.as_mut_ptr(),
        weights.embed_tokens.offset_ptr(token * hidden * 2),
        hidden * 2,
    )
    .expect("embed gather");
    hidden_bf16
}

fn dispatch_proj(
    ordinal: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    out: &mut GpuBuffer,
) {
    let qtype = infer_lowbit_type(weight, k, false);
    if is_gqh_qtype(qtype) {
        matmul_gqh(ordinal, n, k, lhs, weight, qtype, out).unwrap_or_else(|e| {
            panic!("gqh n={n} k={k} qtype={qtype}: {e}");
        });
        return;
    }
    assert!(
        qtype == LOWBIT_GGML_Q8_0
            || qtype == LOWBIT_GGML_Q4_K
            || qtype == LOWBIT_GGML_Q5_K
            || qtype == LOWBIT_GGML_Q6_K,
        "unsupported packed qtype {qtype} for n={n} k={k}"
    );
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
        ordinal, 1, 1, n, k, lhs, weight, weight, weight, None, 0, qtype, out,
    )
    .unwrap_or_else(|e| panic!("ggml-K n={n} k={k} qtype={qtype}: {e}"));
}

#[test]
fn rung9_gqh_dispatch_and_ggml_k_kv() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&hf_dir()).expect("hf config");
    let weights = Qwen35Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let hidden = config.hidden_size;
    let token = 9707usize;
    let hidden_bf16 = gather_embed(ordinal, &weights, token);

    let qkv = &weights.layers[0].linear.as_ref().unwrap().qkv_proj_w;
    let qkv_ty = infer_lowbit_type(qkv, hidden, false);
    assert!(is_gqh_qtype(qkv_ty), "qkv qtype {qkv_ty}");
    let registered = kernel_ffi::gqh::lookup_header(qkv.as_ptr()).expect("qkv header registry");
    let stored = weights
        .gqh_header("layers.0.linear_attn.in_proj_qkv")
        .expect("qkv stored header");
    assert_eq!(registered.tensor_scale, stored.tensor_scale);
    assert_eq!(registered.grid_code, stored.grid_code);

    let n = qkv.shape()[0];
    let mut qkv_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[n]).expect("qkv out");
    dispatch_proj(ordinal, n, hidden, &hidden_bf16, qkv, &mut qkv_out);
    let qkv_vals = bf16_vals(&qkv_out);
    assert_finite_energy(&qkv_vals, "gqh qkv dispatch");

    let full = weights.layers[3].full.as_ref().expect("full layer 3");
    let k_ty = infer_lowbit_type(&full.k_proj_w, hidden, false);
    let v_ty = infer_lowbit_type(&full.v_proj_w, hidden, false);
    let a_ty = infer_lowbit_type(
        &weights.layers[0].linear.as_ref().unwrap().a_proj_w,
        hidden,
        false,
    );
    let b_ty = infer_lowbit_type(
        &weights.layers[0].linear.as_ref().unwrap().b_proj_w,
        hidden,
        false,
    );
    println!(
        "rung9 qtypes: qkv={} k={} v={} a={} b={}",
        qtype_name(qkv_ty),
        qtype_name(k_ty),
        qtype_name(v_ty),
        qtype_name(a_ty),
        qtype_name(b_ty)
    );
    assert_ne!(k_ty, LOWBIT_GGML_Q2_K, "k_proj must not be Q2_K");
    assert_ne!(v_ty, LOWBIT_GGML_Q2_K, "v_proj must not be Q2_K");

    let kv_n = full.k_proj_w.shape()[0];
    let mut k_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_n]).expect("k out");
    dispatch_proj(ordinal, kv_n, hidden, &hidden_bf16, &full.k_proj_w, &mut k_out);
    assert_finite_energy(&bf16_vals(&k_out), "layer3 k_proj");

    let mut v_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_n]).expect("v out");
    dispatch_proj(ordinal, kv_n, hidden, &hidden_bf16, &full.v_proj_w, &mut v_out);
    assert_finite_energy(&bf16_vals(&v_out), "layer3 v_proj");

    let ggml_k = [k_ty, v_ty, a_ty, b_ty]
        .into_iter()
        .any(|ty| matches!(ty, LOWBIT_GGML_Q8_0 | LOWBIT_GGML_Q4_K | LOWBIT_GGML_Q5_K | LOWBIT_GGML_Q6_K));
    println!("rung9: ggml-K among k/v/a/b = {ggml_k}");
}

#[test]
fn rung10_linear_layer0_norm_projs_mlp() {
    let Some(path) = gguf_path() else {
        return;
    };
    if !hf_dir().join("config.json").is_file() {
        return;
    }
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&hf_dir()).expect("hf config");
    let weights = Qwen35Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let nv = config.linear_num_value_heads;
    let token = 9707usize;
    let hidden_bf16 = gather_embed(ordinal, &weights, token);

    let rms = if config.rms_norm_add_unit_offset {
        kernel_ffi::prefill_ffi::rms_norm_rows
    } else {
        kernel_ffi::prefill_ffi::rms_norm_rows_plain
    };
    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("normed");
    rms(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        config.rms_norm_eps as f32,
        &hidden_bf16,
        &weights.layers[0].input_norm_w,
        &mut normed,
    )
    .expect("input rms");
    assert_finite_energy(&bf16_vals(&normed), "input rms");

    let lin = weights.layers[0].linear.as_ref().unwrap();
    let qkv_n = lin.qkv_proj_w.shape()[0];
    let z_n = lin.z_proj_w.shape()[0];
    let mut qkv = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_n]).expect("qkv");
    let mut z = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[z_n]).expect("z");
    let mut a = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv]).expect("a");
    let mut b = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv]).expect("b");
    dispatch_proj(ordinal, qkv_n, hidden, &normed, &lin.qkv_proj_w, &mut qkv);
    dispatch_proj(ordinal, z_n, hidden, &normed, &lin.z_proj_w, &mut z);
    dispatch_proj(ordinal, nv, hidden, &normed, &lin.a_proj_w, &mut a);
    dispatch_proj(ordinal, nv, hidden, &normed, &lin.b_proj_w, &mut b);
    assert_finite_energy(&bf16_vals(&qkv), "layer0 qkv");
    assert_finite_energy(&bf16_vals(&z), "layer0 z");
    assert_finite_energy(&bf16_vals(&a), "layer0 a");
    assert_finite_energy(&bf16_vals(&b), "layer0 b");

    let mut post = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("post");
    rms(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        config.rms_norm_eps as f32,
        &hidden_bf16,
        &weights.layers[0].post_attn_norm_w,
        &mut post,
    )
    .expect("post rms");

    let gate_ty = infer_lowbit_type(&weights.layers[0].gate_proj_w, hidden, false);
    let up_ty = infer_lowbit_type(&weights.layers[0].up_proj_w, hidden, false);
    let down_ty = infer_lowbit_type(&weights.layers[0].down_proj_w, inter, false);
    println!(
        "rung10 mlp qtypes: gate={} up={} down={}",
        qtype_name(gate_ty),
        qtype_name(up_ty),
        qtype_name(down_ty)
    );
    summarize(&bf16_vals(&post), "post rms");

    let mut gate = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[inter]).expect("gate");
    let mut up = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[inter]).expect("up");
    let mut swiglu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[inter]).expect("swiglu");
    let mut down = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("down");
    dispatch_proj(ordinal, inter, hidden, &post, &weights.layers[0].gate_proj_w, &mut gate);
    dispatch_proj(ordinal, inter, hidden, &post, &weights.layers[0].up_proj_w, &mut up);
    summarize(&bf16_vals(&gate), "layer0 gate");
    summarize(&bf16_vals(&up), "layer0 up");
    kernel_ffi::prefill_ffi::swiglu_mul(ordinal, ScalarType::BF16, inter, &gate, &up, &mut swiglu)
        .expect("swiglu");
    summarize(&bf16_vals(&swiglu), "layer0 swiglu");
    dispatch_proj(ordinal, hidden, inter, &swiglu, &weights.layers[0].down_proj_w, &mut down);
    let down_vals = bf16_vals(&down);
    assert_finite_energy(&down_vals, "layer0 mlp down");
    println!(
        "rung10: layer0 qkv={} z={} a={} mlp_down energy={:.4}",
        qkv_n,
        z_n,
        nv,
        down_vals.iter().map(|v| v * v).sum::<f32>()
    );
}
