//! Incremental Qwen3.8 GQH GGUF crawl. Each test is one rung; later rungs
//! assume earlier ones still pass.

use std::path::PathBuf;

use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::gqh::{self, RUNG_GQH2_H, RUNG_GQH3};
use model_store::gguf::GgufFile;
use model_store::gqh::GqhRung;
use qwen38::desc_builder::build_int4_scale_descs;
use qwen38::gguf_ingest::{check_mapping, load_text_config};
use qwen38::weights::{
    infer_lowbit_type, is_gqh_qtype, matmul_gqh, LayerKind, Qwen38Weights, LOWBIT_GGML_Q2_K,
    LOWBIT_GGML_Q4_K, LOWBIT_GGML_Q5_K, LOWBIT_GGML_Q6_K, LOWBIT_GGML_Q8_0, LOWBIT_GQH2_H,
    LOWBIT_GQH3,
};

fn gguf_path() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_GQH_GGUF") else {
        if require_gqh_artifacts() {
            panic!("SUPERSONIC_GQH_GGUF is required for Qwen3.8 GQH artifact tests");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if path.is_file() {
        Some(path)
    } else if require_gqh_artifacts() {
        panic!(
            "SUPERSONIC_GQH_GGUF points to a missing artifact: {}",
            path.display()
        );
    } else {
        None
    }
}

fn require_gqh_artifacts() -> bool {
    std::env::var("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").as_deref() == Ok("1")
}

fn qwen38_model_dir() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_QWEN38_MODEL_DIR") else {
        if require_gqh_artifacts() {
            panic!("SUPERSONIC_QWEN38_MODEL_DIR is required for Qwen3.8 GQH artifact tests");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if !path.is_dir() {
        if require_gqh_artifacts() {
            panic!(
                "SUPERSONIC_QWEN38_MODEL_DIR points to a missing model directory: {}",
                path.display()
            );
        }
        return None;
    }
    for required in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let child = path.join(required);
        if !child.is_file() {
            if require_gqh_artifacts() {
                panic!(
                    "SUPERSONIC_QWEN38_MODEL_DIR is missing required file: {}",
                    child.display()
                );
            }
            return None;
        }
    }
    Some(path)
}

#[test]
fn rung2_hf_config_matches_gguf_geometry() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    assert_eq!(config.num_hidden_layers, 64);
    assert_eq!(config.linear_num_value_heads, 48);
    assert_eq!(config.linear_value_dim(), 6144);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.num_full_attention_layers(), 16);

    let file = GgufFile::open(&path).expect("open gguf");
    // `qwen35` is the historical architecture key in the external GQH wire
    // schema; the crate and public model identity are Qwen3.8.
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
#[ignore = "requires R9700 artifact CI"]
fn rung5_upload_packed_mapped_tensors_to_device0() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    if kernel_ffi::query_gpu_info(0).is_err() {
        eprintln!("skip: no HIP device 0");
        return;
    }
    let config = load_text_config(&model_dir).expect("hf config");
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
    assert!(
        bytes < 16 * 1024 * 1024 * 1024,
        "packed upload exceeded 16 GiB"
    );
    drop(buffers);
}

fn require_hip() -> Option<usize> {
    kernel_ffi::query_gpu_info(0).ok().map(|_| 0)
}

#[test]
#[ignore = "requires R9700 artifact CI"]
fn rung6_load_gguf_weights() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
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
    assert!(weights.gqh_sidecars.contains_key("layers.0.mlp.gate_proj"));
    let descs = build_int4_scale_descs(&weights).expect("GQH scale descs");
    assert_eq!(descs.len(), 64);
    assert!(
        descs[0].gate_proj_type == LOWBIT_GQH3 || descs[0].gate_proj_type == LOWBIT_GQH2_H,
        "desc gate qtype {}",
        descs[0].gate_proj_type
    );
    assert!(!descs[0].gate_proj_scale.is_null());
    assert_eq!(descs[0].b_proj_type, LOWBIT_GGML_Q8_0);
    assert_eq!(descs[0].a_proj_type, LOWBIT_GGML_Q8_0);

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
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let file = GgufFile::open(&path).expect("open");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
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
    let packed_tight = &packed_all[..row_bytes * rows];
    let mut cpu_w = vec![0.0f32; rows * cols];
    for r in 0..rows {
        model_store::gqh::decode_row(
            rung,
            &packed_tight[r * row_bytes..(r + 1) * row_bytes],
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
    let device_row_bytes = model_store::gqh::device_row_bytes(rung, cols).expect("device row");
    assert_eq!(row_bytes_w, device_row_bytes, "loaded GQH device stride");
    let packed_device =
        model_store::gqh::planarize(rung, rows, cols, packed_tight).expect("planarize rows");
    assert_eq!(
        packed_device.len(),
        rows * device_row_bytes,
        "GQH upload boundary requires device-planar rows ({device_row_bytes} B)"
    );
    let mut y = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("y");
    let x_buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[cols], &{
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
    })
    .expect("x");
    // Convert the first tight GGUF rows to the device-planar layout used by matvec.
    let packed_prefix = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::U8,
        &[rows, device_row_bytes],
        &packed_device,
    )
    .expect("prefix");
    gqh::matvec(
        ordinal,
        match rung {
            GqhRung::Gqh3 => RUNG_GQH3,
            GqhRung::Gqh2H => RUNG_GQH2_H,
            GqhRung::Gqh2C => gqh::RUNG_GQH2_C,
            GqhRung::Gqh4 => gqh::RUNG_GQH4,
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
fn rung7b_batched_gqh_matvec_ncols4() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let file = GgufFile::open(&path).expect("open");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let header = weights
        .gqh_header("layers.0.mlp.up_proj")
        .cloned()
        .expect("up header");
    let tensor = file.tensor("blk.0.ffn_up.weight").expect("up tensor");
    let rung = GqhRung::from_ggml_type(tensor.tensor_type).expect("gqh");
    let cols = tensor.dims[0];
    let rows = 8usize;
    let ncols = 4usize;
    let packed_all = file.tensor_bytes("blk.0.ffn_up.weight").expect("bytes");
    let row_bytes = packed_all.len() / tensor.dims[1];
    let packed_tight = &packed_all[..row_bytes * rows];
    let mut cpu_w = vec![0.0f32; rows * cols];
    for r in 0..rows {
        model_store::gqh::decode_row(
            rung,
            &packed_tight[r * row_bytes..(r + 1) * row_bytes],
            cols,
            Some(header.clone()),
            &mut cpu_w[r * cols..(r + 1) * cols],
        )
        .expect("cpu decode");
    }
    let mut x = vec![0.0f32; ncols * cols];
    let mut want = vec![0.0f32; ncols * rows];
    for col in 0..ncols {
        for i in 0..cols {
            x[col * cols + i] = ((i + 3 * col) % 17) as f32 - 8.0;
            x[col * cols + i] /= 8.0;
        }
        for r in 0..rows {
            let mut lane = [0.0f32; 32];
            for sb in 0..(cols / 256) {
                for lane_i in 0..32 {
                    let j0 = lane_i * 8;
                    for t in 0..8 {
                        let j = sb * 256 + j0 + t;
                        lane[lane_i] += cpu_w[r * cols + j] * x[col * cols + j];
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
            want[col * rows + r] = accs[0];
        }
    }
    let x_buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[ncols, cols], &{
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
    })
    .expect("x");
    let device_row_bytes = model_store::gqh::device_row_bytes(rung, cols).expect("device row");
    let packed_device =
        model_store::gqh::planarize(rung, rows, cols, packed_tight).expect("planarize rows");
    assert_eq!(
        packed_device.len(),
        rows * device_row_bytes,
        "GQH upload boundary requires device-planar rows ({device_row_bytes} B)"
    );
    let packed_prefix = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::U8,
        &[rows, device_row_bytes],
        &packed_device,
    )
    .expect("prefix");
    let mut y = GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols, rows]).expect("y");
    gqh::matvec(
        ordinal,
        match rung {
            GqhRung::Gqh3 => RUNG_GQH3,
            GqhRung::Gqh2H => RUNG_GQH2_H,
            GqhRung::Gqh2C => gqh::RUNG_GQH2_C,
            GqhRung::Gqh4 => gqh::RUNG_GQH4,
        },
        &packed_prefix,
        &x_buf,
        &mut y,
        cols,
        rows,
        ncols,
        cols,
        rows,
        header.tensor_scale,
        header.grid_code,
    )
    .expect("hip batched matvec");
    let got_bytes = y.to_host_bytes().expect("d2h");
    let got: Vec<f32> = got_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "batched ffn_up [{i}] got {g} want {w}"
        );
    }
}

/// Large-m GQH must use dequant+GEMM (llama MMVQ cap is 8) and stay close
/// to the fused matvec on the same weights/activations.
#[test]
fn rung7c_gqh_large_m_dequant_gemm_matches_fused() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let file = GgufFile::open(&path).expect("open");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
    let _file = file;
    let header = weights
        .gqh_header("layers.0.mlp.up_proj")
        .cloned()
        .expect("up header");
    let w = &weights.layers[0].up_proj_w;
    let qtype = infer_lowbit_type(w, config.hidden_size, false);
    assert!(
        qtype == LOWBIT_GQH3 || qtype == LOWBIT_GQH2_H,
        "up qtype {qtype}"
    );
    let k = config.hidden_size;
    let n = 64usize;
    let m = 16usize;
    let mut x_bf16 = vec![0u8; m * k * 2];
    for row in 0..m {
        for col in 0..k {
            let v = (((col + 3 * row) % 17) as f32 - 8.0) / 32.0;
            let bits = half::bf16::from_f32(v).to_bits().to_le_bytes();
            x_bf16[(row * k + col) * 2] = bits[0];
            x_bf16[(row * k + col) * 2 + 1] = bits[1];
        }
    }
    let x = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &x_bf16).expect("x");
    let _ = header;

    std::env::set_var("SUPERSONIC_GQH_FORCE_FUSED", "1");
    let mut fused = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n]).expect("fused");
    matmul_gqh(ordinal, m, n, k, &x, w, qtype, &mut fused).expect("fused");
    std::env::remove_var("SUPERSONIC_GQH_FORCE_FUSED");
    let mut gemm = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n]).expect("gemm");
    matmul_gqh(ordinal, m, n, k, &x, w, qtype, &mut gemm).expect("gemm");
    // Dequant+GEMM queues hipBLAS work on a non-default stream. Cross that
    // stream boundary before the direct D2H read so release timing cannot
    // observe the output buffer before GEMM has completed.
    kernel_ffi::gqh::gemm_flush();

    let decode = |buf: &GpuBuffer| -> Vec<f32> {
        buf.to_host_bytes()
            .expect("d2h")
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    };
    let a = decode(&fused);
    let b = decode(&gemm);
    println!(
        "rung7c sample fused {:?} gemm {:?}",
        &a[..4.min(a.len())],
        &b[..4.min(b.len())]
    );
    assert_eq!(a.len(), m * n);
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut maxabs = 0.0f32;
    let mut n_finite = 0usize;
    for (x, y) in a.iter().zip(&b) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        n_finite += 1;
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        maxabs = maxabs.max((x - y).abs());
    }
    assert!(
        n_finite > a.len() / 2,
        "too many non-finite pairs {n_finite}/{}",
        a.len()
    );
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    assert!(
        cos > 0.999,
        "large-m GQH GEMM vs fused cos={cos} maxabs={maxabs}"
    );
    println!("rung7c: m={m} n={n} k={k} cos={cos:.6} maxabs={maxabs:.5}");
}

#[test]
fn rung8_embed_row_then_prefill_gqh_qkv() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
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
        1,
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
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
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

fn gather_embed(ordinal: usize, weights: &Qwen38Weights, token: usize) -> GpuBuffer {
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
        matmul_gqh(ordinal, 1, n, k, lhs, weight, qtype, out).unwrap_or_else(|e| {
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
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
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
    dispatch_proj(
        ordinal,
        kv_n,
        hidden,
        &hidden_bf16,
        &full.k_proj_w,
        &mut k_out,
    );
    assert_finite_energy(&bf16_vals(&k_out), "layer3 k_proj");

    let mut v_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_n]).expect("v out");
    dispatch_proj(
        ordinal,
        kv_n,
        hidden,
        &hidden_bf16,
        &full.v_proj_w,
        &mut v_out,
    );
    assert_finite_energy(&bf16_vals(&v_out), "layer3 v_proj");

    let ggml_k = [k_ty, v_ty, a_ty, b_ty].into_iter().any(|ty| {
        matches!(
            ty,
            LOWBIT_GGML_Q8_0 | LOWBIT_GGML_Q4_K | LOWBIT_GGML_Q5_K | LOWBIT_GGML_Q6_K
        )
    });
    println!("rung9: ggml-K among k/v/a/b = {ggml_k}");
}

#[test]
fn rung10_linear_layer0_norm_projs_mlp() {
    let Some(path) = gguf_path() else {
        return;
    };
    let Some(model_dir) = qwen38_model_dir() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let config = load_text_config(&model_dir).expect("hf config");
    let weights = Qwen38Weights::load_gguf(&path, &config, ordinal).expect("load_gguf");
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
    dispatch_proj(
        ordinal,
        inter,
        hidden,
        &post,
        &weights.layers[0].gate_proj_w,
        &mut gate,
    );
    dispatch_proj(
        ordinal,
        inter,
        hidden,
        &post,
        &weights.layers[0].up_proj_w,
        &mut up,
    );
    summarize(&bf16_vals(&gate), "layer0 gate");
    summarize(&bf16_vals(&up), "layer0 up");
    kernel_ffi::prefill_ffi::swiglu_mul(ordinal, ScalarType::BF16, inter, &gate, &up, &mut swiglu)
        .expect("swiglu");
    summarize(&bf16_vals(&swiglu), "layer0 swiglu");
    dispatch_proj(
        ordinal,
        hidden,
        inter,
        &swiglu,
        &weights.layers[0].down_proj_w,
        &mut down,
    );
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

fn gguf_8192_path() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_GQH_8192_GGUF") else {
        return None;
    };
    let path = PathBuf::from(value);
    if path.is_file() {
        Some(path)
    } else if require_gqh_artifacts() {
        panic!(
            "SUPERSONIC_GQH_8192_GGUF points to a missing artifact: {}",
            path.display()
        );
    } else {
        None
    }
}

#[test]
fn mix105_onehot_matches_cpu_decode() {
    let Some(path) = gguf_8192_path() else {
        return;
    };
    let Some(ordinal) = require_hip() else {
        return;
    };
    let file = GgufFile::open(&path).expect("open 8192");
    let name = "blk.21.ssm_out.weight";
    let t = file.tensor(name).expect("ssm_out");
    assert_eq!(t.tensor_type, 105);
    let cols = t.dims[0];
    let rows = 4usize;
    let header = file.mix_header(name).expect("dmix2").clone();
    assert_eq!(header.mode, 1);
    let packed_all = file.tensor_bytes(name).expect("bytes");
    let row_b = model_store::dmix2::row_bytes(105, cols).expect("row");
    let packed = &packed_all[..row_b * rows];
    let mut want = vec![0.0f32; rows];
    let mut x = vec![0.0f32; cols];
    x[0] = 1.0;
    x[17] = 1.0;
    x[100] = 1.0;
    for r in 0..rows {
        let mut wrow = vec![0.0f32; cols];
        model_store::dmix2::decode_row(
            105,
            &packed[r * row_b..(r + 1) * row_b],
            cols,
            &header,
            &mut wrow,
        )
        .expect("cpu decode");
        want[r] = wrow[0] + wrow[17] + wrow[100];
    }
    let x_buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[cols], &{
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
    })
    .expect("x");
    let w_buf =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[rows, row_b], packed).expect("w");
    kernel_ffi::gqh::register_mix(w_buf.as_ptr(), 105, header.mode, header.lut);
    let mut y = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("y");
    kernel_ffi::gqh::mix_matvec(
        ordinal,
        105,
        &w_buf,
        &x_buf,
        &mut y,
        cols,
        rows,
        1,
        false,
        header.mode,
        &header.lut,
    )
    .expect("hip mix");
    let got_bytes = y.to_host_bytes().expect("d2h");
    let got: Vec<f32> = got_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let den = w.abs().max(1e-6);
        assert!((g - w).abs() / den < 1e-5, "mix105 [{i}] got {g} want {w}");
    }

    // Full-row ones vector: stresses the warp-tree vs sequential fold.
    let x1 = vec![1.0f32; cols];
    let mut want1 = vec![0.0f32; rows];
    for r in 0..rows {
        let mut wrow = vec![0.0f32; cols];
        model_store::dmix2::decode_row(
            105,
            &packed[r * row_b..(r + 1) * row_b],
            cols,
            &header,
            &mut wrow,
        )
        .expect("cpu decode ones");
        want1[r] = wrow.iter().sum();
    }
    let x1_buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[cols], &{
        x1.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
    })
    .expect("x1");
    let mut y1 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("y1");
    kernel_ffi::gqh::mix_matvec(
        ordinal,
        105,
        &w_buf,
        &x1_buf,
        &mut y1,
        cols,
        rows,
        1,
        false,
        header.mode,
        &header.lut,
    )
    .expect("hip mix ones");
    let got1: Vec<f32> = y1
        .to_host_bytes()
        .expect("d2h ones")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (g, w)) in got1.iter().zip(&want1).enumerate() {
        let den = w.abs().max(1.0);
        assert!(
            (g - w).abs() / den < 1e-4,
            "mix105 ones [{i}] got {g} want {w} rel {}",
            (g - w).abs() / den
        );
    }
}
