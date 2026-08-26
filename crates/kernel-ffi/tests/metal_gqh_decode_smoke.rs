//! Metal GQH decode and dequant+GEMM correctness against CPU reference.
//!
//! ```bash
//! export SUPERSONIC_GQH_GGUF=/path/to/Qwen3.8-27B-GQH-Q3KXL.gguf
//! SUPERSONIC_BACKEND=metal cargo test -p kernel-ffi --test metal_gqh_decode_smoke -- --nocapture
//! ```

#![cfg(supersonic_backend_metal)]

use std::path::PathBuf;

use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::gqh::{self, RUNG_GQH3, RUNG_GQH4};
use kernel_ffi::prefill_ffi;
use model_store::gguf::GgufFile;
use model_store::gqh::GqhRung;

const TENSORS: &[(&str, i32)] = &[
    ("blk.0.ffn_gate.weight", RUNG_GQH3),
    ("blk.21.ssm_out.weight", RUNG_GQH4),
];

fn gguf_path() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_GQH_GGUF") else {
        eprintln!("skip: SUPERSONIC_GQH_GGUF is not configured");
        return None;
    };
    let path = PathBuf::from(value);
    path.is_file().then_some(path)
}

fn read_f32(buf: &GpuBuffer) -> Vec<f32> {
    buf.to_host_bytes()
        .expect("d2h")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_bf16(buf: &GpuBuffer) -> Vec<f32> {
    buf.to_host_bytes()
        .expect("d2h")
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

fn cpu_matmul_rhs_transposed(w: &[f32], x: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w[col * k + kk];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn assert_decode_matches_cpu(
    ordinal: usize,
    file: &GgufFile,
    tensor_name: &str,
    rung: i32,
    rows: usize,
) {
    let tensor = file.tensor(tensor_name).expect("tensor");
    let cpu_rung = GqhRung::from_ggml_type(tensor.tensor_type).expect("gqh");
    let cols = tensor.dims[0];
    let packed_all = file.tensor_bytes(tensor_name).expect("bytes");
    let row_bytes = packed_all.len() / tensor.dims[1];
    let header = file.gqh_header(tensor_name).cloned().expect("header");
    let packed = &packed_all[..row_bytes * rows];

    let mut cpu = vec![0.0f32; rows * cols];
    for row in 0..rows {
        model_store::gqh::decode_row(
            cpu_rung,
            &packed[row * row_bytes..(row + 1) * row_bytes],
            cols,
            Some(header.clone()),
            &mut cpu[row * cols..(row + 1) * cols],
        )
        .expect("cpu decode");
    }

    let device_wire =
        model_store::gqh::planarize(cpu_rung, rows, cols, packed).expect("planarize");
    let wire = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device_wire.len()], &device_wire)
        .expect("upload wire");

    let mut dst_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows * cols]).expect("dst f32");
    gqh::decode(
        ordinal,
        rung,
        &wire,
        header.tensor_scale,
        header.grid_code,
        &mut dst_f32,
        rows,
        cols,
    )
    .expect("metal decode f32");
    let got_f32 = read_f32(&dst_f32);
    let mut max_abs = 0.0f32;
    for (g, w) in got_f32.iter().zip(&cpu) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(max_abs < 1e-4, "decode f32 max_abs={max_abs} on {tensor_name}");

    let mut dst_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[rows * cols]).expect("dst bf16");
    gqh::decode(
        ordinal,
        rung,
        &wire,
        header.tensor_scale,
        header.grid_code,
        &mut dst_bf16,
        rows,
        cols,
    )
    .expect("metal decode bf16");
    let got_bf16 = read_bf16(&dst_bf16);
    for (g, w) in got_bf16.iter().zip(&cpu) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(max_abs < 0.05, "decode bf16 max_abs={max_abs} on {tensor_name}");
    eprintln!("metal_gqh_decode ok: {tensor_name} max_abs={max_abs:.6}");
}

fn assert_dequant_gemm_matches_cpu(
    ordinal: usize,
    file: &GgufFile,
    tensor_name: &str,
    rung: i32,
    m: usize,
    n: usize,
) {
    let tensor = file.tensor(tensor_name).expect("tensor");
    let cpu_rung = GqhRung::from_ggml_type(tensor.tensor_type).expect("gqh");
    let k = tensor.dims[0];
    let packed_all = file.tensor_bytes(tensor_name).expect("bytes");
    let row_bytes = packed_all.len() / tensor.dims[1];
    let header = file.gqh_header(tensor_name).cloned().expect("header");
    let packed = &packed_all[..row_bytes * n];

    let mut w_cpu = vec![0.0f32; n * k];
    for row in 0..n {
        model_store::gqh::decode_row(
            cpu_rung,
            &packed[row * row_bytes..(row + 1) * row_bytes],
            k,
            Some(header.clone()),
            &mut w_cpu[row * k..(row + 1) * k],
        )
        .expect("cpu decode row");
    }

    let x_vals: Vec<f32> = (0..m * k)
        .map(|i| (((i % 17) as f32) - 8.0) / 32.0)
        .collect();
    let want = cpu_matmul_rhs_transposed(&w_cpu, &x_vals, m, n, k);

    let device_wire =
        model_store::gqh::planarize(cpu_rung, n, k, packed).expect("planarize");
    let wire = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device_wire.len()], &device_wire)
        .expect("upload wire");
    gqh::register_header(
        ordinal,
        wire.as_ptr(),
        header.tensor_scale,
        header.grid_code,
    );

    let lhs = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &bf16_bytes(&x_vals))
        .expect("upload lhs");
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n]).expect("alloc out");

    gqh::dequant_gemm_bf16(
        ordinal,
        rung,
        &wire,
        header.tensor_scale,
        header.grid_code,
        &lhs,
        &mut out,
        m,
        n,
        k,
    )
    .expect("metal dequant_gemm_bf16");

    let got = read_bf16(&out);
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 0.05,
        "dequant_gemm_bf16 max_abs={max_abs} on {tensor_name}"
    );

    let mut out_matmul = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n]).expect("alloc matmul");
    prefill_ffi::matmul_rhs_transposed_gqh(
        ordinal,
        m,
        n,
        k,
        &lhs,
        &wire,
        header.tensor_scale,
        header.grid_code,
        rung,
        &mut out_matmul,
    )
    .expect("matmul_rhs_transposed_gqh");
    let matmul = read_bf16(&out_matmul);
    let mut matmul_delta = 0.0f32;
    for (a, b) in got.iter().zip(&matmul) {
        matmul_delta = matmul_delta.max((a - b).abs());
    }
    assert!(
        matmul_delta < 0.05,
        "dequant_gemm vs matmul_rhs_transposed_gqh delta={matmul_delta} on {tensor_name}"
    );
    eprintln!(
        "metal_gqh_dequant_gemm ok: {tensor_name} max_abs={max_abs:.6} matmul_delta={matmul_delta:.6}"
    );
}

#[test]
fn metal_gqh_decode_and_dequant_gemm_match_cpu() {
    let Some(path) = gguf_path() else {
        return;
    };
    gpu_hal::set_device(0).expect("set device");
    let ordinal = 0usize;
    let file = GgufFile::open(&path).expect("open gguf");
    for (tensor_name, rung) in TENSORS {
        assert_decode_matches_cpu(ordinal, &file, tensor_name, *rung, 4);
        assert_dequant_gemm_matches_cpu(ordinal, &file, tensor_name, *rung, 9, 4);
    }
}
