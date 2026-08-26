//! Metal GQH fused matvec correctness against CPU-dequant reference.
//!
//! ```bash
//! export SUPERSONIC_GQH_GGUF=/path/to/Qwen3.8-27B-GQH-Q3KXL.gguf
//! SUPERSONIC_BACKEND=metal cargo test -p kernel-ffi --test metal_gqh_matvec_smoke -- --nocapture
//! ```

#![cfg(supersonic_backend_metal)]

use std::path::PathBuf;

use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::gqh::{self, RUNG_GQH3, RUNG_GQH4};
use model_store::gguf::GgufFile;
use model_store::gqh::GqhRung;

const TENSORS: &[(&str, i32)] = &[
    ("blk.0.ffn_gate.weight", RUNG_GQH3),
    ("blk.0.ffn_up.weight", RUNG_GQH3),
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

fn kernel_order_matvec(weights: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += weights[row * cols + col] * x[col];
        }
        out[row] = acc;
    }
    out
}

fn assert_matvec_matches_cpu(
    ordinal: usize,
    file: &GgufFile,
    tensor_name: &str,
    rung: i32,
    rows: usize,
) {
    let tensor = file
        .tensor(tensor_name)
        .unwrap_or_else(|| panic!("missing {tensor_name}"));
    let cpu_rung =
        GqhRung::from_ggml_type(tensor.tensor_type).expect("gqh tensor");
    assert_eq!(cpu_rung.ggml_type() as u32, gqh::ggml_type_from_rung(rung).unwrap() as u32);

    let cols = tensor.dims[0];
    let packed_all = file.tensor_bytes(tensor_name).expect("tensor bytes");
    let row_bytes = packed_all.len() / tensor.dims[1];
    let header = file
        .gqh_header(tensor_name)
        .cloned()
        .expect("gqh header");
    let packed = &packed_all[..row_bytes * rows];

    let mut cpu_w = vec![0.0f32; rows * cols];
    for row in 0..rows {
        model_store::gqh::decode_row(
            cpu_rung,
            &packed[row * row_bytes..(row + 1) * row_bytes],
            cols,
            Some(header.clone()),
            &mut cpu_w[row * cols..(row + 1) * cols],
        )
        .expect("cpu decode row");
    }

    let x_vals: Vec<f32> = (0..cols)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .collect();
    let want = kernel_order_matvec(&cpu_w, &x_vals, rows, cols);

    let device_wire =
        model_store::gqh::planarize(cpu_rung, rows, cols, packed).expect("planarize");
    let wire = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device_wire.len()], &device_wire)
        .expect("upload wire");
    gqh::register_header(
        ordinal,
        wire.as_ptr(),
        header.tensor_scale,
        header.grid_code,
    );

    let x = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::F32,
        &[cols],
        &x_vals
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("upload x");
    let mut y = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("alloc y");

    gqh::matvec(
        ordinal,
        rung,
        &wire,
        &x,
        &mut y,
        cols,
        rows,
        1,
        cols,
        rows,
        header.tensor_scale,
        header.grid_code,
    )
    .unwrap_or_else(|e| panic!("metal gqh matvec on {tensor_name}: {e}"));

    let got = y
        .to_host_bytes()
        .expect("d2h")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();

    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        assert!(g.is_finite() && w.is_finite());
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 0.05,
        "metal gqh matvec max_abs={max_abs} on {tensor_name} (rows={rows}, cols={cols})"
    );
    eprintln!("metal_gqh_matvec ok: {tensor_name} max_abs={max_abs:.6}");
}

#[test]
fn metal_gqh_matvec_matches_cpu_on_q3kxl_tensors() {
    let Some(path) = gguf_path() else {
        return;
    };
    gpu_hal::set_device(0).expect("set device");
    let ordinal = 0usize;
    let file = GgufFile::open(&path).expect("open gguf");
    for (tensor_name, rung) in TENSORS {
        assert_matvec_matches_cpu(ordinal, &file, tensor_name, *rung, 8);
    }
}
