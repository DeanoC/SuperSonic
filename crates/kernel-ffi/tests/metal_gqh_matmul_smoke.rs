//! Metal GQH matmul correctness against CPU-dequant reference on Q3KXL tensors.
//!
//! ```bash
//! export SUPERSONIC_GQH_GGUF=/path/to/Qwen3.8-27B-GQH-Q3KXL.gguf
//! SUPERSONIC_BACKEND=metal cargo test -p kernel-ffi --test metal_gqh_matmul_smoke -- --nocapture
//! ```

#![cfg(supersonic_backend_metal)]

use std::path::PathBuf;

use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::gqh;
use kernel_ffi::prefill_ffi;
use model_store::gguf::GgufFile;
use model_store::gqh::GqhRung;

const GQH_QTYPES: &[(u32, &str)] = &[
    (108, "GQH3"),
    (109, "GQH2_H"),
    (110, "GQH2_C"),
    (111, "GQH4"),
];

/// Anchor tensors exercised in the HIP crawl; kept as regression pins.
const ANCHOR_TENSORS: &[&str] = &[
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.21.ssm_out.weight",
];

fn gguf_path() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_GQH_GGUF") else {
        eprintln!("skip: SUPERSONIC_GQH_GGUF is not configured");
        return None;
    };
    let path = PathBuf::from(value);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skip: SUPERSONIC_GQH_GGUF points to a missing file: {}",
            path.display()
        );
        None
    }
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

fn read_bf16(buf: &GpuBuffer) -> Vec<f32> {
    buf.to_host_bytes()
        .expect("d2h")
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
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

fn assert_matmul_matches_cpu(
    ordinal: usize,
    file: &GgufFile,
    tensor_name: &str,
    m: usize,
    n: usize,
) {
    let tensor = file
        .tensor(tensor_name)
        .unwrap_or_else(|| panic!("missing {tensor_name}"));
    let rung = GqhRung::from_ggml_type(tensor.tensor_type).expect("gqh tensor");
    let hip_rung = gqh::rung_from_ggml_type(tensor.tensor_type as u32).expect("gqh rung");

    let k = tensor.dims[0];
    let packed_all = file.tensor_bytes(tensor_name).expect("tensor bytes");
    let row_bytes = packed_all.len() / tensor.dims[1];
    let header = file
        .gqh_header(tensor_name)
        .cloned()
        .expect("gqh header");

    let mut w_cpu = vec![0.0f32; n * k];
    for row in 0..n {
        model_store::gqh::decode_row(
            rung,
            &packed_all[row * row_bytes..(row + 1) * row_bytes],
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

    let device_wire = model_store::gqh::planarize(rung, n, k, &packed_all[..row_bytes * n])
        .expect("planarize wire subset");
    let wire = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device_wire.len()], &device_wire)
        .expect("upload wire");
    gqh::register_header(
        ordinal,
        wire.as_ptr(),
        header.tensor_scale,
        header.grid_code,
    );

    let x = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &bf16_bytes(&x_vals))
        .expect("upload x");
    let mut out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n]).expect("alloc out");

    prefill_ffi::matmul_rhs_transposed_gqh(
        ordinal,
        m,
        n,
        k,
        &x,
        &wire,
        header.tensor_scale,
        header.grid_code,
        hip_rung,
        &mut out,
    )
    .unwrap_or_else(|e| panic!("metal gqh matmul on {tensor_name}: {e}"));

    let got = read_bf16(&out);
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0f32;
    let mut n_compared = 0usize;
    for (g, w) in got.iter().zip(&want) {
        if !g.is_finite() || !w.is_finite() {
            continue;
        }
        n_compared += 1;
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(n_compared > 0, "no finite outputs to compare on {tensor_name}");
    assert!(
        max_abs < 0.05,
        "metal gqh matmul max_abs={max_abs} on {tensor_name} (m={m}, n={n}, k={k}, rung={hip_rung})"
    );
    eprintln!("metal_gqh_matmul ok: {tensor_name} max_abs={max_abs:.6}");
}

fn discover_gqh_tensor_per_rung(file: &GgufFile) -> Vec<(String, u32)> {
    let mut found: Vec<Option<(String, u32)>> = vec![None; GQH_QTYPES.len()];
    for name in file.tensor_names() {
        let tensor = file.tensor(name).expect("tensor metadata");
        let Some(slot) = GQH_QTYPES
            .iter()
            .position(|(qtype, _)| *qtype == tensor.tensor_type)
        else {
            continue;
        };
        if found[slot].is_some() {
            continue;
        }
        if tensor.dims.len() != 2 {
            continue;
        }
        let k = tensor.dims[0];
        let n = tensor.dims[1];
        if k < 256 || n < 4 {
            continue;
        }
        if file.gqh_header(name).is_none() {
            continue;
        }
        found[slot] = Some((name.to_owned(), tensor.tensor_type));
    }
    found.into_iter().flatten().collect()
}

fn collect_tensor_cases(file: &GgufFile) -> Vec<(String, usize, usize)> {
    let discovered = discover_gqh_tensor_per_rung(file);
    assert!(
        discovered.iter().any(|(_, qtype)| *qtype == 108),
        "Q3KXL must expose at least one GQH3 tensor with a header"
    );
    assert!(
        discovered.iter().any(|(_, qtype)| *qtype == 111),
        "Q3KXL must expose at least one GQH4 tensor with a header"
    );

    let mut cases = Vec::new();
    let mut seen = std::collections::BTreeSet::new();

    for name in ANCHOR_TENSORS {
        seen.insert((*name).to_owned());
        cases.push(((*name).to_owned(), 4, 4));
    }

    for (name, qtype) in discovered {
        if seen.insert(name.clone()) {
            let n = file.tensor(&name).expect("tensor").dims[1].min(8);
            cases.push((name, 1, n));
            eprintln!("metal_gqh_matmul discovered rung qtype={qtype}");
        }
    }

    cases
}

#[test]
fn metal_gqh_matmul_matches_cpu_on_q3kxl_tensors() {
    let Some(path) = gguf_path() else {
        return;
    };
    gpu_hal::set_device(0).expect("set device");
    let ordinal = 0usize;
    let file = GgufFile::open(&path).expect("open gguf");
    for (tensor_name, m, n) in collect_tensor_cases(&file) {
        assert_matmul_matches_cpu(ordinal, &file, &tensor_name, m, n);
    }
}
