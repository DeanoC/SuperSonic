#![cfg(supersonic_backend_metal)]

use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi;

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

#[test]
fn metal_prefill_embedding_lookup_smoke() {
    gpu_hal::set_device(0).expect("set device");

    let hidden = 4usize;
    let vocab = 2usize;
    let tokens = 2usize;
    let embeddings = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[vocab, hidden],
        &bf16_bytes(&[
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ]),
    )
    .expect("embeddings");
    let indexes = GpuBuffer::from_host_bytes(
        0,
        ScalarType::U32,
        &[tokens],
        &[0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8],
    )
    .expect("indexes");
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[tokens, hidden]).expect("out");

    prefill_ffi::embedding_lookup(
        0,
        ScalarType::BF16,
        tokens,
        vocab,
        hidden,
        &embeddings,
        &indexes,
        &mut out,
    )
    .expect("embedding_lookup");

    let got = out.to_host_bytes().expect("out bytes");
    let expected = bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(got, expected);
}

#[test]
fn metal_prefill_rms_norm_smoke() {
    gpu_hal::set_device(0).expect("set device");

    let cols = 4usize;
    let rows = 1usize;
    let xs = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[rows, cols],
        &bf16_bytes(&[3.0, 4.0, 0.0, 0.0]),
    )
    .expect("xs");
    let weight = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[cols],
        &bf16_bytes(&[1.0, 1.0, 1.0, 1.0]),
    )
    .expect("weight");
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[rows, cols]).expect("out");

    prefill_ffi::rms_norm_rows(
        0,
        ScalarType::BF16,
        rows,
        cols,
        1e-6,
        &xs,
        &weight,
        &mut out,
    )
    .expect("rms_norm_rows");

    let got: Vec<f32> = out
        .to_host_bytes()
        .expect("out bytes")
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    assert_eq!(got.len(), 4);
    for value in &got {
        assert!(value.is_finite());
    }
    assert!((got[0] - 2.40625).abs() < 0.01);
    assert!((got[1] - 3.203125).abs() < 0.01);
    assert_eq!(got[2], 0.0);
    assert_eq!(got[3], 0.0);
}

#[test]
fn metal_prefill_matmul_smoke() {
    gpu_hal::set_device(0).expect("set device");

    let m = 2usize;
    let n = 2usize;
    let k = 2usize;
    let lhs = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[m, k],
        &bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
    )
    .expect("lhs");
    let rhs = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[n, k],
        &bf16_bytes(&[5.0, 6.0, 7.0, 8.0]),
    )
    .expect("rhs");
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[m, n]).expect("out");

    prefill_ffi::matmul_rhs_transposed(
        0,
        ScalarType::BF16,
        1,
        m,
        n,
        k,
        &lhs,
        &rhs,
        &mut out,
    )
    .expect("matmul_rhs_transposed");

    let got: Vec<f32> = out
        .to_host_bytes()
        .expect("out bytes")
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    assert_eq!(got.len(), 4);
    assert!((got[0] - 17.0).abs() < 0.5);
    assert!((got[1] - 23.0).abs() < 0.5);
    assert!((got[2] - 39.0).abs() < 0.5);
    assert!((got[3] - 53.0).abs() < 0.5);
}

#[test]
fn metal_prefill_cast_smoke() {
    gpu_hal::set_device(0).expect("set device");

    let xs = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[4],
        &bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
    )
    .expect("xs");
    let mut out = GpuBuffer::zeros(0, ScalarType::F32, &[4]).expect("out");

    prefill_ffi::cast(
        0,
        ScalarType::BF16,
        ScalarType::F32,
        4,
        &xs,
        &mut out,
    )
    .expect("cast");

    let got: Vec<f32> = out
        .to_host_bytes()
        .expect("out bytes")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn metal_prefill_element_add_smoke() {
    gpu_hal::set_device(0).expect("set device");

    let lhs = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[4],
        &bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
    )
    .expect("lhs");
    let rhs = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[4],
        &bf16_bytes(&[0.5, 1.5, 2.5, 3.5]),
    )
    .expect("rhs");
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[4]).expect("out");

    prefill_ffi::element_add(0, ScalarType::BF16, 4, &lhs, &rhs, &mut out).expect("element_add");

    let got: Vec<f32> = out
        .to_host_bytes()
        .expect("out bytes")
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    assert!((got[0] - 1.5).abs() < 0.01);
    assert!((got[1] - 3.5).abs() < 0.01);
    assert!((got[2] - 5.5).abs() < 0.01);
    assert!((got[3] - 7.5).abs() < 0.01);
}

#[test]
fn metal_prefill_sigmoid_mul_smoke() {
    gpu_hal::set_device(0).expect("set device");

    let data = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[2],
        &bf16_bytes(&[2.0, 4.0]),
    )
    .expect("data");
    let gate = GpuBuffer::from_host_bytes(
        0,
        ScalarType::BF16,
        &[2],
        &bf16_bytes(&[0.0, 0.0]),
    )
    .expect("gate");
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[2]).expect("out");

    prefill_ffi::sigmoid_mul(0, ScalarType::BF16, 2, &data, &gate, &mut out).expect("sigmoid_mul");

    let got: Vec<f32> = out
        .to_host_bytes()
        .expect("out bytes")
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    assert!((got[0] - 1.0).abs() < 0.05);
    assert!((got[1] - 2.0).abs() < 0.05);
}
