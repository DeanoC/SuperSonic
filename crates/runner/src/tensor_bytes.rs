#![allow(dead_code)]

pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

pub fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        out.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
    }
    out
}

pub fn f32_to_f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}
