/// Squeeze dimension 1 from a depthwise conv1d weight.
/// Shape [C_out, 1, K] becomes [C_out, K]. Bytes are unchanged.
pub fn squeeze_dim1(bytes: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    assert_eq!(shape.len(), 3, "squeeze_dim1: expected 3D shape, got {shape:?}");
    assert_eq!(shape[1], 1, "squeeze_dim1: middle dim must be 1, got {}", shape[1]);
    (bytes.to_vec(), vec![shape[0], shape[2]])
}

/// Reshape a 1D bias from [H] to [1, 1, H]. Bytes are unchanged.
pub fn head_bias_reshape(bytes: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    assert_eq!(shape.len(), 1, "head_bias_reshape: expected 1D shape, got {shape:?}");
    (bytes.to_vec(), vec![1, 1, shape[0]])
}

/// Transform A_log (F32) to exp(A_log) (BF16) and reshape [H] to [1, 1, H].
pub fn a_log_to_exp_bf16(bytes: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    assert_eq!(shape.len(), 1, "a_log_to_exp_bf16: expected 1D shape, got {shape:?}");
    let out: Vec<u8> = bytes
        .chunks_exact(4)
        .flat_map(|b| {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            half::bf16::from_f32(v.exp()).to_le_bytes()
        })
        .collect();
    (out, vec![1, 1, shape[0]])
}
