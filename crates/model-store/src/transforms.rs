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

/// Convert FP8 E4M3 byte to F32.
/// E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7, no inf, NaN=0x7F/0xFF
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xF;
    let mantissa = byte & 0x7;
    if byte == 0x7F || byte == 0xFF {
        return f32::NAN;
    }
    let val = if exp == 0 {
        // Subnormal: 2^(-6) * (mantissa / 8)
        f32::from(mantissa) / 8.0 * (2.0f32).powi(-6)
    } else {
        // Normal: 2^(exp-7) * (1 + mantissa/8)
        (1.0 + f32::from(mantissa) / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };
    if sign == 1 { -val } else { val }
}

/// Dequantize FP8 E4M3 weight to BF16 using block-wise scale_inv.
/// weight_shape: [rows, cols], scale_shape: [rows/block, cols/block]
/// block_size: typically 128
pub fn fp8_dequant_to_bf16(
    fp8_bytes: &[u8],
    weight_shape: &[usize],
    scale_inv_bytes: &[u8],
    scale_shape: &[usize],
    block_size: usize,
) -> (Vec<u8>, Vec<usize>) {
    assert_eq!(weight_shape.len(), 2, "fp8_dequant: expected 2D weight, got {weight_shape:?}");
    assert_eq!(scale_shape.len(), 2, "fp8_dequant: expected 2D scale, got {scale_shape:?}");
    let rows = weight_shape[0];
    let cols = weight_shape[1];
    let scale_rows = scale_shape[0];
    let scale_cols = scale_shape[1];
    assert_eq!(fp8_bytes.len(), rows * cols, "fp8_dequant: byte count mismatch");
    assert_eq!(scale_inv_bytes.len(), scale_rows * scale_cols * 2, "fp8_dequant: scale byte count mismatch");

    let mut out = Vec::with_capacity(rows * cols * 2); // BF16 = 2 bytes each
    for r in 0..rows {
        for c in 0..cols {
            let fp8_val = fp8_e4m3_to_f32(fp8_bytes[r * cols + c]);
            let sr = r / block_size;
            let sc = c / block_size;
            let scale_idx = sr * scale_cols + sc;
            let scale_bytes = &scale_inv_bytes[scale_idx * 2..scale_idx * 2 + 2];
            let scale = half::bf16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
            let dequant = fp8_val * scale;
            out.extend_from_slice(&half::bf16::from_f32(dequant).to_le_bytes());
        }
    }
    (out, weight_shape.to_vec())
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
