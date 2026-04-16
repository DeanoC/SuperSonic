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

/// Quantize BF16 weight to INT4 with asymmetric group quantization.
/// Returns (packed_bytes, scale_bytes, zero_bytes, packed_shape).
/// packed_bytes: [rows, cols/2] uint8 — 2 INT4 values per byte (low nibble = even col).
/// scale_bytes: [rows/group_size, cols/group_size] BF16 — per-group scale.
/// zero_bytes: [rows/group_size, cols/group_size] BF16 — per-group zero point.
/// Quantization: int4_val = clamp(round((bf16_val - min) / scale), 0, 15)
///               dequant: bf16_val ≈ (int4_val - zero) * scale
pub fn bf16_to_int4(
    bf16_bytes: &[u8],
    shape: &[usize],
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<usize>) {
    assert_eq!(shape.len(), 2, "bf16_to_int4: expected 2D shape, got {shape:?}");
    let rows = shape[0];
    let cols = shape[1];
    assert_eq!(bf16_bytes.len(), rows * cols * 2, "bf16_to_int4: byte count mismatch");
    assert!(cols % 2 == 0, "bf16_to_int4: cols must be even for nibble packing");

    let scale_rows = (rows + group_size - 1) / group_size;
    let scale_cols = (cols + group_size - 1) / group_size;

    let mut packed = vec![0u8; rows * (cols / 2)];
    let mut scales = vec![0u8; scale_rows * scale_cols * 2];
    let mut zeros = vec![0u8; scale_rows * scale_cols * 2];

    // Parse BF16 weights into F32 for processing
    let weights: Vec<f32> = bf16_bytes
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    // Pass 1: compute per-group min/max and scale/zero
    for gr in 0..scale_rows {
        for gc in 0..scale_cols {
            let r_start = gr * group_size;
            let r_end = (r_start + group_size).min(rows);
            let c_start = gc * group_size;
            let c_end = (c_start + group_size).min(cols);

            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for r in r_start..r_end {
                for c in c_start..c_end {
                    let v = weights[r * cols + c];
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
            }

            // Asymmetric quantization to [0, 15]
            let range = max_val - min_val;
            let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
            // zero_point is the float value that maps to int4=0
            // dequant: (int4_val - zero_f) * scale  where zero_f = -min_val / scale
            let zero_f = if range > 0.0 { -min_val / scale } else { 0.0 };

            let si = (gr * scale_cols + gc) * 2;
            scales[si..si + 2].copy_from_slice(&half::bf16::from_f32(scale).to_le_bytes());
            zeros[si..si + 2].copy_from_slice(&half::bf16::from_f32(zero_f).to_le_bytes());
        }
    }

    // Pass 2: quantize weights to INT4 and pack
    for r in 0..rows {
        for c in (0..cols).step_by(2) {
            let gr = r / group_size;
            let gc0 = c / group_size;
            let gc1 = (c + 1) / group_size;

            let si0 = gr * scale_cols + gc0;
            let scale0 = half::bf16::from_le_bytes([scales[si0 * 2], scales[si0 * 2 + 1]]).to_f32();
            let zero0 = half::bf16::from_le_bytes([zeros[si0 * 2], zeros[si0 * 2 + 1]]).to_f32();

            let si1 = gr * scale_cols + gc1;
            let scale1 = half::bf16::from_le_bytes([scales[si1 * 2], scales[si1 * 2 + 1]]).to_f32();
            let zero1 = half::bf16::from_le_bytes([zeros[si1 * 2], zeros[si1 * 2 + 1]]).to_f32();

            let v0 = weights[r * cols + c];
            let v1 = weights[r * cols + c + 1];

            let q0 = ((v0 / scale0 + zero0).round() as i32).clamp(0, 15) as u8;
            let q1 = ((v1 / scale1 + zero1).round() as i32).clamp(0, 15) as u8;

            // Pack: low nibble = even col (c), high nibble = odd col (c+1)
            packed[r * (cols / 2) + c / 2] = q0 | (q1 << 4);
        }
    }

    let packed_shape = vec![rows, cols / 2];
    (packed, scales, zeros, packed_shape)
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
