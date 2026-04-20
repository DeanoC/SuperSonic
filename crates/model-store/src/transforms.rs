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

/// Split a fused qkv_proj weight into (q, k, v) slices along row-major dim 0.
///
/// Phi-3 / Phi-4 stores `[q_rows + k_rows + v_rows, hidden]` where
/// - q_rows = num_attention_heads * head_dim
/// - k_rows = v_rows = num_key_value_heads * head_dim
///
/// Bytes are contiguous in row-major order (Q first, then K, then V). This
/// helper slices them into three owned `Vec<u8>` with accompanying shapes.
pub fn split_qkv_proj(
    bytes: &[u8],
    shape: &[usize],
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
    dtype_bytes: usize,
) -> (Vec<u8>, Vec<usize>, Vec<u8>, Vec<usize>, Vec<u8>, Vec<usize>) {
    assert_eq!(shape.len(), 2, "split_qkv_proj: expected 2D, got {shape:?}");
    assert_eq!(
        shape[0],
        q_rows + k_rows + v_rows,
        "split_qkv_proj: dim0 {} != q {} + k {} + v {}",
        shape[0],
        q_rows,
        k_rows,
        v_rows
    );
    let cols = shape[1];
    let row_stride = cols * dtype_bytes;
    assert_eq!(
        bytes.len(),
        shape[0] * row_stride,
        "split_qkv_proj: byte count mismatch"
    );
    let q_end = q_rows * row_stride;
    let k_end = q_end + k_rows * row_stride;
    let v_end = k_end + v_rows * row_stride;
    (
        bytes[..q_end].to_vec(),
        vec![q_rows, cols],
        bytes[q_end..k_end].to_vec(),
        vec![k_rows, cols],
        bytes[k_end..v_end].to_vec(),
        vec![v_rows, cols],
    )
}

/// Split a fused `gate_up_proj` weight into (gate, up) slices along row-major dim 0.
///
/// Phi-3 / Phi-4 stores `[2 * intermediate_size, hidden]` with gate first, then up.
pub fn split_gate_up_proj(
    bytes: &[u8],
    shape: &[usize],
    intermediate_size: usize,
    dtype_bytes: usize,
) -> (Vec<u8>, Vec<usize>, Vec<u8>, Vec<usize>) {
    assert_eq!(shape.len(), 2, "split_gate_up_proj: expected 2D, got {shape:?}");
    assert_eq!(
        shape[0],
        2 * intermediate_size,
        "split_gate_up_proj: dim0 {} != 2 * intermediate_size {}",
        shape[0],
        intermediate_size
    );
    let cols = shape[1];
    let row_stride = cols * dtype_bytes;
    assert_eq!(
        bytes.len(),
        shape[0] * row_stride,
        "split_gate_up_proj: byte count mismatch"
    );
    let mid = intermediate_size * row_stride;
    (
        bytes[..mid].to_vec(),
        vec![intermediate_size, cols],
        bytes[mid..].to_vec(),
        vec![intermediate_size, cols],
    )
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn qkv_split_preserves_row_order() {
        // Simulate Phi-4-mini qkv_proj: q_rows=3072, k_rows=v_rows=1024, hidden=3072, dtype=bf16.
        // Use small sizes: q=6, k=v=2, hidden=4, bf16 (2 bytes each) → rows 0..10, each row 8 bytes.
        let q_rows = 6;
        let k_rows = 2;
        let v_rows = 2;
        let hidden = 4;
        let dt_bytes = 2;
        let total_rows = q_rows + k_rows + v_rows;
        let mut bytes = Vec::with_capacity(total_rows * hidden * dt_bytes);
        for r in 0..total_rows {
            for c in 0..hidden {
                // Encode (r,c) as two-byte value: hi=r, lo=c, so we can validate ordering.
                bytes.push(r as u8);
                bytes.push(c as u8);
            }
        }
        let shape = vec![total_rows, hidden];
        let (qb, qs, kb, ks, vb, vs) =
            split_qkv_proj(&bytes, &shape, q_rows, k_rows, v_rows, dt_bytes);
        assert_eq!(qs, vec![q_rows, hidden]);
        assert_eq!(ks, vec![k_rows, hidden]);
        assert_eq!(vs, vec![v_rows, hidden]);
        assert_eq!(qb.len(), q_rows * hidden * dt_bytes);
        assert_eq!(kb.len(), k_rows * hidden * dt_bytes);
        assert_eq!(vb.len(), v_rows * hidden * dt_bytes);
        // First row of Q is (r=0, c=0..hidden).
        assert_eq!(qb[0], 0);
        assert_eq!(qb[1], 0);
        assert_eq!(qb[2], 0);
        assert_eq!(qb[3], 1);
        // First row of K is row index q_rows=6.
        assert_eq!(kb[0], 6);
        // First row of V is row index q_rows + k_rows = 8.
        assert_eq!(vb[0], 8);
    }

    #[test]
    fn gate_up_split_preserves_row_order() {
        let intermediate = 4;
        let hidden = 3;
        let dt_bytes = 2;
        let total_rows = 2 * intermediate;
        let mut bytes = Vec::with_capacity(total_rows * hidden * dt_bytes);
        for r in 0..total_rows {
            for c in 0..hidden {
                bytes.push(r as u8);
                bytes.push(c as u8);
            }
        }
        let shape = vec![total_rows, hidden];
        let (gb, gs, ub, us) = split_gate_up_proj(&bytes, &shape, intermediate, dt_bytes);
        assert_eq!(gs, vec![intermediate, hidden]);
        assert_eq!(us, vec![intermediate, hidden]);
        assert_eq!(gb[0], 0, "gate starts at row 0");
        assert_eq!(ub[0], intermediate as u8, "up starts at row intermediate");
    }

    #[test]
    #[should_panic(expected = "dim0")]
    fn qkv_split_rejects_wrong_total_rows() {
        let bytes = vec![0u8; 4 * 2 * 2];
        let shape = vec![4, 2];
        split_qkv_proj(&bytes, &shape, 2, 2, 2, 2); // q+k+v = 6 != 4
    }
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
