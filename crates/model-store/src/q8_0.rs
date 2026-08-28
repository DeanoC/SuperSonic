//! ggml Q8_0 block dequant (CPU).
//!
//! Layout matches `block_q8_0` in ggml-common.h: one fp16 `d` scale, then 32
//! signed int8 quants. 34 bytes per 32 weights. Used by the DFlash2 draft
//! model's Q8_0 weights for the CPU reference forward pass.

use crate::Error;

pub const Q8_0_BLOCK: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub fn row_bytes(cols: usize) -> Result<usize, Error> {
    if cols == 0 || cols % Q8_0_BLOCK != 0 {
        return Err(Error::Other(format!(
            "Q8_0 cols {cols} is not a positive multiple of {Q8_0_BLOCK}"
        )));
    }
    Ok((cols / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES)
}

/// Dequantize one Q8_0-packed row of `cols` weights to f32.
pub fn decode_row(packed: &[u8], cols: usize, out: &mut [f32]) -> Result<(), Error> {
    let want = row_bytes(cols)?;
    if packed.len() != want {
        return Err(Error::Other(format!(
            "Q8_0 packed row is {} B, expected {want} B",
            packed.len()
        )));
    }
    if out.len() != cols {
        return Err(Error::Other(format!(
            "Q8_0 decode output len {} != cols {cols}",
            out.len()
        )));
    }
    let nb = cols / Q8_0_BLOCK;
    for b in 0..nb {
        let off = b * Q8_0_BLOCK_BYTES;
        let d = half::f16::from_le_bytes([packed[off], packed[off + 1]]).to_f32();
        for j in 0..Q8_0_BLOCK {
            let q = i8::from_le_bytes([packed[off + 2 + j]]);
            out[b * Q8_0_BLOCK + j] = d * (q as f32);
        }
    }
    Ok(())
}

/// Dequantize a full weight matrix to a row-major f32 buffer.
///
/// `packed` holds `rows` contiguous Q8_0-packed rows, each `cols` weights
/// wide (`row_bytes(cols)` bytes per row). `out` is `[rows * cols]` in
/// row-major order: `out[row * cols + col]`.
pub fn decode_matrix(
    packed: &[u8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    let rb = row_bytes(cols)?;
    if out.len() < rows * cols {
        return Err(Error::Other(format!(
            "Q8_0 decode_matrix out len {} < {}*{}",
            out.len(),
            rows,
            cols
        )));
    }
    for r in 0..rows {
        decode_row(
            &packed[r * rb..(r + 1) * rb],
            cols,
            &mut out[r * cols..(r + 1) * cols],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequants_constant_q8_0_block() {
        let mut block = [0u8; Q8_0_BLOCK_BYTES];
        // d = 1.0 (f16)
        let d = half::f16::from_f32(1.0);
        block[0] = d.to_le_bytes()[0];
        block[1] = d.to_le_bytes()[1];
        for j in 0..Q8_0_BLOCK {
            block[2 + j] = (j as i8) as u8;
        }
        let mut out = vec![0.0f32; Q8_0_BLOCK];
        decode_row(&block, Q8_0_BLOCK, &mut out).expect("decode");
        for j in 0..Q8_0_BLOCK {
            assert_eq!(out[j], j as f32, "q8_0 block element {j}");
        }
    }

    #[test]
    fn rejects_non_multiple_cols() {
        let mut out = vec![0.0f32; 33];
        let err = decode_row(&[0u8; 70], 33, &mut out).unwrap_err();
        assert!(err.to_string().contains("not a positive multiple of 32"));
    }

    #[test]
    fn decode_matrix_matches_row_by_row() {
        let cols = Q8_0_BLOCK;
        let rows = 3;
        let rb = Q8_0_BLOCK_BYTES;
        let mut packed = vec![0u8; rows * rb];
        let d = half::f16::from_f32(0.5);
        for r in 0..rows {
            packed[r * rb] = d.to_le_bytes()[0];
            packed[r * rb + 1] = d.to_le_bytes()[1];
            for j in 0..Q8_0_BLOCK {
                packed[r * rb + 2 + j] = ((r + 1) as i8) as u8;
            }
        }
        let mut mat = vec![0.0f32; rows * cols];
        decode_matrix(&packed, rows, cols, &mut mat).expect("decode_matrix");
        for r in 0..rows {
            let mut row = vec![0.0f32; cols];
            decode_row(&packed[r * rb..(r + 1) * rb], cols, &mut row).expect("row");
            assert_eq!(&mat[r * cols..(r + 1) * cols], &row[..], "row {r}");
        }
    }
}
