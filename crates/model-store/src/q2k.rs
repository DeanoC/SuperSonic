//! ggml Q2_K row dequant (CPU). Used for the Qwen3.8 token embed.
//!
//! Layout matches `block_q2_K` in ggml-common.h: 16 scale/min nibbles, 64 B
//! of 2-bit codes, then fp16 `d` and `dmin`. 84 bytes per 256 weights.

use crate::Error;

pub const Q2K_BLOCK: usize = 256;
pub const Q2K_BLOCK_BYTES: usize = 84;

pub fn row_bytes(cols: usize) -> Result<usize, Error> {
    if cols == 0 || cols % Q2K_BLOCK != 0 {
        return Err(Error::Other(format!(
            "Q2_K cols {cols} is not a positive multiple of {Q2K_BLOCK}"
        )));
    }
    Ok((cols / Q2K_BLOCK) * Q2K_BLOCK_BYTES)
}

pub fn decode_row(packed: &[u8], cols: usize, out: &mut [f32]) -> Result<(), Error> {
    let want = row_bytes(cols)?;
    if packed.len() != want {
        return Err(Error::Other(format!(
            "Q2_K packed row is {} B, expected {want} B",
            packed.len()
        )));
    }
    if out.len() != cols {
        return Err(Error::Other(format!(
            "Q2_K decode output len {} != cols {cols}",
            out.len()
        )));
    }
    let nsb = cols / Q2K_BLOCK;
    for sb in 0..nsb {
        let b = &packed[sb * Q2K_BLOCK_BYTES..(sb + 1) * Q2K_BLOCK_BYTES];
        let scales = &b[..16];
        let qs = &b[16..80];
        let d = half::f16::from_le_bytes([b[80], b[81]]).to_f32();
        let dmin = half::f16::from_le_bytes([b[82], b[83]]).to_f32();
        let mut y = sb * Q2K_BLOCK;
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _n in 0..2 {
            let mut shift = 0;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = dmin * f32::from(sc >> 4);
                for l in 0..16 {
                    let q = i32::from((qs[q_off + l] >> shift) & 3);
                    out[y] = dl * (q as f32) - ml;
                    y += 1;
                }
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = dmin * f32::from(sc >> 4);
                for l in 0..16 {
                    let q = i32::from((qs[q_off + 16 + l] >> shift) & 3);
                    out[y] = dl * (q as f32) - ml;
                    y += 1;
                }
                shift += 2;
            }
            q_off += 32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequants_constant_q2k_block() {
        let mut block = [0u8; Q2K_BLOCK_BYTES];
        // scale nibble 1, min nibble 0
        for s in &mut block[..16] {
            *s = 0x01;
        }
        // 0x55 → four q=1 codes per byte
        for q in &mut block[16..80] {
            *q = 0x55;
        }
        block[80..82].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
        let mut out = [0.0f32; Q2K_BLOCK];
        decode_row(&block, Q2K_BLOCK, &mut out).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 2.0, "idx {i}");
        }
    }
}
