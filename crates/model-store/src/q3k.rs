//! ggml Q3_K row dequant (CPU). Used for the Qwen3.8 token embed when the
//! artifact stores `token_embd.weight` as type 11.
//!
//! Layout matches `block_q3_K`: 32 B hmask, 64 B qs, 12 B scales, fp16 `d`.
//! 110 bytes per 256 weights.

use crate::Error;

pub const Q3K_BLOCK: usize = 256;
pub const Q3K_BLOCK_BYTES: usize = 110;

pub fn row_bytes(cols: usize) -> Result<usize, Error> {
    if cols == 0 || cols % Q3K_BLOCK != 0 {
        return Err(Error::Other(format!(
            "Q3_K cols {cols} is not a positive multiple of {Q3K_BLOCK}"
        )));
    }
    Ok((cols / Q3K_BLOCK) * Q3K_BLOCK_BYTES)
}

pub fn decode_row(packed: &[u8], cols: usize, out: &mut [f32]) -> Result<(), Error> {
    let want = row_bytes(cols)?;
    if packed.len() != want {
        return Err(Error::Other(format!(
            "Q3_K packed row is {} B, expected {want} B",
            packed.len()
        )));
    }
    if out.len() != cols {
        return Err(Error::Other(format!(
            "Q3_K decode output len {} != cols {cols}",
            out.len()
        )));
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let nsb = cols / Q3K_BLOCK;
    for sb in 0..nsb {
        let b = &packed[sb * Q3K_BLOCK_BYTES..(sb + 1) * Q3K_BLOCK_BYTES];
        let hmask = &b[..32];
        let mut qs = &b[32..96];
        let d_all = half::f16::from_le_bytes([b[108], b[109]]).to_f32();
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes(b[96..100].try_into().unwrap());
        aux[1] = u32::from_le_bytes(b[100..104].try_into().unwrap());
        aux[2] = u32::from_le_bytes(b[104..108].try_into().unwrap());
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales = unsafe { std::slice::from_raw_parts(aux.as_ptr() as *const i8, 16) };
        let mut y = sb * Q3K_BLOCK;
        let mut is = 0usize;
        let mut m = 1u8;
        for _n in 0..2 {
            let mut shift = 0;
            for _j in 0..4 {
                let dl = d_all * (f32::from(scales[is]) - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = i32::from((qs[l] >> shift) & 3);
                    let h = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    out[y] = dl * (q - h) as f32;
                    y += 1;
                }
                let dl = d_all * (f32::from(scales[is]) - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = i32::from((qs[l + 16] >> shift) & 3);
                    let h = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    out[y] = dl * (q - h) as f32;
                    y += 1;
                }
                shift += 2;
                m <<= 1;
            }
            qs = &qs[32..];
        }
    }
    Ok(())
}
