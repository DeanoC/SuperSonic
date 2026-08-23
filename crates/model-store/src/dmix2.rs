//! Dense ROCmFPX mix sidecar (`geoquant.dmix2.sidecar`).
//!
//! Wire matches llama.cpp `llama-rocmfpx-mix.cpp` / lucebox `dmix2_sidecar.cpp`:
//!   header : magic "DMX2s1\0\0" (8) | entry_count u32 | reserved u32 (=0)
//!   entry  : name_len u32 | name | qtype u32 (105|106) | C u32 (=2)
//!            | K u32 (8 for 105, 4 for 106) | mode u8 | pad[3]
//!            | codebook (C*K) x bf16 LE

use std::collections::BTreeMap;

use crate::Error;

pub const DMIX2_KV: &str = "geoquant.dmix2.sidecar";
pub const GGML_TYPE_Q3_1_ROCMFP3_MIX: u32 = 105;
pub const GGML_TYPE_Q2_1_ROCMFP2_MIX: u32 = 106;
pub const MIX_QK: usize = 32;
pub const MIX_FP3_BLOCK_BYTES: usize = 14;
pub const MIX_FP2_BLOCK_BYTES: usize = 10;

const MAGIC: [u8; 8] = *b"DMX2s1\0\0";

#[derive(Debug, Clone)]
pub struct MixHeader {
    pub qtype: u32,
    pub mode: i32,
    pub k: i32,
    pub lut: [f32; 16],
}

pub fn block_bytes(qtype: u32) -> Option<usize> {
    match qtype {
        GGML_TYPE_Q3_1_ROCMFP3_MIX => Some(MIX_FP3_BLOCK_BYTES),
        GGML_TYPE_Q2_1_ROCMFP2_MIX => Some(MIX_FP2_BLOCK_BYTES),
        _ => None,
    }
}

pub fn row_bytes(qtype: u32, cols: usize) -> Result<usize, Error> {
    let blk =
        block_bytes(qtype).ok_or_else(|| Error::Other(format!("not a mix qtype: {qtype}")))?;
    if cols == 0 || cols % MIX_QK != 0 {
        return Err(Error::Other(format!(
            "mix qtype {qtype} cols {cols} is not a multiple of {MIX_QK}"
        )));
    }
    Ok((cols / MIX_QK) * blk)
}

pub fn levels_for_qtype(qtype: u32) -> Option<i32> {
    match qtype {
        GGML_TYPE_Q3_1_ROCMFP3_MIX => Some(8),
        GGML_TYPE_Q2_1_ROCMFP2_MIX => Some(4),
        _ => None,
    }
}

/// 72-byte device sidecar: lut[16] f32 + mode i32 + k i32.
pub fn sidecar_bytes(h: &MixHeader) -> [u8; 72] {
    let mut out = [0u8; 72];
    for (i, v) in h.lut.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out[64..68].copy_from_slice(&h.mode.to_le_bytes());
    out[68..72].copy_from_slice(&h.k.to_le_bytes());
    out
}

fn mix_ue4m3(e: u8) -> f32 {
    if e > 0x7e {
        return 0.0;
    }
    let exp = e >> 3;
    let mant = e & 7;
    if exp == 0 {
        return f32::from(mant) * 0.0009765625;
    }
    f32::from(8 + mant) * 2f32.powi(i32::from(exp) - 11)
}

fn mix_fp3_code(qs: &[u8], i: usize) -> u32 {
    let bit = i * 3;
    let byte = bit >> 3;
    let shift = bit & 7;
    let mut v = u32::from(qs[byte]);
    if byte + 1 < MIX_FP3_BLOCK_BYTES - 2 {
        v |= u32::from(qs[byte + 1]) << 8;
    }
    if byte + 2 < MIX_FP3_BLOCK_BYTES - 2 {
        v |= u32::from(qs[byte + 2]) << 16;
    }
    (v >> shift) & 7
}

fn mix_fp3_fixed(code: u32) -> f32 {
    let m = code & 3;
    let mag = if m == 3 { 4 } else { m as i32 };
    if code & 4 != 0 {
        -(mag as f32)
    } else {
        mag as f32
    }
}

fn mix_fp2_code(qs: &[u8], i: usize) -> u32 {
    u32::from(qs[i >> 2] >> (2 * (i & 3))) & 3
}

fn mix_fp2_fixed(code: u32) -> f32 {
    (code as i32 - 1) as f32
}

/// Dequantize one output row of a dense 105/106 tensor.
pub fn decode_row(
    qtype: u32,
    packed: &[u8],
    cols: usize,
    header: &MixHeader,
    out: &mut [f32],
) -> Result<(), Error> {
    let row = row_bytes(qtype, cols)?;
    if packed.len() != row {
        return Err(Error::Other(format!(
            "mix packed row is {} B, expected {row} B",
            packed.len()
        )));
    }
    if out.len() != cols {
        return Err(Error::Other(format!(
            "mix decode out len {} != cols {cols}",
            out.len()
        )));
    }
    let blk = block_bytes(qtype).unwrap();
    let qs_len = blk - 2;
    let klev = header.k as usize;
    let nsb = cols / MIX_QK;
    for sb in 0..nsb {
        let b = &packed[sb * blk..(sb + 1) * blk];
        let m0 = b[qs_len];
        let m1 = b[qs_len + 1];
        for j in 0..MIX_QK {
            let meta = if j < MIX_QK / 2 { m0 } else { m1 };
            let code = if qtype == GGML_TYPE_Q3_1_ROCMFP3_MIX {
                mix_fp3_code(b, j)
            } else {
                mix_fp2_code(b, j)
            };
            let w = if header.mode == 0 {
                let s = mix_ue4m3(meta);
                let lvl = if qtype == GGML_TYPE_Q3_1_ROCMFP3_MIX {
                    mix_fp3_fixed(code)
                } else {
                    mix_fp2_fixed(code)
                };
                s * lvl
            } else {
                let s = mix_ue4m3(meta & 0x7f);
                let bk = (meta >> 7) as usize;
                s * header.lut[bk * klev + code as usize]
            };
            out[sb * MIX_QK + j] = w;
        }
    }
    Ok(())
}

pub fn parse_dmix2_kv(blob: &[u8]) -> Result<BTreeMap<String, MixHeader>, Error> {
    if blob.len() < 16 {
        return Err(Error::Other(format!(
            "{DMIX2_KV} truncated ({} B < 16 B header)",
            blob.len()
        )));
    }
    if blob[..8] != MAGIC {
        return Err(Error::Other(format!("{DMIX2_KV} bad magic")));
    }
    let count = u32::from_le_bytes(blob[8..12].try_into().unwrap());
    let reserved = u32::from_le_bytes(blob[12..16].try_into().unwrap());
    if reserved != 0 {
        return Err(Error::Other(format!(
            "{DMIX2_KV} reserved field {reserved} != 0"
        )));
    }
    let mut entries = BTreeMap::new();
    let mut off = 16usize;
    for i in 0..count {
        if off + 4 > blob.len() {
            return Err(Error::Other(format!("{DMIX2_KV} entry {i} truncated")));
        }
        let name_len = u32::from_le_bytes(blob[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if name_len == 0 || name_len > 1024 || off + name_len > blob.len() {
            return Err(Error::Other(format!(
                "{DMIX2_KV} entry {i} bad name_len {name_len}"
            )));
        }
        let name = String::from_utf8(blob[off..off + name_len].to_vec())
            .map_err(|e| Error::Other(format!("{DMIX2_KV} name utf-8: {e}")))?;
        off += name_len;
        if entries.contains_key(&name) {
            return Err(Error::Other(format!("{DMIX2_KV} duplicate '{name}'")));
        }
        if off + 16 > blob.len() {
            return Err(Error::Other(format!(
                "{DMIX2_KV} '{name}' truncated metadata"
            )));
        }
        let qtype = u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
        let c = u32::from_le_bytes(blob[off + 4..off + 8].try_into().unwrap());
        let k = u32::from_le_bytes(blob[off + 8..off + 12].try_into().unwrap());
        let mode = blob[off + 12];
        off += 16;
        let want_k = levels_for_qtype(qtype).ok_or_else(|| {
            Error::Other(format!("{DMIX2_KV} '{name}' qtype {qtype} is not 105/106"))
        })?;
        if c != 2 || k as i32 != want_k || mode > 1 {
            return Err(Error::Other(format!(
                "{DMIX2_KV} '{name}' bad C/K/mode C={c} K={k} mode={mode}"
            )));
        }
        let cb_bytes = (c * k * 2) as usize;
        if off + cb_bytes > blob.len() {
            return Err(Error::Other(format!(
                "{DMIX2_KV} '{name}' truncated codebook"
            )));
        }
        let mut lut = [0.0f32; 16];
        for (i, chunk) in blob[off..off + cb_bytes].chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            lut[i] = half::bf16::from_bits(bits).to_f32();
        }
        off += cb_bytes;
        entries.insert(
            name,
            MixHeader {
                qtype,
                mode: i32::from(mode),
                k: want_k,
                lut,
            },
        );
    }
    if off != blob.len() {
        return Err(Error::Other(format!(
            "{DMIX2_KV} {} trailing bytes after {count} entries",
            blob.len() - off
        )));
    }
    Ok(entries)
}
