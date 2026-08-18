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
    let blk = block_bytes(qtype).ok_or_else(|| {
        Error::Other(format!("not a mix qtype: {qtype}"))
    })?;
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
            return Err(Error::Other(format!("{DMIX2_KV} '{name}' truncated metadata")));
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
            return Err(Error::Other(format!("{DMIX2_KV} '{name}' truncated codebook")));
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
