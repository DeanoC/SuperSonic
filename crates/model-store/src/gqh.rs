//! GQH (Geo-Quant Hierarchical) codecs 13–15 / GGUF qtypes 108–110.
//!
//! Decode is bit-exact against the geo-lucebox CPU reference and the
//! `tests/gqh-vectors` wires. Do not reassociate the float products.

mod tables {
    include!("gqh_tables.rs");
}

use crate::Error;

pub const GGML_TYPE_GQH3: u32 = 108;
pub const GGML_TYPE_GQH2_H: u32 = 109;
pub const GGML_TYPE_GQH2_C: u32 = 110;

pub const GQH_HEADERS_KV: &str = "geoquant.gqh.headers";
const GQH_MAGIC: &[u8; 8] = b"GQHh1\0\0\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GqhRung {
    Gqh3,
    Gqh2H,
    Gqh2C,
}

impl GqhRung {
    pub fn from_ggml_type(ty: u32) -> Option<Self> {
        match ty {
            GGML_TYPE_GQH3 => Some(Self::Gqh3),
            GGML_TYPE_GQH2_H => Some(Self::Gqh2H),
            GGML_TYPE_GQH2_C => Some(Self::Gqh2C),
            _ => None,
        }
    }

    pub fn from_flm_codec(semantic_id: u16) -> Option<Self> {
        match semantic_id {
            crate::flm::CODEC_GQH3 => Some(Self::Gqh3),
            crate::flm::CODEC_GQH2_H => Some(Self::Gqh2H),
            crate::flm::CODEC_GQH2_C => Some(Self::Gqh2C),
            _ => None,
        }
    }

    pub fn ggml_type(self) -> u32 {
        match self {
            Self::Gqh3 => GGML_TYPE_GQH3,
            Self::Gqh2H => GGML_TYPE_GQH2_H,
            Self::Gqh2C => GGML_TYPE_GQH2_C,
        }
    }

    pub fn flm_codec(self) -> u16 {
        match self {
            Self::Gqh3 => crate::flm::CODEC_GQH3,
            Self::Gqh2H => crate::flm::CODEC_GQH2_H,
            Self::Gqh2C => crate::flm::CODEC_GQH2_C,
        }
    }

    pub fn superblock_bytes(self) -> usize {
        match self {
            Self::Gqh3 => tables::GQH3_SB_BYTES,
            Self::Gqh2H => tables::GQH2H_SB_BYTES,
            Self::Gqh2C => tables::GQH2C_SB_BYTES,
        }
    }

    pub fn has_header(self) -> bool {
        !matches!(self, Self::Gqh2C)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GqhHeader {
    pub qtype: u32,
    pub tensor_scale: f32,
    pub grid_code: u8,
}

fn bits_f32(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn subblock_scale(block: &[u8], sub: usize, tensor_scale: f32) -> f32 {
    let d = block[0];
    let d_real = bits_f32(tables::E4M3_LUT[(d >> 3) as usize][(d & 7) as usize]) * tensor_scale;
    let rb = block[1 + (sub >> 1)];
    let ratio = if sub & 1 == 1 { rb >> 4 } else { rb & 0x0f };
    d_real * bits_f32(tables::RATIO_Q[ratio as usize])
}

/// Packed stored length for a rank-2 logical matrix `(rows, cols)`.
pub fn packed_nbytes(rung: GqhRung, rows: usize, cols: usize) -> Result<usize, Error> {
    if cols == 0 || cols % tables::SUPERBLOCK != 0 {
        return Err(Error::Other(format!(
            "GQH input axis {cols} is not a positive multiple of {}",
            tables::SUPERBLOCK
        )));
    }
    rows.checked_mul(cols / tables::SUPERBLOCK)
        .and_then(|nsb| nsb.checked_mul(rung.superblock_bytes()))
        .ok_or_else(|| Error::Other("GQH packed byte length overflows".into()))
}

const GQH_PLANE_ALIGN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    pub off_ratio: usize,
    pub off_lo: usize,
    pub off_hi: usize,
    pub stride: usize,
}

/// Per-row 4-plane layout. Must match `gqh_plane_offsets` in kernels/gqh-stride.h.
pub fn plane_layout(nsb: usize, is3: bool) -> PlaneLayout {
    let mut o = (nsb + 7) & !7;
    let off_ratio = o;
    o += nsb * 8;
    o = (o + (GQH_PLANE_ALIGN - 1)) & !(GQH_PLANE_ALIGN - 1);
    let off_lo = o;
    o += nsb * 64;
    let off_hi = o;
    if is3 {
        o += nsb * 32;
    }
    let stride = (o + (GQH_PLANE_ALIGN - 1)) & !(GQH_PLANE_ALIGN - 1);
    PlaneLayout {
        off_ratio,
        off_lo,
        off_hi,
        stride,
    }
}

pub fn device_row_bytes(rung: GqhRung, cols: usize) -> Option<usize> {
    if cols == 0 || cols % tables::SUPERBLOCK != 0 {
        return None;
    }
    if matches!(rung, GqhRung::Gqh2C) {
        return Some((cols / tables::SUPERBLOCK) * tables::GQH2C_SB_BYTES);
    }
    Some(plane_layout(cols / tables::SUPERBLOCK, matches!(rung, GqhRung::Gqh3)).stride)
}

pub fn device_nbytes(rung: GqhRung, rows: usize, cols: usize) -> Result<usize, Error> {
    let row = device_row_bytes(rung, cols).ok_or_else(|| {
        Error::Other(format!(
            "GQH input axis {cols} is not a positive multiple of {}",
            tables::SUPERBLOCK
        ))
    })?;
    rows.checked_mul(row)
        .ok_or_else(|| Error::Other("GQH device byte length overflows".into()))
}

/// Scatter tight AoS superblocks into 4 planes (d / ratio / lo / hi).
pub fn planarize(rung: GqhRung, rows: usize, cols: usize, tight: &[u8]) -> Result<Vec<u8>, Error> {
    let want = packed_nbytes(rung, rows, cols)?;
    if tight.len() != want {
        return Err(Error::Other(format!(
            "GQH tight wire {} B, expected {want} B",
            tight.len()
        )));
    }
    if matches!(rung, GqhRung::Gqh2C) {
        return Ok(tight.to_vec());
    }
    let nsb = cols / tables::SUPERBLOCK;
    let payload = rung.superblock_bytes();
    let is3 = matches!(rung, GqhRung::Gqh3);
    let lay = plane_layout(nsb, is3);
    let mut out = vec![0u8; rows * lay.stride];
    for r in 0..rows {
        let src_row = r * nsb * payload;
        let dst_row = r * lay.stride;
        for sb in 0..nsb {
            let src = src_row + sb * payload;
            out[dst_row + sb] = tight[src];
            let ratio = dst_row + lay.off_ratio + sb * 8;
            out[ratio..ratio + 8].copy_from_slice(&tight[src + 1..src + 9]);
            let lo = dst_row + lay.off_lo + sb * 64;
            out[lo..lo + 64].copy_from_slice(&tight[src + 9..src + 73]);
            if is3 {
                let hi = dst_row + lay.off_hi + sb * 32;
                out[hi..hi + 32].copy_from_slice(&tight[src + 73..src + 105]);
            }
        }
    }
    Ok(out)
}

pub fn decode_row(
    rung: GqhRung,
    packed: &[u8],
    cols: usize,
    header: Option<GqhHeader>,
    out: &mut [f32],
) -> Result<(), Error> {
    if cols % tables::SUPERBLOCK != 0 {
        return Err(Error::Other(format!(
            "GQH decode cols {cols} is not a multiple of {}",
            tables::SUPERBLOCK
        )));
    }
    if out.len() != cols {
        return Err(Error::Other(format!(
            "GQH decode output len {} != cols {cols}",
            out.len()
        )));
    }
    let nsb = cols / tables::SUPERBLOCK;
    let want = nsb * rung.superblock_bytes();
    if packed.len() != want {
        return Err(Error::Other(format!(
            "GQH packed row is {} B, expected {want} B",
            packed.len()
        )));
    }
    match rung {
        GqhRung::Gqh3 => {
            let h = header.ok_or_else(|| Error::Other("GQH3 requires a per-tensor header".into()))?;
            decode_gqh3(packed, nsb, h.tensor_scale, h.grid_code, out)?;
        }
        GqhRung::Gqh2H => {
            let h = header.ok_or_else(|| Error::Other("GQH2-H requires a per-tensor header".into()))?;
            decode_gqh2_h(packed, nsb, h.tensor_scale, h.grid_code, out)?;
        }
        GqhRung::Gqh2C => decode_gqh2_c(packed, nsb, out)?,
    }
    Ok(())
}

fn decode_gqh3(
    packed: &[u8],
    nsb: usize,
    tensor_scale: f32,
    grid_code: u8,
    out: &mut [f32],
) -> Result<(), Error> {
    if grid_code as usize >= tables::GRID_CODES {
        return Err(Error::Other(format!("GQH3 grid_code {grid_code} >= 12")));
    }
    let grid = tables::GQH3_GRID[grid_code as usize];
    let sb_bytes = tables::GQH3_SB_BYTES;
    for sb in 0..nsb {
        let b = &packed[sb * sb_bytes..(sb + 1) * sb_bytes];
        for sub in 0..tables::N_SUB {
            let s_b = subblock_scale(b, sub, tensor_scale);
            for t in 0..tables::SUBBLOCK {
                let j = sub * tables::SUBBLOCK + t;
                let lo = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
                let hi = (b[73 + (j >> 3)] >> (j & 7)) & 0x01;
                out[sb * tables::SUPERBLOCK + j] =
                    bits_f32(grid[(lo | (hi << 2)) as usize]) * s_b;
            }
        }
    }
    Ok(())
}

fn decode_gqh2_h(
    packed: &[u8],
    nsb: usize,
    tensor_scale: f32,
    grid_code: u8,
    out: &mut [f32],
) -> Result<(), Error> {
    if grid_code as usize >= tables::GRID_CODES {
        return Err(Error::Other(format!("GQH2-H grid_code {grid_code} >= 12")));
    }
    let grid = tables::GQH2H_GRID[grid_code as usize];
    let sb_bytes = tables::GQH2H_SB_BYTES;
    for sb in 0..nsb {
        let b = &packed[sb * sb_bytes..(sb + 1) * sb_bytes];
        for sub in 0..tables::N_SUB {
            let s_b = subblock_scale(b, sub, tensor_scale);
            for t in 0..tables::SUBBLOCK {
                let j = sub * tables::SUBBLOCK + t;
                let code = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
                out[sb * tables::SUPERBLOCK + j] = bits_f32(grid[code as usize]) * s_b;
            }
        }
    }
    Ok(())
}

fn decode_gqh2_c(packed: &[u8], nsb: usize, out: &mut [f32]) -> Result<(), Error> {
    let sb_bytes = tables::GQH2C_SB_BYTES;
    for sb in 0..nsb {
        let b = &packed[sb * sb_bytes..(sb + 1) * sb_bytes];
        let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        for blk in 0..tables::GQH2C_BLOCKS_PER_SB {
            let p = &b[2 + blk * 8..2 + (blk + 1) * 8];
            let u = u32::from_le_bytes([p[4], p[5], p[6], p[7]]);
            let s_blk = d * bits_f32(tables::RATIO_Q[((u >> 28) & 0x0f) as usize]);
            for grp in 0..tables::GQH2C_GROUPS_PER_BLOCK {
                let mask = tables::GQH2C_SIGN_MASK[((u >> (7 * grp)) & 0x7f) as usize];
                let cb = tables::GQH2C_CODEBOOK[p[grp] as usize];
                let base = sb * tables::SUPERBLOCK + blk * tables::GQH2C_BLOCK + grp * tables::GQH2C_GROUP;
                for e in 0..tables::GQH2C_GROUP {
                    let raw = bits_f32(cb[e]);
                    let mag = if (mask >> e) & 1 == 1 { -raw } else { raw };
                    out[base + e] = mag * s_blk;
                }
            }
        }
    }
    Ok(())
}

/// Parse the frozen `geoquant.gqh.headers` GGUF KV blob.
pub fn parse_gqh_headers_kv(blob: &[u8]) -> Result<Vec<(String, GqhHeader)>, Error> {
    if blob.len() < 16 {
        return Err(Error::Other(format!(
            "GQH header KV truncated ({} B < 16)",
            blob.len()
        )));
    }
    if blob[..8] != GQH_MAGIC[..] {
        return Err(Error::Other("GQH header KV bad magic (expected GQHh1)".into()));
    }
    let count = u32::from_le_bytes(blob[8..12].try_into().unwrap());
    let reserved = u32::from_le_bytes(blob[12..16].try_into().unwrap());
    if reserved != 0 {
        return Err(Error::Other(format!(
            "GQH header KV reserved field {reserved} != 0"
        )));
    }
    let mut off = 16usize;
    let mut out = Vec::with_capacity(count as usize);
    let mut seen = std::collections::HashSet::new();
    for i in 0..count {
        if off + 4 > blob.len() {
            return Err(Error::Other(format!("GQH header KV entry {i} truncated")));
        }
        let name_len = u32::from_le_bytes(blob[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if name_len == 0 || name_len > 1024 || off + name_len > blob.len() {
            return Err(Error::Other(format!(
                "GQH header KV entry {i} bad name_len {name_len}"
            )));
        }
        let name = std::str::from_utf8(&blob[off..off + name_len])
            .map_err(|_| Error::Other(format!("GQH header KV entry {i} name is not UTF-8")))?
            .to_string();
        off += name_len;
        if !seen.insert(name.clone()) {
            return Err(Error::Other(format!("GQH header KV duplicate entry {name}")));
        }
        if off + 12 > blob.len() {
            return Err(Error::Other(format!("GQH header KV '{name}' truncated metadata")));
        }
        let qtype = u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
        let tensor_scale = f32::from_le_bytes(blob[off + 4..off + 8].try_into().unwrap());
        let grid_code = blob[off + 8];
        if blob[off + 9] | blob[off + 10] | blob[off + 11] != 0 {
            return Err(Error::Other(format!("GQH header KV '{name}' padding is not zero")));
        }
        off += 12;
        if qtype != GGML_TYPE_GQH3 && qtype != GGML_TYPE_GQH2_H {
            return Err(Error::Other(format!(
                "GQH header KV '{name}' qtype {qtype} is not 108/109"
            )));
        }
        if grid_code as usize >= tables::GRID_CODES {
            return Err(Error::Other(format!(
                "GQH header KV '{name}' grid_code {grid_code} >= 12"
            )));
        }
        if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
            return Err(Error::Other(format!(
                "GQH header KV '{name}' tensor_scale is not finite and positive"
            )));
        }
        out.push((
            name,
            GqhHeader {
                qtype,
                tensor_scale,
                grid_code,
            },
        ));
    }
    if off != blob.len() {
        return Err(Error::Other(format!(
            "GQH header KV has {} trailing bytes",
            blob.len() - off
        )));
    }
    Ok(out)
}

/// Decode a lucebox-style wire: optional 5-byte header then packed superblocks.
pub fn decode_wire(rung: GqhRung, rows: usize, cols: usize, wire: &[u8]) -> Result<Vec<f32>, Error> {
    let header = if rung.has_header() {
        if wire.len() < tables::HEADER_BYTES {
            return Err(Error::Other("GQH wire missing 5-byte header".into()));
        }
        Some(GqhHeader {
            qtype: rung.ggml_type(),
            tensor_scale: f32::from_le_bytes(wire[..4].try_into().unwrap()),
            grid_code: wire[4],
        })
    } else {
        None
    };
    let packed = if rung.has_header() {
        &wire[tables::HEADER_BYTES..]
    } else {
        wire
    };
    let want = packed_nbytes(rung, rows, cols)?;
    if packed.len() != want {
        return Err(Error::Other(format!(
            "GQH wire body is {} B, expected {want} B",
            packed.len()
        )));
    }
    let row_bytes = want / rows;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        decode_row(
            rung,
            &packed[r * row_bytes..(r + 1) * row_bytes],
            cols,
            header.clone(),
            &mut out[r * cols..(r + 1) * cols],
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn vector_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gqh-vectors")
    }

    fn load_case(rung: &str, rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let stem = vector_dir().join(format!("{rung}_{rows}x{cols}"));
        let wire = std::fs::read(stem.with_extension("wire.bin")).expect("wire.bin");
        let raw = std::fs::read(stem.with_extension("decode.f32")).expect("decode.f32");
        assert_eq!(raw.len(), rows * cols * 4);
        let mut reference = vec![0.0f32; rows * cols];
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            reference[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        (wire, reference)
    }

    fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label} length");
        let mut bad = 0usize;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            if g.to_bits() != w.to_bits() {
                if bad < 4 {
                    panic!(
                        "{label} [{i}] got 0x{:08x} ({g}) want 0x{:08x} ({w})",
                        g.to_bits(),
                        w.to_bits()
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{label} had {bad} mismatches");
    }

    #[test]
    fn decodes_official_gqh_vectors_bit_exactly() {
        let cases = [
            ("gqh3", GqhRung::Gqh3, 1, 256),
            ("gqh3", GqhRung::Gqh3, 4, 512),
            ("gqh2_h", GqhRung::Gqh2H, 1, 256),
            ("gqh2_h", GqhRung::Gqh2H, 4, 512),
            ("gqh2_c", GqhRung::Gqh2C, 1, 256),
            ("gqh2_c", GqhRung::Gqh2C, 4, 512),
        ];
        for (name, rung, rows, cols) in cases {
            let (wire, reference) = load_case(name, rows, cols);
            let got = decode_wire(rung, rows, cols, &wire).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_bits_eq(&got, &reference, &format!("{name} {rows}x{cols}"));
        }
    }

    #[test]
    fn planarize_scatters_fields_onto_shifted_planes() {
        let rows = 2usize;
        let cols = 256usize;
        let nsb = 1usize;
        let mut tight = vec![0u8; rows * 105];
        for r in 0..rows {
            let b = r * 105;
            tight[b] = 0x10 + r as u8;
            tight[b + 1] = 0x20 + r as u8;
            tight[b + 9] = 0x30 + r as u8;
            tight[b + 73] = 0x40 + r as u8;
        }
        let plane = planarize(GqhRung::Gqh3, rows, cols, &tight).expect("planarize");
        let lay = plane_layout(nsb, true);
        assert_eq!(plane.len(), rows * lay.stride);
        assert_eq!(lay.off_lo % 64, 0);
        for r in 0..rows {
            let row = r * lay.stride;
            assert_eq!(plane[row], 0x10 + r as u8);
            assert_eq!(plane[row + lay.off_ratio], 0x20 + r as u8);
            assert_eq!(plane[row + lay.off_lo], 0x30 + r as u8);
            assert_eq!(plane[row + lay.off_hi], 0x40 + r as u8);
        }
    }

    #[test]
    fn maps_gguf_qtypes_onto_flm_codecs() {
        assert_eq!(GqhRung::from_ggml_type(108).unwrap().flm_codec(), 13);
        assert_eq!(GqhRung::from_ggml_type(109).unwrap().flm_codec(), 14);
        assert_eq!(GqhRung::from_ggml_type(110).unwrap().flm_codec(), 15);
        assert_eq!(GqhRung::from_flm_codec(13).unwrap().ggml_type(), 108);
        assert!(GqhRung::from_ggml_type(107).is_none());
    }

    #[test]
    fn parses_and_rejects_gqh_header_kv() {
        let mut blob = Vec::new();
        blob.extend_from_slice(GQH_MAGIC);
        blob.extend_from_slice(&1u32.to_le_bytes());
        blob.extend_from_slice(&0u32.to_le_bytes());
        let name = b"blk.0.attn_q.weight";
        blob.extend_from_slice(&(name.len() as u32).to_le_bytes());
        blob.extend_from_slice(name);
        blob.extend_from_slice(&108u32.to_le_bytes());
        blob.extend_from_slice(&1.5f32.to_le_bytes());
        blob.push(3);
        blob.extend_from_slice(&[0, 0, 0]);
        let parsed = parse_gqh_headers_kv(&blob).expect("valid KV");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0, "blk.0.attn_q.weight");
        assert_eq!(parsed[0].1.grid_code, 3);
        assert_eq!(parsed[0].1.tensor_scale, 1.5);

        blob[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse_gqh_headers_kv(&blob).is_err());
    }
}
