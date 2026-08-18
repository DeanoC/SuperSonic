//! Minimal GGUF v3 reader for Qwen3.8 GQH artifacts.
//!
//! Captures tensor bytes and the `geoquant.gqh.headers` KV. Unknown qtypes
//! still need a packed-size formula so the data section can be bounded.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::dmix2::{self, MixHeader};
use crate::gqh::{self, GqhHeader, GqhRung};
use crate::Error;

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_BF16: u32 = 30;

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub dims: Vec<usize>,
    pub tensor_type: u32,
    pub offset: usize,
    pub nbytes: usize,
}

pub struct GgufFile {
    mmap: Mmap,
    tensors: BTreeMap<String, GgufTensor>,
    data_offset: usize,
    kv: BTreeMap<String, String>,
    gqh_headers: BTreeMap<String, GqhHeader>,
    mix_headers: BTreeMap<String, MixHeader>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap);
        if cursor.take(4)? != b"GGUF" {
            return Err(Error::Other(format!(
                "{} does not start with GGUF magic",
                path.display()
            )));
        }
        let version = cursor.read_u32()?;
        if version != 3 {
            return Err(Error::Other(format!(
                "unsupported GGUF version {version}; expected v3"
            )));
        }
        let tensor_count = cursor.read_usize("tensor count")?;
        let metadata_count = cursor.read_usize("metadata count")?;
        let mut alignment = 32usize;
        let mut kv = BTreeMap::new();
        let mut gqh_headers = BTreeMap::new();
        let mut mix_headers = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = cursor.read_string()?;
            let value_type = cursor.read_u32()?;
            read_metadata(
                &mut cursor,
                value_type,
                &key,
                &mut alignment,
                &mut kv,
                &mut gqh_headers,
                &mut mix_headers,
            )?;
        }
        if alignment == 0 {
            return Err(Error::Other("general.alignment must be non-zero".into()));
        }

        let mut raw = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let n_dims = cursor.read_u32()? as usize;
            if n_dims == 0 {
                return Err(Error::Other(format!("tensor {name} has zero dimensions")));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cursor.read_usize("tensor dimension")?);
            }
            let tensor_type = cursor.read_u32()?;
            let offset = cursor.read_usize("tensor offset")?;
            let nbytes = tensor_nbytes(&dims, tensor_type)?;
            raw.push((name, dims, tensor_type, offset, nbytes));
        }

        let data_offset = align_up(cursor.pos, alignment)?;
        let mut tensors = BTreeMap::new();
        for (name, dims, tensor_type, offset, nbytes) in raw {
            let start = data_offset
                .checked_add(offset)
                .ok_or_else(|| Error::Other(format!("tensor {name} offset overflows")))?;
            let end = start
                .checked_add(nbytes)
                .ok_or_else(|| Error::Other(format!("tensor {name} size overflows")))?;
            if end > mmap.len() {
                return Err(Error::Other(format!(
                    "tensor {name} range {start}..{end} exceeds file length {}",
                    mmap.len()
                )));
            }
            tensors.insert(
                name,
                GgufTensor {
                    dims,
                    tensor_type,
                    offset,
                    nbytes,
                },
            );
        }
        Ok(Self {
            mmap,
            tensors,
            data_offset,
            kv,
            gqh_headers,
            mix_headers,
        })
    }

    pub fn kv(&self, key: &str) -> Option<&str> {
        self.kv.get(key).map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], Error> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let start = self
            .data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| Error::Other("tensor offset overflows".into()))?;
        let end = start
            .checked_add(tensor.nbytes)
            .ok_or_else(|| Error::Other("tensor size overflows".into()))?;
        self.mmap
            .get(start..end)
            .ok_or_else(|| Error::Other(format!("tensor {name} range is out of bounds")))
    }

    pub fn gqh_header(&self, name: &str) -> Option<&GqhHeader> {
        self.gqh_headers.get(name)
    }

    pub fn gqh_header_count(&self) -> usize {
        self.gqh_headers.len()
    }

    pub fn mix_header(&self, name: &str) -> Option<&MixHeader> {
        self.mix_headers.get(name)
    }
}

pub fn tensor_nbytes(dims: &[usize], tensor_type: u32) -> Result<usize, Error> {
    let elems = dims.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::Other("tensor element count overflows".into()))
    })?;
    match tensor_type {
        GGML_TYPE_F32 => elems
            .checked_mul(4)
            .ok_or_else(|| Error::Other("F32 tensor byte size overflows".into())),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => elems
            .checked_mul(2)
            .ok_or_else(|| Error::Other("16-bit tensor byte size overflows".into())),
        GGML_TYPE_Q8_0 => kquant_nbytes(dims, 32, 34, "Q8_0"),
        GGML_TYPE_Q2_K => kquant_nbytes(dims, 256, 84, "Q2_K"),
        GGML_TYPE_Q3_K => kquant_nbytes(dims, 256, 110, "Q3_K"),
        GGML_TYPE_Q4_K => kquant_nbytes(dims, 256, 144, "Q4_K"),
        t if dmix2::block_bytes(t).is_some() => {
            if dims.len() != 2 {
                return Err(Error::Other(format!(
                    "mix tensor must be rank-2, got {dims:?}"
                )));
            }
            dmix2::row_bytes(t, dims[0]).and_then(|row| {
                dims[1]
                    .checked_mul(row)
                    .ok_or_else(|| Error::Other("mix tensor byte size overflows".into()))
            })
        }
        GGML_TYPE_Q5_K => kquant_nbytes(dims, 256, 176, "Q5_K"),
        GGML_TYPE_Q6_K => kquant_nbytes(dims, 256, 210, "Q6_K"),
        t if GqhRung::from_ggml_type(t).is_some() => {
            if dims.len() != 2 {
                return Err(Error::Other(format!(
                    "GQH tensor must be rank-2, got {dims:?}"
                )));
            }
            let rung = GqhRung::from_ggml_type(t).unwrap();
            gqh::packed_nbytes(rung, dims[1], dims[0])
        }
        other => Err(Error::Other(format!(
            "unsupported GGUF tensor type {other}"
        ))),
    }
}

fn kquant_nbytes(dims: &[usize], block: usize, bytes: usize, name: &str) -> Result<usize, Error> {
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "{name} tensor must be rank-2, got {dims:?}"
        )));
    }
    if dims[0] == 0 || dims[0] % block != 0 {
        return Err(Error::Other(format!(
            "{name} ne[0]={} is not a multiple of {block}",
            dims[0]
        )));
    }
    let row = (dims[0] / block)
        .checked_mul(bytes)
        .ok_or_else(|| Error::Other(format!("{name} row size overflows")))?;
    dims[1]
        .checked_mul(row)
        .ok_or_else(|| Error::Other(format!("{name} tensor byte size overflows")))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, Error> {
    let padded = value
        .checked_add(alignment - 1)
        .ok_or_else(|| Error::Other("alignment overflow".into()))?;
    Ok((padded / alignment) * alignment)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Error::Other(format!("cursor overflow at {}", self.pos)))?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| {
            Error::Other(format!(
                "unexpected EOF while reading {len} bytes at {}",
                self.pos
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, Error> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, Error> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, Error> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_usize(&mut self, what: &str) -> Result<usize, Error> {
        usize::try_from(self.read_u64()?)
            .map_err(|_| Error::Other(format!("{what} does not fit in usize")))
    }

    fn read_string(&mut self) -> Result<String, Error> {
        let len = self.read_usize("string length")?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|err| Error::Other(format!("invalid UTF-8 in GGUF string: {err}")))
    }
}

fn read_metadata(
    cursor: &mut Cursor<'_>,
    value_type: u32,
    key: &str,
    alignment: &mut usize,
    kv: &mut BTreeMap<String, String>,
    gqh_headers: &mut BTreeMap<String, GqhHeader>,
    mix_headers: &mut BTreeMap<String, MixHeader>,
) -> Result<(), Error> {
    match value_type {
        0 | 1 => {
            let _ = cursor.read_u8()?;
        }
        2 | 3 => {
            let _ = cursor.read_u16()?;
        }
        4 | 5 => {
            let value = cursor.read_u32()?;
            if key == "general.alignment" {
                *alignment = value as usize;
            }
            if !key.is_empty() {
                kv.insert(key.to_string(), value.to_string());
            }
        }
        6 => {
            let bits = cursor.read_u32()?;
            if !key.is_empty() {
                kv.insert(key.to_string(), f32::from_bits(bits).to_string());
            }
        }
        7 => {
            let _ = cursor.read_u8()?;
        }
        8 => {
            let value = cursor.read_string()?;
            if !key.is_empty() {
                kv.insert(key.to_string(), value);
            }
        }
        9 => {
            let elem_type = cursor.read_u32()?;
            let len = cursor.read_usize("array length")?;
            if key == gqh::GQH_HEADERS_KV && elem_type == 0 {
                let blob = cursor.take(len)?.to_vec();
                for (name, header) in gqh::parse_gqh_headers_kv(&blob)? {
                    gqh_headers.insert(name, header);
                }
            } else if key == dmix2::DMIX2_KV && elem_type == 0 {
                let blob = cursor.take(len)?.to_vec();
                *mix_headers = dmix2::parse_dmix2_kv(&blob)?;
            } else {
                for _ in 0..len {
                    read_metadata(
                        cursor,
                        elem_type,
                        "",
                        alignment,
                        kv,
                        gqh_headers,
                        mix_headers,
                    )?;
                }
            }
        }
        10 | 11 => {
            let value = cursor.read_u64()?;
            if key == "general.alignment" {
                *alignment = usize::try_from(value)
                    .map_err(|_| Error::Other("general.alignment does not fit in usize".into()))?;
            }
        }
        12 => {
            let _ = cursor.read_u64()?;
        }
        other => {
            return Err(Error::Other(format!(
                "unsupported metadata value type {other} for key {key}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn qwen38_gguf() -> Option<PathBuf> {
        let default = PathBuf::from("/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf");
        default.is_file().then_some(default)
    }

    #[test]
    fn opens_qwen38_gqh_q2kxl_gptq_gguf() {
        let Some(path) = qwen38_gguf() else {
            return;
        };
        let file = GgufFile::open(&path).expect("open");
        assert_eq!(file.kv("general.architecture"), Some("qwen35"));
        assert_eq!(file.kv("general.basename"), Some("qwen38"));
        assert_eq!(file.gqh_header_count(), 350);
        let qkv = file.tensor("blk.0.attn_qkv.weight").expect("qkv");
        assert_eq!(qkv.dims, vec![5120, 10240]);
        assert_eq!(qkv.tensor_type, gqh::GGML_TYPE_GQH2_H);
        assert!(file.gqh_header("blk.0.attn_qkv.weight").is_some());
        assert!(file.gqh_header("output.weight").is_some());
        let embed = file.tensor("token_embd.weight").expect("embed");
        assert_eq!(embed.tensor_type, 10);
        assert_eq!(embed.dims, vec![5120, 248320]);
    }
}
