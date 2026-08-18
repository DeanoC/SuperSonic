//! Safetensors loader for the DFlash draft checkpoint.
//!
//! The draft ships as a single `model.safetensors` (~2 GiB BF16, 58 tensors).
//! Mirrors the lean per-family loader pattern in `crates/phi4/src/loader.rs` —
//! no row-slicing needed because DFlash doesn't pack q/k/v into a single
//! `qkv_proj` tensor; each projection is already separate.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use gpu_hal::{GpuBuffer, GpuError, ScalarType};
use half::{bf16, f16};
use memmap2::Mmap;
use qwen35::weights::{
    ggml_k_row_bytes, LOWBIT_GGML_Q2_K, LOWBIT_GGML_Q3_K, LOWBIT_GGML_Q4_K, LOWBIT_GGML_Q5_K,
    LOWBIT_GGML_Q6_K, LOWBIT_GGML_Q8_0, LOWBIT_ROCMFP2_MIX, LOWBIT_ROCMFP3_MIX,
};
use safetensors::SafeTensors;

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Safetensors(safetensors::SafeTensorError),
    Gpu(GpuError),
    NotFound(String),
    UnsupportedDtype(String),
    UnexpectedTensor(String),
    InvalidGguf(String),
    Json(serde_json::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Safetensors(e) => write!(f, "safetensors error: {e}"),
            Self::Gpu(e) => write!(f, "GPU error: {e}"),
            Self::NotFound(name) => write!(f, "tensor not found: {name}"),
            Self::UnsupportedDtype(msg) => write!(f, "unsupported dtype: {msg}"),
            Self::UnexpectedTensor(msg) => write!(f, "unexpected tensor in draft: {msg}"),
            Self::InvalidGguf(msg) => write!(f, "invalid GGUF draft: {msg}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<safetensors::SafeTensorError> for LoadError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(e)
    }
}
impl From<GpuError> for LoadError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}
impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub struct WeightLoader {
    shards: Vec<Mmap>,
    index: BTreeMap<String, usize>,
}

impl WeightLoader {
    pub fn from_dir(dir: &Path) -> Result<Self, LoadError> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Self::from_single(&single);
        }
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            return Self::from_sharded(dir, &index_path);
        }
        Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no safetensors files found in {}", dir.display()),
        )))
    }

    fn from_single(path: &Path) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        let mut index = BTreeMap::new();
        for name in tensors.names() {
            index.insert(name.to_string(), 0);
        }
        Ok(Self {
            shards: vec![mmap],
            index,
        })
    }

    fn from_sharded(dir: &Path, index_path: &Path) -> Result<Self, LoadError> {
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(index_path)?)?;
        let weight_map = raw["weight_map"]
            .as_object()
            .ok_or_else(|| LoadError::NotFound("weight_map key in index.json".into()))?;
        let mut shard_files: Vec<String> = Vec::new();
        let mut shard_idx_map: BTreeMap<String, usize> = BTreeMap::new();
        for filename in weight_map.values() {
            let filename = filename.as_str().unwrap_or("").to_string();
            if !shard_idx_map.contains_key(&filename) {
                shard_idx_map.insert(filename.clone(), shard_files.len());
                shard_files.push(filename);
            }
        }
        let mut shards = Vec::with_capacity(shard_files.len());
        for filename in &shard_files {
            let file = File::open(dir.join(filename))?;
            shards.push(unsafe { Mmap::map(&file)? });
        }
        let mut index = BTreeMap::new();
        for (tensor_name, filename) in weight_map {
            let filename = filename.as_str().unwrap_or("");
            if let Some(&shard_idx) = shard_idx_map.get(filename) {
                index.insert(tensor_name.clone(), shard_idx);
            }
        }
        Ok(Self { shards, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn load_to_gpu(&self, name: &str, ordinal: usize) -> Result<GpuBuffer, LoadError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))?;
        let tensors = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = tensors.tensor(name)?;
        let dtype = ScalarType::from_safetensors(view.dtype())
            .ok_or_else(|| LoadError::UnsupportedDtype(format!("{:?}", view.dtype())))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let buf = GpuBuffer::from_host_bytes(ordinal, dtype, &shape, view.data())?;
        Ok(buf)
    }

    pub fn load_concat_dim0_to_gpu(
        &self,
        first: &str,
        second: &str,
        ordinal: usize,
    ) -> Result<GpuBuffer, LoadError> {
        let (dtype_a, shape_a, data_a) = self.tensor_bytes(first)?;
        let (dtype_b, shape_b, data_b) = self.tensor_bytes(second)?;
        if dtype_a != dtype_b {
            return Err(LoadError::UnsupportedDtype(format!(
                "cannot concat tensors with different dtypes: {first}={dtype_a:?}, {second}={dtype_b:?}"
            )));
        }
        if shape_a.len() != shape_b.len() || shape_a.is_empty() || shape_a[1..] != shape_b[1..] {
            return Err(LoadError::UnexpectedTensor(format!(
                "cannot concat tensors with incompatible shapes: {first}={shape_a:?}, {second}={shape_b:?}"
            )));
        }

        let mut shape = shape_a.clone();
        shape[0] += shape_b[0];
        let mut bytes = Vec::with_capacity(data_a.len() + data_b.len());
        bytes.extend_from_slice(&data_a);
        bytes.extend_from_slice(&data_b);
        Ok(GpuBuffer::from_host_bytes(
            ordinal, dtype_a, &shape, &bytes,
        )?)
    }

    fn tensor_bytes(&self, name: &str) -> Result<(ScalarType, Vec<usize>, Vec<u8>), LoadError> {
        let &shard_idx = self
            .index
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))?;
        let tensors = SafeTensors::deserialize(&self.shards[shard_idx])?;
        let view = tensors.tensor(name)?;
        let dtype = ScalarType::from_safetensors(view.dtype())
            .ok_or_else(|| LoadError::UnsupportedDtype(format!("{:?}", view.dtype())))?;
        Ok((dtype, view.shape().to_vec(), view.data().to_vec()))
    }
}

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_BF16: u32 = 30;
const GGML_TYPE_ROCMFP3_MIX: u32 = 105;
const GGML_TYPE_ROCMFP2_MIX: u32 = 106;
const GGML_TYPE_GQH3: u32 = 108;
const GGML_TYPE_GQH2_H: u32 = 109;
const GGML_TYPE_GQH2_C: u32 = 110;

#[derive(Debug, Clone)]
struct GgufTensor {
    dims: Vec<usize>,
    tensor_type: u32,
    offset: usize,
    nbytes: usize,
}

struct LinearHostParts {
    dtype: ScalarType,
    quant_type: i32,
    logical_rows: usize,
    logical_cols: usize,
    upload_shape: Vec<usize>,
    bytes: Vec<u8>,
    row_bytes: usize,
}

pub struct GgufWeightLoader {
    mmap: Mmap,
    tensors: BTreeMap<String, GgufTensor>,
    data_offset: usize,
}

impl GgufWeightLoader {
    pub fn from_file(path: &Path) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = GgufCursor::new(&mmap);
        if cursor.take(4)? != b"GGUF" {
            return Err(LoadError::InvalidGguf(format!(
                "{} does not start with GGUF magic",
                path.display()
            )));
        }
        let version = cursor.read_u32()?;
        if version != 3 {
            return Err(LoadError::InvalidGguf(format!(
                "unsupported GGUF version {version}; expected v3"
            )));
        }
        let tensor_count = cursor.read_u64_usize("tensor count")?;
        let metadata_count = cursor.read_u64_usize("metadata count")?;
        let mut alignment = 32usize;
        for _ in 0..metadata_count {
            let key = cursor.read_string()?;
            let value_type = cursor.read_u32()?;
            read_gguf_metadata_value(&mut cursor, value_type, &key, &mut alignment)?;
        }
        if alignment == 0 {
            return Err(LoadError::InvalidGguf(
                "general.alignment must be non-zero".into(),
            ));
        }

        let mut raw_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let n_dims = cursor.read_u32()? as usize;
            if n_dims == 0 {
                return Err(LoadError::InvalidGguf(format!(
                    "tensor {name} has zero dimensions"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64_usize("tensor dimension")?);
            }
            let tensor_type = cursor.read_u32()?;
            let offset = cursor.read_u64_usize("tensor offset")?;
            let nbytes = gguf_tensor_nbytes(&dims, tensor_type)?;
            raw_infos.push((name, dims, tensor_type, offset, nbytes));
        }

        let data_offset = align_up(cursor.position(), alignment)?;
        let mut tensors = BTreeMap::new();
        for (name, dims, tensor_type, offset, nbytes) in raw_infos {
            let start = data_offset.checked_add(offset).ok_or_else(|| {
                LoadError::InvalidGguf(format!("tensor {name} offset overflows usize"))
            })?;
            let end = start.checked_add(nbytes).ok_or_else(|| {
                LoadError::InvalidGguf(format!("tensor {name} size overflows usize"))
            })?;
            if end > mmap.len() {
                return Err(LoadError::InvalidGguf(format!(
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
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn load_norm_bf16_to_gpu(
        &self,
        name: &str,
        ordinal: usize,
    ) -> Result<GpuBuffer, LoadError> {
        let tensor = self.tensor(name)?;
        let data = self.tensor_slice(tensor)?;
        let bytes = tensor_to_bf16_bytes(tensor.tensor_type, data)?;
        Ok(GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::BF16,
            &tensor.dims,
            &bytes,
        )?)
    }

    pub fn load_linear_to_gpu(
        &self,
        name: &str,
        ordinal: usize,
    ) -> Result<(GpuBuffer, i32, usize, usize), LoadError> {
        let parts = self.linear_host_parts(name)?;
        let buffer =
            GpuBuffer::from_host_bytes(ordinal, parts.dtype, &parts.upload_shape, &parts.bytes)?;
        Ok((
            buffer,
            parts.quant_type,
            parts.logical_rows,
            parts.logical_cols,
        ))
    }

    pub fn load_concat_dim0_linear_to_gpu(
        &self,
        first: &str,
        second: &str,
        ordinal: usize,
    ) -> Result<(GpuBuffer, i32, usize, usize), LoadError> {
        let mut lhs = self.linear_host_parts(first)?;
        let rhs = self.linear_host_parts(second)?;
        if lhs.dtype != rhs.dtype
            || lhs.quant_type != rhs.quant_type
            || lhs.logical_cols != rhs.logical_cols
            || lhs.row_bytes != rhs.row_bytes
        {
            return Err(LoadError::UnexpectedTensor(format!(
                "cannot concat GGUF linears: {first} rows={} cols={} qtype={} dtype={:?}, {second} rows={} cols={} qtype={} dtype={:?}",
                lhs.logical_rows,
                lhs.logical_cols,
                lhs.quant_type,
                lhs.dtype,
                rhs.logical_rows,
                rhs.logical_cols,
                rhs.quant_type,
                rhs.dtype
            )));
        }
        lhs.logical_rows += rhs.logical_rows;
        if lhs.quant_type == 0 {
            lhs.upload_shape = vec![lhs.logical_rows, lhs.logical_cols];
        } else {
            lhs.upload_shape = vec![lhs.logical_rows, lhs.row_bytes];
        }
        lhs.bytes.extend_from_slice(&rhs.bytes);
        let buffer = GpuBuffer::from_host_bytes(ordinal, lhs.dtype, &lhs.upload_shape, &lhs.bytes)?;
        Ok((buffer, lhs.quant_type, lhs.logical_rows, lhs.logical_cols))
    }

    fn tensor(&self, name: &str) -> Result<&GgufTensor, LoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| LoadError::NotFound(name.to_string()))
    }

    fn tensor_slice<'a>(&'a self, tensor: &GgufTensor) -> Result<&'a [u8], LoadError> {
        let start = self
            .data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| LoadError::InvalidGguf("tensor offset overflows usize".to_string()))?;
        let end = start
            .checked_add(tensor.nbytes)
            .ok_or_else(|| LoadError::InvalidGguf("tensor size overflows usize".to_string()))?;
        self.mmap.get(start..end).ok_or_else(|| {
            LoadError::InvalidGguf(format!("tensor byte range {start}..{end} is out of bounds"))
        })
    }

    fn linear_host_parts(&self, name: &str) -> Result<LinearHostParts, LoadError> {
        let tensor = self.tensor(name)?;
        let data = self.tensor_slice(tensor)?;
        if tensor.dims.len() != 2 {
            return Err(LoadError::UnexpectedTensor(format!(
                "GGUF linear tensor {name} must be rank-2, got {:?}",
                tensor.dims
            )));
        }
        let logical_cols = tensor.dims[0];
        let logical_rows = tensor.dims[1];
        match tensor.tensor_type {
            GGML_TYPE_GQH3 | GGML_TYPE_GQH2_H | GGML_TYPE_GQH2_C => {
                let rung = model_store::gqh::GqhRung::from_ggml_type(tensor.tensor_type)
                    .ok_or_else(|| {
                        LoadError::UnsupportedDtype(format!(
                            "GGUF linear tensor {name} has unsupported GQH type {}",
                            tensor.tensor_type
                        ))
                    })?;
                let file_row = model_store::gqh::packed_nbytes(rung, 1, logical_cols)
                    .map_err(|e| LoadError::UnexpectedTensor(e.to_string()))?;
                if data.len() != logical_rows * file_row {
                    return Err(LoadError::UnexpectedTensor(format!(
                        "GGUF GQH tensor {name} size {} != {logical_rows}*{file_row}",
                        data.len()
                    )));
                }
                let bytes = model_store::gqh::planarize(rung, logical_rows, logical_cols, data)
                    .map_err(|e| LoadError::UnexpectedTensor(e.to_string()))?;
                let row_bytes = model_store::gqh::device_row_bytes(rung, logical_cols).ok_or_else(
                    || LoadError::UnexpectedTensor(format!("GQH device row cols={logical_cols}")),
                )?;
                Ok(LinearHostParts {
                    dtype: ScalarType::U8,
                    quant_type: tensor.tensor_type as i32,
                    logical_rows,
                    logical_cols,
                    upload_shape: vec![logical_rows, row_bytes],
                    bytes,
                    row_bytes,
                })
            }
            GGML_TYPE_Q8_0 | GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K
            | GGML_TYPE_Q6_K | GGML_TYPE_ROCMFP3_MIX | GGML_TYPE_ROCMFP2_MIX => {
                let quant_type = gguf_linear_quant_type(tensor.tensor_type).ok_or_else(|| {
                    LoadError::UnsupportedDtype(format!(
                        "GGUF linear tensor {name} has unsupported ggml type {}",
                        tensor.tensor_type
                    ))
                })?;
                let row_bytes = ggml_k_row_bytes(quant_type, logical_cols).ok_or_else(|| {
                    LoadError::UnexpectedTensor(format!(
                        "GGUF quantized tensor {name} has invalid K={logical_cols} for type {}",
                        tensor.tensor_type
                    ))
                })?;
                Ok(LinearHostParts {
                    dtype: ScalarType::U8,
                    quant_type,
                    logical_rows,
                    logical_cols,
                    upload_shape: vec![logical_rows, row_bytes],
                    bytes: data.to_vec(),
                    row_bytes,
                })
            }
            GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16 => {
                let bytes = tensor_to_bf16_bytes(tensor.tensor_type, data)?;
                let row_bytes = logical_cols.checked_mul(2).ok_or_else(|| {
                    LoadError::InvalidGguf(format!(
                        "GGUF dense tensor {name} row byte size overflows"
                    ))
                })?;
                Ok(LinearHostParts {
                    dtype: ScalarType::BF16,
                    quant_type: 0,
                    logical_rows,
                    logical_cols,
                    upload_shape: vec![logical_rows, logical_cols],
                    bytes,
                    row_bytes,
                })
            }
            other => Err(LoadError::UnsupportedDtype(format!(
                "GGUF linear tensor {name} has unsupported ggml type {other}"
            ))),
        }
    }
}

struct GgufCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GgufCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], LoadError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| LoadError::InvalidGguf(format!("cursor overflow at {}", self.pos)))?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| {
            LoadError::InvalidGguf(format!(
                "unexpected EOF while reading {len} bytes at {}",
                self.pos
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, LoadError> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, LoadError> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, LoadError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, LoadError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_u64_usize(&mut self, what: &str) -> Result<usize, LoadError> {
        usize::try_from(self.read_u64()?).map_err(|_| {
            LoadError::InvalidGguf(format!("{what} does not fit in usize on this platform"))
        })
    }

    fn read_string(&mut self) -> Result<String, LoadError> {
        let len = self.read_u64_usize("string length")?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|err| {
            LoadError::InvalidGguf(format!(
                "invalid UTF-8 string in metadata/tensor name: {err}"
            ))
        })
    }
}

fn read_gguf_metadata_value(
    cursor: &mut GgufCursor<'_>,
    value_type: u32,
    key: &str,
    alignment: &mut usize,
) -> Result<(), LoadError> {
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
        }
        6 => {
            let _ = cursor.read_u32()?;
        }
        7 => {
            let _ = cursor.read_u8()?;
        }
        8 => {
            let _ = cursor.read_string()?;
        }
        9 => {
            let elem_type = cursor.read_u32()?;
            let len = cursor.read_u64_usize("array length")?;
            for _ in 0..len {
                read_gguf_metadata_value(cursor, elem_type, "", alignment)?;
            }
        }
        10 | 11 => {
            let value = cursor.read_u64()?;
            if key == "general.alignment" {
                *alignment = usize::try_from(value).map_err(|_| {
                    LoadError::InvalidGguf(
                        "general.alignment does not fit in usize on this platform".into(),
                    )
                })?;
            }
        }
        12 => {
            let _ = cursor.read_u64()?;
        }
        other => {
            return Err(LoadError::InvalidGguf(format!(
                "unsupported metadata value type {other} for key {key}"
            )));
        }
    }
    Ok(())
}

fn gguf_tensor_nbytes(dims: &[usize], tensor_type: u32) -> Result<usize, LoadError> {
    let elems = dims.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| LoadError::InvalidGguf("tensor element count overflows".into()))
    })?;
    match tensor_type {
        GGML_TYPE_F32 => elems
            .checked_mul(4)
            .ok_or_else(|| LoadError::InvalidGguf("F32 tensor byte size overflows".into())),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => elems
            .checked_mul(2)
            .ok_or_else(|| LoadError::InvalidGguf("16-bit tensor byte size overflows".into())),
        GGML_TYPE_Q8_0 | GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K
        | GGML_TYPE_Q6_K | GGML_TYPE_ROCMFP3_MIX | GGML_TYPE_ROCMFP2_MIX => {
            if dims.len() != 2 {
                return Err(LoadError::UnexpectedTensor(format!(
                    "quantized GGUF tensor must be rank-2, got {dims:?}"
                )));
            }
            let quant_type = gguf_linear_quant_type(tensor_type).ok_or_else(|| {
                LoadError::UnsupportedDtype(format!("unsupported GGUF tensor type {tensor_type}"))
            })?;
            let row_bytes = ggml_k_row_bytes(quant_type, dims[0]).ok_or_else(|| {
                LoadError::UnexpectedTensor(format!(
                    "quantized GGUF tensor has invalid K={} for type {tensor_type}",
                    dims[0]
                ))
            })?;
            dims[1].checked_mul(row_bytes).ok_or_else(|| {
                LoadError::InvalidGguf("quantized GGUF tensor byte size overflows".into())
            })
        }
        GGML_TYPE_GQH3 | GGML_TYPE_GQH2_H | GGML_TYPE_GQH2_C => {
            if dims.len() != 2 {
                return Err(LoadError::UnexpectedTensor(format!(
                    "GQH GGUF tensor must be rank-2, got {dims:?}"
                )));
            }
            let rung = model_store::gqh::GqhRung::from_ggml_type(tensor_type).ok_or_else(|| {
                LoadError::UnsupportedDtype(format!("unsupported GQH qtype {tensor_type}"))
            })?;
            // GGUF ne[0] is the row length (in_features / cols).
            model_store::gqh::packed_nbytes(rung, dims[1], dims[0]).map_err(|e| {
                LoadError::UnexpectedTensor(e.to_string())
            })
        }
        other => Err(LoadError::UnsupportedDtype(format!(
            "unsupported GGUF tensor type {other}"
        ))),
    }
}

fn gguf_linear_quant_type(tensor_type: u32) -> Option<i32> {
    match tensor_type {
        GGML_TYPE_Q8_0 => Some(LOWBIT_GGML_Q8_0),
        GGML_TYPE_Q2_K => Some(LOWBIT_GGML_Q2_K),
        GGML_TYPE_Q3_K => Some(LOWBIT_GGML_Q3_K),
        GGML_TYPE_Q4_K => Some(LOWBIT_GGML_Q4_K),
        GGML_TYPE_Q5_K => Some(LOWBIT_GGML_Q5_K),
        GGML_TYPE_Q6_K => Some(LOWBIT_GGML_Q6_K),
        GGML_TYPE_ROCMFP3_MIX => Some(LOWBIT_ROCMFP3_MIX),
        GGML_TYPE_ROCMFP2_MIX => Some(LOWBIT_ROCMFP2_MIX),
        _ => None,
    }
}

fn tensor_to_bf16_bytes(tensor_type: u32, data: &[u8]) -> Result<Vec<u8>, LoadError> {
    match tensor_type {
        GGML_TYPE_BF16 => Ok(data.to_vec()),
        GGML_TYPE_F16 => {
            if data.len() % 2 != 0 {
                return Err(LoadError::InvalidGguf(
                    "F16 tensor has odd byte length".into(),
                ));
            }
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                let value = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                out.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
            Ok(out)
        }
        GGML_TYPE_F32 => {
            if data.len() % 4 != 0 {
                return Err(LoadError::InvalidGguf(
                    "F32 tensor byte length is not divisible by 4".into(),
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
            Ok(out)
        }
        other => Err(LoadError::UnsupportedDtype(format!(
            "cannot convert GGUF tensor type {other} to BF16"
        ))),
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, LoadError> {
    let padding_base = value.checked_add(alignment - 1).ok_or_else(|| {
        LoadError::InvalidGguf(format!(
            "alignment overflow for value={value} alignment={alignment}"
        ))
    })?;
    Ok((padding_base / alignment) * alignment)
}
