use std::borrow::Cow;
use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::c_int;
use std::ffi::c_void;
use std::fs::File;
use std::path::{Path, PathBuf};

use gpu_hal::{
    copy_h2d, copy_h2d_async, current_backend, sync, Backend, GpuBuffer, GpuStream,
    RegisteredHostBuffer, ScalarType, VirtualAllocationRole, VirtualArena, VirtualBacking,
};
use memmap2::Mmap;

use crate::manifest::{LayoutTag, Manifest, TensorMeta};
use crate::Error;

const FLM_MAGIC: &[u8; 8] = b"FLM1\0\0\0\0";
const FLM_SUPERBLOCK_SIZE: usize = 4096;
const FLM_INDEX_RECORD_SIZE: usize = 64;
const FLM_SHARD_DESC_SIZE: usize = 24;
const FLM_HASH_RECORD_SIZE: usize = 40;
const FLM_DTYPE_FP32: u16 = 0;
const FLM_DTYPE_FP16: u16 = 1;
const FLM_DTYPE_BF16: u16 = 2;
const FLM_DTYPE_FP8_E4M3: u16 = 3;
const FLM_DTYPE_UINT8: u16 = 4;
const FLM_DTYPE_INT32: u16 = 5;
const FLM_DTYPE_INT64: u16 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HostRegistrationRange {
    ptr: *mut c_void,
    len: usize,
    data_offset: usize,
}

#[cfg(unix)]
unsafe extern "C" {
    fn getpagesize() -> c_int;
}

fn host_page_size() -> usize {
    #[cfg(unix)]
    {
        let page_size = unsafe { getpagesize() };
        if page_size > 0 {
            return page_size as usize;
        }
    }
    4096
}

fn round_up_to(value: usize, align: usize) -> Result<usize, Error> {
    if align == 0 {
        return Err(Error::Other("alignment must be > 0".into()));
    }
    let remainder = value % align;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(align - remainder)
        .ok_or_else(|| Error::Other("round_up_to overflow".into()))
}

fn host_registration_range_for_mmap_slice(
    mmap_start: usize,
    mmap_len: usize,
    data_start: usize,
    data_len: usize,
    page_size: usize,
) -> Result<HostRegistrationRange, Error> {
    if mmap_len == 0 {
        return Err(Error::Other(
            "host registration requires a non-empty mmap backing".into(),
        ));
    }
    if data_len == 0 {
        return Err(Error::Other(
            "host registration requires a non-empty data slice".into(),
        ));
    }
    if page_size == 0 {
        return Err(Error::Other(
            "host registration requires page_size > 0".into(),
        ));
    }
    let mmap_end = mmap_start
        .checked_add(mmap_len)
        .ok_or_else(|| Error::Other("mmap backing range overflows".into()))?;
    let data_end = data_start
        .checked_add(data_len)
        .ok_or_else(|| Error::Other("data slice range overflows".into()))?;
    if data_start < mmap_start || data_end > mmap_end {
        return Err(Error::Other(format!(
            "data slice [{data_start:#x}, {data_end:#x}) is outside mmap backing \
             [{mmap_start:#x}, {mmap_end:#x})"
        )));
    }
    let register_start = data_start - (data_start % page_size);
    let register_end = round_up_to(data_end, page_size)?;
    let mmap_page_end = round_up_to(mmap_end, page_size)?;
    if register_start < mmap_start - (mmap_start % page_size) || register_end > mmap_page_end {
        return Err(Error::Other(format!(
            "host registration range [{register_start:#x}, {register_end:#x}) is outside \
             page-rounded mmap backing ending at {mmap_page_end:#x}"
        )));
    }
    Ok(HostRegistrationRange {
        ptr: register_start as *mut c_void,
        len: register_end - register_start,
        data_offset: data_start - register_start,
    })
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FlmLoadOptions {
    /// Verify BLAKE3 payload hashes for FLM tensors that reference the block
    /// hash table. This reads and hashes the referenced payload bytes during
    /// open, so callers should opt in when integrity matters more than startup
    /// latency.
    pub verify_block_hashes: bool,

    /// Expose FLM logical INT4 weights through loadable tensor aliases. Stage 3
    /// logical/storage bindings are preferred; manifest groups remain as a
    /// transition fallback for older fixtures.
    pub flm_int4_logical_aliases: bool,
}

#[derive(Debug, Clone)]
struct FlmSuperblock {
    tensor_count: usize,
    index_offset: usize,
    index_len: usize,
    metadata_offset: usize,
    metadata_len: usize,
    hashtable_offset: usize,
    hashtable_len: usize,
    shard_table_offset: usize,
    shard_count: usize,
    runtime_dir_offset: usize,
    runtime_dir_len: usize,
}

#[derive(Debug, Clone)]
struct FlmShard {
    offset: u64,
    length: u64,
}

#[derive(Debug, Clone)]
struct FlmIndexEntry {
    name: String,
    shape: Vec<usize>,
    dtype: u16,
    codec: u8,
    file_offset: u64,
    stored_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorUploadView {
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CtSymInt4Bf16Fallback {
    packed_tensor: String,
    scale_tensor: String,
    shape: Vec<usize>,
    group_size: usize,
}

/// A memory-mapped baked weight store for fast GPU loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorStorageSourceKind {
    BakedWeights,
    FlmContainer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStorageExtent {
    pub source_kind: TensorStorageSourceKind,
    pub source_path: PathBuf,
    pub name: String,
    pub file_offset: u64,
    pub byte_len: u64,
    pub storage_dtype: String,
    pub storage_shape: Vec<usize>,
    pub layout: LayoutTag,
    pub upload_dtype: String,
    pub upload_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStorageRange {
    /// Full direct file-backed tensor extent this range comes from.
    pub extent: TensorStorageExtent,
    /// Byte offset inside the tensor payload.
    pub tensor_byte_offset: u64,
    /// Absolute byte offset in the source file for this range.
    pub file_offset: u64,
    /// Number of bytes in this range.
    pub byte_len: u64,
}

pub struct BakedStore {
    _mmap: Mmap,
    data: *const u8,
    data_len: usize,
    source_kind: TensorStorageSourceKind,
    source_path: PathBuf,
    index: HashMap<String, TensorMeta>,
    synthetic: HashMap<String, Vec<u8>>,
    upload_views: HashMap<String, TensorUploadView>,
    ct_int4_bf16_fallbacks: HashMap<String, CtSymInt4Bf16Fallback>,
    runtime: Option<crate::flm::FlmRuntimeDirectory>,
}

// Safety: the mmap is immutable and lives as long as BakedStore.
unsafe impl Send for BakedStore {}
unsafe impl Sync for BakedStore {}

pub fn read_flm_runtime_identity(
    path: &Path,
) -> Result<Option<crate::flm::FlmRuntimeIdentity>, Error> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let sb = flm_parse_superblock(&mmap)?;
    match (sb.runtime_dir_offset, sb.runtime_dir_len) {
        (0, 0) => Ok(None),
        (0, len) => Err(Error::Other(format!(
            "FLM runtime directory length is {len} but offset is zero"
        ))),
        (offset, 0) => Err(Error::Other(format!(
            "FLM runtime directory offset is {offset} but length is zero"
        ))),
        (offset, len) => {
            let runtime = read_exact_range(&mmap, offset, len, "FLM runtime directory")?;
            crate::flm::FlmRuntimeDirectory::parse_identity(runtime).map(Some)
        }
    }
}

fn parse_dtype(name: &str) -> Result<ScalarType, Error> {
    ScalarType::from_name(name).ok_or_else(|| Error::UnsupportedDtype(name.to_string()))
}

fn gpu_upload_shape(meta: &TensorMeta) -> Result<Vec<usize>, Error> {
    if matches!(meta.layout, LayoutTag::Int4Quantized) {
        let byte_len = usize::try_from(meta.byte_len).map_err(|_| {
            Error::Other(format!(
                "tensor '{}' byte_len={} does not fit usize",
                meta.name, meta.byte_len
            ))
        })?;
        if meta.shape.len() == 2 {
            let rows = meta.shape[0];
            if rows == 0 {
                return Err(Error::Other(format!(
                    "tensor '{}' INT4 upload shape has zero rows",
                    meta.name
                )));
            }
            if byte_len % rows != 0 {
                return Err(Error::Other(format!(
                    "tensor '{}' INT4 byte_len={} is not divisible by rows={}",
                    meta.name, meta.byte_len, rows
                )));
            }
            return Ok(vec![rows, byte_len / rows]);
        }
        return Ok(vec![byte_len]);
    }
    Ok(meta.shape.clone())
}

fn read_exact_range<'a>(
    buf: &'a [u8],
    offset: usize,
    len: usize,
    what: &str,
) -> Result<&'a [u8], Error> {
    let end = offset.checked_add(len).ok_or_else(|| {
        Error::Other(format!(
            "{what}: range overflows (offset={offset}, len={len})"
        ))
    })?;
    if end > buf.len() {
        return Err(Error::Other(format!(
            "{what}: range [{offset}, {end}) exceeds file length {}",
            buf.len()
        )));
    }
    Ok(&buf[offset..end])
}

fn read_u16(buf: &[u8], offset: usize, what: &str) -> Result<u16, Error> {
    let bytes: [u8; 2] = read_exact_range(buf, offset, 2, what)?
        .try_into()
        .expect("slice length checked");
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(buf: &[u8], offset: usize, what: &str) -> Result<u32, Error> {
    let bytes: [u8; 4] = read_exact_range(buf, offset, 4, what)?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(buf: &[u8], offset: usize, what: &str) -> Result<u64, Error> {
    let bytes: [u8; 8] = read_exact_range(buf, offset, 8, what)?
        .try_into()
        .expect("slice length checked");
    Ok(u64::from_le_bytes(bytes))
}

fn u64_to_usize(value: u64, what: &str) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::Other(format!("{what}: {value} does not fit usize")))
}

fn flm_crc64_ecma(data: &[u8]) -> u64 {
    let mut crc = 0u64;
    const POLY: u64 = 0x42F0_E1EB_A9EA_3693;
    for byte in data {
        crc ^= (*byte as u64) << 56;
        for _ in 0..8 {
            crc = if crc & (1 << 63) != 0 {
                (crc << 1) ^ POLY
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn flm_head_crc64(head: &[u8]) -> Result<u64, Error> {
    let mut checked = head.to_vec();
    read_exact_range(&checked, 144, 8, "FLM head CRC64 field")?;
    checked[144..152].fill(0);
    Ok(flm_crc64_ecma(&checked))
}

fn flm_parse_superblock(buf: &[u8]) -> Result<FlmSuperblock, Error> {
    let sb = read_exact_range(buf, 0, FLM_SUPERBLOCK_SIZE, "FLM superblock")?;
    if &sb[..8] != FLM_MAGIC {
        return Err(Error::Other(format!(
            "bad FLM magic: expected {:?}, got {:?}",
            FLM_MAGIC,
            &sb[..8]
        )));
    }
    let version = read_u32(sb, 8, "FLM format_version")?;
    if version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM format_version {version}; expected 1"
        )));
    }
    let tensor_count = u64_to_usize(read_u64(sb, 16, "FLM tensor_count")?, "FLM tensor_count")?;
    let index_offset = u64_to_usize(read_u64(sb, 24, "FLM index_offset")?, "FLM index_offset")?;
    let index_len = u64_to_usize(read_u64(sb, 32, "FLM index_len")?, "FLM index_len")?;
    let metadata_offset = u64_to_usize(
        read_u64(sb, 40, "FLM metadata_offset")?,
        "FLM metadata_offset",
    )?;
    let metadata_len = u64_to_usize(read_u64(sb, 48, "FLM metadata_len")?, "FLM metadata_len")?;
    let hashtable_offset = u64_to_usize(
        read_u64(sb, 72, "FLM hashtable_offset")?,
        "FLM hashtable_offset",
    )?;
    let hashtable_len = u64_to_usize(read_u64(sb, 80, "FLM hashtable_len")?, "FLM hashtable_len")?;
    let shard_table_offset = u64_to_usize(
        read_u64(sb, 88, "FLM shard_table_offset")?,
        "FLM shard_table_offset",
    )?;
    if shard_table_offset < FLM_SUPERBLOCK_SIZE {
        return Err(Error::Other(format!(
            "malformed FLM head CRC region: shard_table_offset={shard_table_offset}"
        )));
    }
    let stored_head_crc64 = read_u64(sb, 144, "FLM head_crc64")?;
    let head = read_exact_range(buf, 0, shard_table_offset, "FLM head CRC64 region")?;
    let actual_head_crc64 = flm_head_crc64(head)?;
    if stored_head_crc64 != actual_head_crc64 {
        return Err(Error::Other(format!(
            "FLM head CRC mismatch: stored={stored_head_crc64}, computed={actual_head_crc64}"
        )));
    }
    let shard_count = read_u32(sb, 96, "FLM shard_count")? as usize;
    let runtime_dir_offset = u64_to_usize(
        read_u64(sb, 168, "FLM runtime_dir_offset")?,
        "FLM runtime_dir_offset",
    )?;
    let runtime_dir_len = u64_to_usize(
        read_u64(sb, 176, "FLM runtime_dir_len")?,
        "FLM runtime_dir_len",
    )?;

    let expected_index_len = tensor_count
        .checked_mul(FLM_INDEX_RECORD_SIZE)
        .ok_or_else(|| Error::Other("FLM index length overflows".to_string()))?;
    if index_len < expected_index_len {
        return Err(Error::Other(format!(
            "FLM index_len {index_len} shorter than tensor_count stride {expected_index_len}"
        )));
    }

    Ok(FlmSuperblock {
        tensor_count,
        index_offset,
        index_len,
        metadata_offset,
        metadata_len,
        hashtable_offset,
        hashtable_len,
        shard_table_offset,
        shard_count,
        runtime_dir_offset,
        runtime_dir_len,
    })
}

fn flm_read_string_table(buf: &[u8], sb: &FlmSuperblock) -> Result<Vec<String>, Error> {
    let meta = read_exact_range(buf, sb.metadata_offset, sb.metadata_len, "FLM metadata")?;
    if meta.len() < 4 {
        return Err(Error::Other(
            "FLM metadata missing string table length".to_string(),
        ));
    }
    let mut offset = 0usize;
    let count = read_u32(meta, offset, "FLM string count")? as usize;
    offset += 4;
    let mut strings = Vec::with_capacity(count);
    for idx in 0..count {
        let len = read_u32(meta, offset, "FLM string length")? as usize;
        offset += 4;
        let bytes = read_exact_range(meta, offset, len, "FLM string bytes")?;
        offset += len;
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Error::Other(format!("FLM string table entry {idx} is not UTF-8: {e}")))?;
        strings.push(text.to_string());
    }
    Ok(strings)
}

fn flm_read_shards(buf: &[u8], sb: &FlmSuperblock) -> Result<HashMap<u32, FlmShard>, Error> {
    let table_len = sb
        .shard_count
        .checked_mul(FLM_SHARD_DESC_SIZE)
        .ok_or_else(|| Error::Other("FLM shard table length overflows".to_string()))?;
    let table = read_exact_range(buf, sb.shard_table_offset, table_len, "FLM shard table")?;
    let mut shards = HashMap::with_capacity(sb.shard_count);
    for idx in 0..sb.shard_count {
        let off = idx * FLM_SHARD_DESC_SIZE;
        let shard_id = read_u32(table, off, "FLM shard_id")?;
        let file_offset = read_u64(table, off + 4, "FLM shard file offset")?;
        let length = read_u64(table, off + 12, "FLM shard length")?;
        let end = file_offset.checked_add(length).ok_or_else(|| {
            Error::Other(format!(
                "FLM shard {shard_id} range overflows (offset={file_offset}, len={length})"
            ))
        })?;
        if end > buf.len() as u64 {
            return Err(Error::Other(format!(
                "FLM shard {shard_id} extends past end of file (offset={file_offset}, len={length}, file_len={})",
                buf.len()
            )));
        }
        shards.insert(
            shard_id,
            FlmShard {
                offset: file_offset,
                length,
            },
        );
    }
    Ok(shards)
}

fn flm_dtype_name(dtype: u16) -> Result<&'static str, Error> {
    match dtype {
        FLM_DTYPE_FP32 => Ok("f32"),
        FLM_DTYPE_FP16 => Ok("f16"),
        FLM_DTYPE_BF16 => Ok("bf16"),
        FLM_DTYPE_FP8_E4M3 => Ok("f8_e4m3"),
        FLM_DTYPE_UINT8 => Ok("u8"),
        // SuperSonic's HAL only needs byte width for packed INT4 payloads; u32
        // preserves the 4-byte element layout of compressed-tensors int32.
        FLM_DTYPE_INT32 => Ok("u32"),
        FLM_DTYPE_INT64 => Ok("i64"),
        other => Err(Error::UnsupportedDtype(format!("FLM dtype id {other}"))),
    }
}

fn flm_read_index_entries(
    buf: &[u8],
    sb: &FlmSuperblock,
    strings: &[String],
    shards: &HashMap<u32, FlmShard>,
) -> Result<HashMap<String, FlmIndexEntry>, Error> {
    let index_blob = read_exact_range(buf, sb.index_offset, sb.index_len, "FLM tensor index")?;
    let mut entries = HashMap::with_capacity(sb.tensor_count);
    for idx in 0..sb.tensor_count {
        let off = idx * FLM_INDEX_RECORD_SIZE;
        let rec = read_exact_range(index_blob, off, FLM_INDEX_RECORD_SIZE, "FLM tensor record")?;
        let name_id = read_u32(rec, 0, "FLM tensor name_id")? as usize;
        let name = strings.get(name_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {idx} references missing string table id {name_id}"
            ))
        })?;
        let shard_id = read_u32(rec, 8, "FLM tensor shard_id")?;
        let shard_offset = read_u64(rec, 12, "FLM tensor shard_offset")?;
        let stored_len = read_u64(rec, 20, "FLM tensor stored_len")?;
        let dtype = read_u16(rec, 36, "FLM tensor dtype")?;
        let codec = rec[40];
        let n_dims = rec[41] as usize;
        let inline_dims = n_dims.min(4);
        let mut shape = Vec::with_capacity(inline_dims);
        for dim_idx in 0..inline_dims {
            shape.push(read_u32(rec, 42 + dim_idx * 4, "FLM tensor shape")? as usize);
        }
        let shard = shards.get(&shard_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {name} references missing shard {shard_id}"
            ))
        })?;
        let shard_end = shard_offset.checked_add(stored_len).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {name} shard range overflows (offset={shard_offset}, len={stored_len})"
            ))
        })?;
        if shard_end > shard.length {
            return Err(Error::Other(format!(
                "FLM tensor {name} extends past shard {shard_id} (offset={shard_offset}, len={stored_len}, shard_len={})",
                shard.length
            )));
        }
        let file_offset = shard.offset.checked_add(shard_offset).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {name} file offset overflows (shard_offset={}, tensor_offset={shard_offset})",
                shard.offset
            ))
        })?;

        entries.insert(
            name.clone(),
            FlmIndexEntry {
                name: name.clone(),
                shape,
                dtype,
                codec,
                file_offset,
                stored_len,
            },
        );
    }
    Ok(entries)
}

fn flm_build_index(
    entries: &HashMap<String, FlmIndexEntry>,
) -> Result<HashMap<String, TensorMeta>, Error> {
    let mut index = HashMap::with_capacity(entries.len());
    for entry in entries.values() {
        let dtype_name = if entry.codec == 0 {
            flm_dtype_name(entry.dtype)?
        } else {
            // Coded tensors are exposed as raw byte payloads until a runtime
            // alias maps them onto a native SuperSonic layout.
            "u8"
        };

        index.insert(
            entry.name.clone(),
            TensorMeta {
                name: entry.name.clone(),
                shape: entry.shape.clone(),
                dtype: dtype_name.to_string(),
                layout: LayoutTag::Raw,
                offset: entry.file_offset,
                byte_len: entry.stored_len,
            },
        );
    }
    Ok(index)
}

fn flm_verify_block_hashes(
    buf: &[u8],
    sb: &FlmSuperblock,
    strings: &[String],
    shards: &HashMap<u32, FlmShard>,
) -> Result<(), Error> {
    if sb.hashtable_offset == 0 && sb.hashtable_len != 0 {
        return Err(Error::Other(format!(
            "FLM block hash table length is {} but offset is zero",
            sb.hashtable_len
        )));
    }
    if sb.hashtable_offset != 0 && sb.hashtable_len == 0 {
        return Err(Error::Other(format!(
            "FLM block hash table offset is {} but length is zero",
            sb.hashtable_offset
        )));
    }
    if sb.hashtable_len % FLM_HASH_RECORD_SIZE != 0 {
        return Err(Error::Other(format!(
            "FLM block hash table length {} is not a multiple of {FLM_HASH_RECORD_SIZE}",
            sb.hashtable_len
        )));
    }

    let hash_records = sb.hashtable_len / FLM_HASH_RECORD_SIZE;
    let hashtable = if sb.hashtable_len == 0 {
        &[][..]
    } else {
        read_exact_range(
            buf,
            sb.hashtable_offset,
            sb.hashtable_len,
            "FLM block hash table",
        )?
    };
    let index_blob = read_exact_range(buf, sb.index_offset, sb.index_len, "FLM tensor index")?;

    for idx in 0..sb.tensor_count {
        let off = idx * FLM_INDEX_RECORD_SIZE;
        let rec = read_exact_range(index_blob, off, FLM_INDEX_RECORD_SIZE, "FLM tensor record")?;
        let block_hash_idx = read_u32(rec, 58, "FLM tensor block_hash_idx")? as usize;
        if block_hash_idx == 0 {
            continue;
        }
        if block_hash_idx > hash_records {
            return Err(Error::Other(format!(
                "FLM tensor {idx} references block_hash_idx {block_hash_idx}, but hash table has {hash_records} records"
            )));
        }

        let name_id = read_u32(rec, 0, "FLM tensor name_id")? as usize;
        let name = strings.get(name_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {idx} references missing string table id {name_id}"
            ))
        })?;
        let shard_id = read_u32(rec, 8, "FLM tensor shard_id")?;
        let shard_offset = read_u64(rec, 12, "FLM tensor shard_offset")?;
        let stored_len = read_u64(rec, 20, "FLM tensor stored_len")?;
        let hash_off = (block_hash_idx - 1) * FLM_HASH_RECORD_SIZE;
        let expected_digest = read_exact_range(hashtable, hash_off, 32, "FLM block hash digest")?;
        let expected_len = read_u64(hashtable, hash_off + 32, "FLM block hash stored_len")?;
        if expected_len != stored_len {
            return Err(Error::Other(format!(
                "FLM tensor {name} block_hash_idx {block_hash_idx} stored_len mismatch: index has {stored_len}, hash table has {expected_len}"
            )));
        }

        let shard = shards.get(&shard_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {name} references missing shard {shard_id}"
            ))
        })?;
        let shard_end = shard_offset.checked_add(stored_len).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {name} shard range overflows (offset={shard_offset}, len={stored_len})"
            ))
        })?;
        if shard_end > shard.length {
            return Err(Error::Other(format!(
                "FLM tensor {name} extends past shard {shard_id} (offset={shard_offset}, len={stored_len}, shard_len={})",
                shard.length
            )));
        }
        let file_offset = shard.offset.checked_add(shard_offset).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor {name} file offset overflows (shard_offset={}, tensor_offset={shard_offset})",
                shard.offset
            ))
        })?;
        let payload_offset = u64_to_usize(file_offset, "FLM tensor payload offset")?;
        let payload_len = u64_to_usize(stored_len, "FLM tensor stored_len")?;
        let payload = read_exact_range(buf, payload_offset, payload_len, "FLM tensor payload")?;
        let actual = blake3::hash(payload);
        if actual.as_bytes().as_slice() != expected_digest {
            return Err(Error::Other(format!(
                "FLM tensor {name} block_hash_idx {block_hash_idx} hash mismatch"
            )));
        }
    }

    Ok(())
}

fn bf16_eight_bytes(len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..(len / 2) {
        bytes.extend_from_slice(&[0x00, 0x41]);
    }
    bytes
}

fn flm_ct_sym_int4_to_bf16(
    packed_bytes: &[u8],
    packed_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
    logical_shape: &[usize],
    group_size: usize,
    tensor_name: &str,
) -> Result<Vec<u8>, Error> {
    if logical_shape.len() != 2 {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} logical shape must be rank 2, got {logical_shape:?}"
        )));
    }
    if packed_shape.len() != 2 {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} packed shape must be rank 2, got {packed_shape:?}"
        )));
    }
    if scale_shape.len() != 2 {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} scale shape must be rank 2, got {scale_shape:?}"
        )));
    }
    if group_size == 0 {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} has zero group size"
        )));
    }
    let rows = logical_shape[0];
    let cols = logical_shape[1];
    let packed_rows = packed_shape[0];
    let packed_cols = packed_shape[1];
    if packed_rows != rows {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} packed rows {} != logical rows {rows}",
            packed_rows
        )));
    }
    let required_packed_cols = cols.div_ceil(8);
    if packed_cols < required_packed_cols {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} packed columns {} cannot cover logical cols {cols}",
            packed_cols
        )));
    }
    let expected_packed_len = packed_rows
        .checked_mul(packed_cols)
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| {
            Error::Other(format!(
                "FLM CT INT4 fallback tensor {tensor_name} packed byte length overflows"
            ))
        })?;
    if packed_bytes.len() != expected_packed_len {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} packed byte length {} != expected {expected_packed_len}",
            packed_bytes.len()
        )));
    }

    let scale_rows = scale_shape[0];
    let scale_cols = scale_shape[1];
    if scale_rows != rows {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} scale rows {} != logical rows {rows}",
            scale_rows
        )));
    }
    let required_scale_cols = cols.div_ceil(group_size);
    if scale_cols < required_scale_cols {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} scale columns {} cannot cover {cols} cols with group size {group_size}",
            scale_cols
        )));
    }
    let expected_scale_len = scale_rows
        .checked_mul(scale_cols)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| {
            Error::Other(format!(
                "FLM CT INT4 fallback tensor {tensor_name} scale byte length overflows"
            ))
        })?;
    if scale_bytes.len() != expected_scale_len {
        return Err(Error::Other(format!(
            "FLM CT INT4 fallback tensor {tensor_name} scale byte length {} != expected {expected_scale_len}",
            scale_bytes.len()
        )));
    }

    let out_len = rows
        .checked_mul(cols)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| {
            Error::Other(format!(
                "FLM CT INT4 fallback tensor {tensor_name} BF16 byte length overflows"
            ))
        })?;
    let mut out = Vec::with_capacity(out_len);
    for row in 0..rows {
        for col in 0..cols {
            let packed_index = row * packed_cols + col / 8;
            let packed_offset = packed_index * 4;
            let word = u32::from_le_bytes([
                packed_bytes[packed_offset],
                packed_bytes[packed_offset + 1],
                packed_bytes[packed_offset + 2],
                packed_bytes[packed_offset + 3],
            ]);
            let nibble = ((word >> ((col % 8) * 4)) & 0x0f) as i32;
            let code = nibble - 8;
            let scale_index = row * scale_cols + col / group_size;
            let scale_offset = scale_index * 2;
            let scale = half::bf16::from_le_bytes([
                scale_bytes[scale_offset],
                scale_bytes[scale_offset + 1],
            ])
            .to_f32();
            let value = code as f32 * scale;
            out.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
        }
    }
    Ok(out)
}

fn manifest_row_shape(row: &crate::flm::FlmTensorManifestRow) -> Vec<usize> {
    row.shape[..row.rank as usize]
        .iter()
        .map(|&dim| dim as usize)
        .collect()
}

fn validate_flm_manifest_against_index(
    runtime: &crate::flm::FlmRuntimeDirectory,
    entries: &HashMap<String, FlmIndexEntry>,
) -> Result<(), Error> {
    for row in &runtime.tensor_manifest().rows {
        if row.flags & crate::flm::MANIFEST_FLAG_DERIVED_ALIAS != 0 {
            continue;
        }
        let Some(entry) = entries.get(&row.name) else {
            if row.flags & crate::flm::MANIFEST_FLAG_REQUIRED != 0 {
                return Err(Error::Other(format!(
                    "FLM manifest required tensor {} missing from index",
                    row.name
                )));
            }
            continue;
        };
        if entry.shape != manifest_row_shape(row) {
            return Err(Error::Other(format!(
                "FLM manifest shape mismatch for {}",
                row.name
            )));
        }
        if row.dtype != entry.dtype {
            return Err(Error::Other(format!(
                "FLM manifest dtype mismatch for {}",
                row.name
            )));
        }
        if row.codec_id != entry.codec {
            return Err(Error::Other(format!(
                "FLM manifest codec mismatch for {}",
                row.name
            )));
        }
    }
    Ok(())
}

#[derive(Default)]
struct ManifestInt4Group<'a> {
    packed: Option<&'a crate::flm::FlmTensorManifestRow>,
    scale: Option<&'a crate::flm::FlmTensorManifestRow>,
    required: bool,
}

fn add_manifest_int4_aliases(
    runtime: &crate::flm::FlmRuntimeDirectory,
    index: &mut HashMap<String, TensorMeta>,
    synthetic: &mut HashMap<String, Vec<u8>>,
) -> Result<(), Error> {
    let mut groups: HashMap<u32, ManifestInt4Group<'_>> = HashMap::new();
    for row in &runtime.tensor_manifest().rows {
        if row.group_id == 0 {
            continue;
        }
        let group = groups.entry(row.group_id).or_default();
        group.required |= row.flags & crate::flm::MANIFEST_FLAG_REQUIRED != 0;
        match row.companion_kind {
            crate::flm::MANIFEST_COMPANION_PACKED => group.packed = Some(row),
            crate::flm::MANIFEST_COMPANION_SCALE => group.scale = Some(row),
            _ => {}
        }
    }

    for (group_id, group) in groups {
        let (Some(packed_row), Some(scale_row)) = (group.packed, group.scale) else {
            if group.required {
                return Err(Error::Other(format!(
                    "FLM manifest required INT4 group {group_id} missing packed or scale tensor"
                )));
            }
            continue;
        };

        let Some(base) = packed_row.name.strip_suffix(".weight_packed") else {
            return Err(Error::Other(format!(
                "FLM manifest INT4 packed tensor {} does not end with .weight_packed",
                packed_row.name
            )));
        };
        let Some(packed_meta) = index.get(&packed_row.name).cloned() else {
            if group.required {
                return Err(Error::Other(format!(
                    "FLM manifest required INT4 packed tensor {} missing from index",
                    packed_row.name
                )));
            }
            continue;
        };
        let Some(scale_meta) = index.get(&scale_row.name).cloned() else {
            if group.required {
                return Err(Error::Other(format!(
                    "FLM manifest required INT4 scale tensor {} missing from index",
                    scale_row.name
                )));
            }
            continue;
        };
        if packed_meta.shape.len() != 2 {
            return Err(Error::Other(format!(
                "FLM manifest INT4 packed tensor {} shape must be rank 2",
                packed_row.name
            )));
        }

        let alias_weight = format!("{base}.weight");
        index
            .entry(alias_weight.clone())
            .or_insert_with(|| TensorMeta {
                name: alias_weight,
                shape: vec![packed_meta.shape[0], packed_meta.shape[1] * 4],
                dtype: "u8".to_string(),
                layout: LayoutTag::Int4Quantized,
                offset: packed_meta.offset,
                byte_len: packed_meta.byte_len,
            });

        let alias_scale = format!("{base}.weight_int4_scale");
        let alias_scale_dtype = flm_dtype_name(scale_row.logical_dtype)?;
        index
            .entry(alias_scale.clone())
            .or_insert_with(|| TensorMeta {
                name: alias_scale,
                shape: scale_meta.shape.clone(),
                dtype: alias_scale_dtype.to_string(),
                layout: LayoutTag::Raw,
                offset: scale_meta.offset,
                byte_len: scale_meta.byte_len,
            });

        let alias_zero = format!("{base}.weight_int4_zero");
        index
            .entry(alias_zero.clone())
            .or_insert_with(|| TensorMeta {
                name: alias_zero.clone(),
                shape: scale_meta.shape.clone(),
                dtype: "bf16".to_string(),
                layout: LayoutTag::Raw,
                offset: 0,
                byte_len: scale_meta.byte_len,
            });
        synthetic
            .entry(alias_zero)
            .or_insert_with(|| bf16_eight_bytes(scale_meta.byte_len as usize));
    }
    Ok(())
}

fn add_stage3_raw_value_aliases(
    runtime: &crate::flm::FlmRuntimeDirectory,
    index: &mut HashMap<String, TensorMeta>,
    upload_views: &mut HashMap<String, TensorUploadView>,
) -> Result<(), Error> {
    let direct_plan: HashMap<(u32, u16), &crate::flm::FlmPlanStep> = runtime
        .plan_steps()
        .iter()
        .filter(|step| step.consume_strategy == crate::flm::CONSUME_STRATEGY_DIRECT)
        .map(|step| ((step.logical_tensor_id, step.storage_role), step))
        .collect();
    let mut by_logical: HashMap<u32, Vec<&crate::flm::FlmStorageBinding>> = HashMap::new();
    for binding in runtime.storage_bindings() {
        by_logical
            .entry(binding.logical_tensor_id)
            .or_default()
            .push(binding);
    }

    for logical in runtime.logical_tensors() {
        if logical.value_format_id != crate::flm::VALUE_FORMAT_RAW_DENSE {
            continue;
        }
        let Some(value_step) =
            direct_plan.get(&(logical.tensor_id, crate::flm::STORAGE_ROLE_VALUE))
        else {
            continue;
        };
        let owned = by_logical.get(&logical.tensor_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 logical tensor {} has no storage bindings",
                logical.name
            ))
        })?;
        let value = owned
            .iter()
            .find(|binding| binding.storage_role == crate::flm::STORAGE_ROLE_VALUE)
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 raw VALUE tensor {} missing value binding",
                    logical.name
                ))
            })?;
        let value_meta = index.get(&value.tensor_name).cloned().ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 raw VALUE tensor {} missing from index",
                value.tensor_name
            ))
        })?;
        let view_dtype = flm_dtype_name(value_step.target_dtype)?.to_string();
        let view_shape = flm_plan_target_shape(value_step)?;
        let view_layout = stage3_raw_value_alias_layout(&logical.name, &view_shape);
        index.insert(
            logical.name.clone(),
            TensorMeta {
                name: logical.name.clone(),
                shape: view_shape.clone(),
                dtype: view_dtype.clone(),
                layout: view_layout,
                offset: value_meta.offset,
                byte_len: value_meta.byte_len,
            },
        );
        upload_views
            .entry(logical.name.clone())
            .or_insert(TensorUploadView {
                dtype: view_dtype,
                shape: view_shape,
            });
    }
    Ok(())
}

fn stage3_raw_value_alias_layout(name: &str, shape: &[usize]) -> LayoutTag {
    if name.contains(".linear_attn.") {
        if name.ends_with(".conv1d.weight") && shape.len() == 2 {
            return LayoutTag::DepthwiseConvSqueezed;
        }
        if name.ends_with(".dt_bias")
            && shape.len() == 3
            && shape.first() == Some(&1)
            && shape.get(1) == Some(&1)
        {
            return LayoutTag::HeadBiasReshaped;
        }
        if name.ends_with(".A_log")
            && shape.len() == 3
            && shape.first() == Some(&1)
            && shape.get(1) == Some(&1)
        {
            return LayoutTag::HeadExpReshaped;
        }
    }
    LayoutTag::Raw
}

fn stage3_int4_sidecar_alias(logical_name: &str, suffix: &str) -> String {
    if let Some(base) = logical_name.strip_suffix(".weight") {
        format!("{base}.weight_int4_{suffix}")
    } else {
        format!("{logical_name}_int4_{suffix}")
    }
}

fn add_stage3_int4_aliases(
    runtime: &crate::flm::FlmRuntimeDirectory,
    index: &mut HashMap<String, TensorMeta>,
    upload_views: &mut HashMap<String, TensorUploadView>,
    ct_int4_bf16_fallbacks: &mut HashMap<String, CtSymInt4Bf16Fallback>,
) -> Result<(), Error> {
    let direct_plan: HashMap<(u32, u16), &crate::flm::FlmPlanStep> = runtime
        .plan_steps()
        .iter()
        .filter(|step| step.consume_strategy == crate::flm::CONSUME_STRATEGY_DIRECT)
        .map(|step| ((step.logical_tensor_id, step.storage_role), step))
        .collect();
    let mut by_logical: HashMap<u32, Vec<&crate::flm::FlmStorageBinding>> = HashMap::new();
    for binding in runtime.storage_bindings() {
        by_logical
            .entry(binding.logical_tensor_id)
            .or_default()
            .push(binding);
    }

    for logical in runtime.logical_tensors() {
        if logical.value_format_id != crate::flm::VALUE_FORMAT_SYM_INT4 {
            continue;
        }
        let Some(packed_step) =
            direct_plan.get(&(logical.tensor_id, crate::flm::STORAGE_ROLE_PACKED))
        else {
            continue;
        };
        let scale_step = direct_plan
            .get(&(logical.tensor_id, crate::flm::STORAGE_ROLE_SCALE))
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} missing scale direct plan",
                    logical.name
                ))
            })?;
        let owned = by_logical.get(&logical.tensor_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 logical tensor {} has no storage bindings",
                logical.name
            ))
        })?;
        let packed = owned
            .iter()
            .find(|binding| binding.storage_role == crate::flm::STORAGE_ROLE_PACKED)
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} missing packed binding",
                    logical.name
                ))
            })?;
        let scale = owned
            .iter()
            .find(|binding| binding.storage_role == crate::flm::STORAGE_ROLE_SCALE)
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} missing scale binding",
                    logical.name
                ))
            })?;
        let shape_binding = owned
            .iter()
            .find(|binding| binding.storage_role == crate::flm::STORAGE_ROLE_SHAPE);
        let zero = owned
            .iter()
            .find(|binding| binding.storage_role == crate::flm::STORAGE_ROLE_ZERO);
        let packed_meta = index.get(&packed.tensor_name).cloned().ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 packed tensor {} missing from index",
                packed.tensor_name
            ))
        })?;
        let scale_meta = index.get(&scale.tensor_name).cloned().ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 scale tensor {} missing from index",
                scale.tensor_name
            ))
        })?;
        let storage_abi = runtime
            .storage_abis()
            .iter()
            .find(|abi| abi.storage_abi_id == packed.storage_abi_id)
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} references missing storage ABI {}",
                    logical.name, packed.storage_abi_id
                ))
            })?;
        if packed.storage_dtype == FLM_DTYPE_INT32 {
            shape_binding.ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} missing shape binding",
                    logical.name
                ))
            })?;
            if logical.rank != 2 {
                return Err(Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} rank {} is unsupported for CT BF16 fallback",
                    logical.name, logical.rank
                )));
            }
            let shape = vec![logical.shape[0] as usize, logical.shape[1] as usize];
            if scale.storage_dtype != FLM_DTYPE_BF16 {
                return Err(Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} scale binding dtype {} is unsupported for BF16 fallback",
                    logical.name, scale.storage_dtype
                )));
            }
            if storage_abi.codec_semantic_id != crate::flm::CODEC_SYM_INT4_G128_BF16
                || storage_abi.bits != 4
            {
                return Err(Error::Other(format!(
                    "FLM Stage 3 INT4 tensor {} uses unsupported CT fallback ABI codec={} bits={}",
                    logical.name, storage_abi.codec_semantic_id, storage_abi.bits
                )));
            }
            let group_size = usize::from(storage_abi.group_size);
            let byte_len = shape
                .iter()
                .try_fold(2usize, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| {
                    Error::Other(format!(
                        "FLM Stage 3 INT4 tensor {} BF16 fallback byte_len overflows",
                        logical.name
                    ))
                })?;
            index
                .entry(logical.name.clone())
                .or_insert_with(|| TensorMeta {
                    name: logical.name.clone(),
                    shape: shape.clone(),
                    dtype: "bf16".to_string(),
                    layout: LayoutTag::Raw,
                    offset: packed_meta.offset,
                    byte_len: byte_len as u64,
                });
            upload_views
                .entry(logical.name.clone())
                .or_insert_with(|| TensorUploadView {
                    dtype: "bf16".to_string(),
                    shape: shape.clone(),
                });
            ct_int4_bf16_fallbacks
                .entry(logical.name.clone())
                .or_insert_with(|| CtSymInt4Bf16Fallback {
                    packed_tensor: packed.tensor_name.clone(),
                    scale_tensor: scale.tensor_name.clone(),
                    shape,
                    group_size,
                });
            continue;
        }
        if packed.storage_dtype != FLM_DTYPE_UINT8 {
            return Err(Error::Other(format!(
                "FLM Stage 3 native INT4 tensor {} packed binding dtype {} is unsupported",
                logical.name, packed.storage_dtype
            )));
        }
        if storage_abi.codec_semantic_id != crate::flm::CODEC_SUPERSONIC_NATIVE_INT4_G128_BF16
            || storage_abi.bits != 4
            || storage_abi.group_size != 128
        {
            return Err(Error::Other(format!(
                "FLM Stage 3 native INT4 tensor {} uses unsupported native INT4 ABI codec={} bits={} group_size={}",
                logical.name,
                storage_abi.codec_semantic_id,
                storage_abi.bits,
                storage_abi.group_size
            )));
        }
        let packed_view = TensorUploadView {
            dtype: flm_dtype_name(packed_step.target_dtype)?.to_string(),
            shape: flm_plan_target_shape(packed_step)?,
        };
        let zero = zero.ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 native INT4 tensor {} missing zero binding",
                logical.name
            ))
        })?;
        let zero_step = direct_plan
            .get(&(logical.tensor_id, crate::flm::STORAGE_ROLE_ZERO))
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 native INT4 tensor {} missing zero direct plan",
                    logical.name
                ))
            })?;
        let zero_meta = index.get(&zero.tensor_name).cloned().ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 zero tensor {} missing from index",
                zero.tensor_name
            ))
        })?;
        let packed_shape = packed_view.shape.clone();
        index.insert(
            logical.name.clone(),
            TensorMeta {
                name: logical.name.clone(),
                shape: packed_shape,
                dtype: "u8".to_string(),
                layout: LayoutTag::Int4Quantized,
                offset: packed_meta.offset,
                byte_len: packed_meta.byte_len,
            },
        );
        upload_views
            .entry(logical.name.clone())
            .or_insert(packed_view);

        let scale_alias = stage3_int4_sidecar_alias(&logical.name, "scale");
        let scale_dtype = flm_dtype_name(scale_step.target_dtype)?;
        let scale_shape = flm_plan_target_shape(scale_step)?;
        index
            .entry(scale_alias.clone())
            .or_insert_with(|| TensorMeta {
                name: scale_alias.clone(),
                shape: scale_shape.clone(),
                dtype: scale_dtype.to_string(),
                layout: LayoutTag::Raw,
                offset: scale_meta.offset,
                byte_len: scale_meta.byte_len,
            });
        upload_views
            .entry(scale_alias)
            .or_insert_with(|| TensorUploadView {
                dtype: scale_dtype.to_string(),
                shape: scale_shape.clone(),
            });

        let zero_alias = stage3_int4_sidecar_alias(&logical.name, "zero");
        let zero_dtype = flm_dtype_name(zero_step.target_dtype)?;
        let zero_shape = flm_plan_target_shape(zero_step)?;
        index
            .entry(zero_alias.clone())
            .or_insert_with(|| TensorMeta {
                name: zero_alias.clone(),
                shape: zero_shape.clone(),
                dtype: zero_dtype.to_string(),
                layout: LayoutTag::Raw,
                offset: zero_meta.offset,
                byte_len: zero_meta.byte_len,
            });
        upload_views
            .entry(zero_alias)
            .or_insert_with(|| TensorUploadView {
                dtype: zero_dtype.to_string(),
                shape: zero_shape,
            });
    }
    Ok(())
}

fn add_stage3_lowbit_aliases(
    runtime: &crate::flm::FlmRuntimeDirectory,
    index: &mut HashMap<String, TensorMeta>,
    upload_views: &mut HashMap<String, TensorUploadView>,
) -> Result<(), Error> {
    let direct_plan: HashMap<(u32, u16), &crate::flm::FlmPlanStep> = runtime
        .plan_steps()
        .iter()
        .filter(|step| step.consume_strategy == crate::flm::CONSUME_STRATEGY_DIRECT)
        .map(|step| ((step.logical_tensor_id, step.storage_role), step))
        .collect();
    let mut by_logical: HashMap<u32, Vec<&crate::flm::FlmStorageBinding>> = HashMap::new();
    for binding in runtime.storage_bindings() {
        by_logical
            .entry(binding.logical_tensor_id)
            .or_default()
            .push(binding);
    }

    for logical in runtime.logical_tensors() {
        let (format_tag, primary_role, sidecars): (&str, u16, &[(u16, &str)]) =
            match logical.value_format_id {
                crate::flm::VALUE_FORMAT_NVFP4_E2M1 => (
                    "nvfp4",
                    crate::flm::STORAGE_ROLE_PACKED,
                    &[
                        (crate::flm::STORAGE_ROLE_SCALE, "scale"),
                        (crate::flm::STORAGE_ROLE_GLOBAL_SCALE, "global_scale"),
                    ],
                ),
                crate::flm::VALUE_FORMAT_MXFP4_E2M1 => (
                    "mxfp4",
                    crate::flm::STORAGE_ROLE_PACKED,
                    &[(crate::flm::STORAGE_ROLE_SCALE, "scale")],
                ),
                crate::flm::VALUE_FORMAT_MXFP8_E4M3 => (
                    "mxfp8",
                    crate::flm::STORAGE_ROLE_VALUE,
                    &[(crate::flm::STORAGE_ROLE_SCALE, "scale")],
                ),
                crate::flm::VALUE_FORMAT_FP8_E4M3_F32 => (
                    "fp8_e4m3_f32",
                    crate::flm::STORAGE_ROLE_VALUE,
                    &[(crate::flm::STORAGE_ROLE_SCALE, "scale")],
                ),
                crate::flm::VALUE_FORMAT_FP8_E4M3_B128_BF16_INV => (
                    "fp8_e4m3_b128_bf16",
                    crate::flm::STORAGE_ROLE_VALUE,
                    &[(crate::flm::STORAGE_ROLE_SCALE, "scale_inv")],
                ),
                crate::flm::VALUE_FORMAT_FP8_E4M3_B64_BF16 => (
                    "fp8_e4m3_b64_bf16",
                    crate::flm::STORAGE_ROLE_VALUE,
                    &[(crate::flm::STORAGE_ROLE_SCALE, "scale")],
                ),
                _ => continue,
            };

        let Some(primary_step) = direct_plan.get(&(logical.tensor_id, primary_role)) else {
            return Err(Error::Other(format!(
                "FLM Stage 3 {format_tag} tensor {} missing primary direct plan",
                logical.name
            )));
        };
        let owned = by_logical.get(&logical.tensor_id).ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 logical tensor {} has no storage bindings",
                logical.name
            ))
        })?;
        let primary = owned
            .iter()
            .find(|binding| binding.storage_role == primary_role)
            .ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 {format_tag} tensor {} missing primary binding",
                    logical.name
                ))
            })?;
        let primary_meta = index.get(&primary.tensor_name).cloned().ok_or_else(|| {
            Error::Other(format!(
                "FLM Stage 3 primary tensor {} missing from index",
                primary.tensor_name
            ))
        })?;
        let primary_dtype = flm_dtype_name(primary_step.target_dtype)?.to_string();
        let primary_upload_shape = flm_plan_target_shape(primary_step)?;
        let logical_shape = flm_logical_shape(logical)?;
        index.insert(
            logical.name.clone(),
            TensorMeta {
                name: logical.name.clone(),
                shape: logical_shape,
                dtype: primary_dtype.clone(),
                layout: LayoutTag::Raw,
                offset: primary_meta.offset,
                byte_len: primary_meta.byte_len,
            },
        );
        upload_views.insert(
            logical.name.clone(),
            TensorUploadView {
                dtype: primary_dtype,
                shape: primary_upload_shape,
            },
        );

        let alias_base = logical
            .name
            .strip_suffix(".weight")
            .unwrap_or(&logical.name);
        for (role, suffix) in sidecars {
            let step = direct_plan
                .get(&(logical.tensor_id, *role))
                .ok_or_else(|| {
                    Error::Other(format!(
                        "FLM Stage 3 {format_tag} tensor {} missing {suffix} direct plan",
                        logical.name
                    ))
                })?;
            let binding = owned
                .iter()
                .find(|binding| binding.storage_role == *role)
                .ok_or_else(|| {
                    Error::Other(format!(
                        "FLM Stage 3 {format_tag} tensor {} missing {suffix} binding",
                        logical.name
                    ))
                })?;
            let meta = index.get(&binding.tensor_name).cloned().ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 sidecar tensor {} missing from index",
                    binding.tensor_name
                ))
            })?;
            let dtype = flm_dtype_name(step.target_dtype)?;
            let shape = flm_plan_target_shape(step)?;
            let alias = format!("{alias_base}.weight_{format_tag}_{suffix}");
            index.entry(alias.clone()).or_insert_with(|| TensorMeta {
                name: alias.clone(),
                shape: shape.clone(),
                dtype: dtype.to_string(),
                layout: LayoutTag::Raw,
                offset: meta.offset,
                byte_len: meta.byte_len,
            });
            upload_views
                .entry(alias)
                .or_insert_with(|| TensorUploadView {
                    dtype: dtype.to_string(),
                    shape,
                });
        }
        if matches!(
            logical.value_format_id,
            crate::flm::VALUE_FORMAT_NVFP4_E2M1 | crate::flm::VALUE_FORMAT_FP8_E4M3_F32
        ) && direct_plan.contains_key(&(logical.tensor_id, crate::flm::STORAGE_ROLE_INPUT_SCALE))
        {
            let step = direct_plan
                .get(&(logical.tensor_id, crate::flm::STORAGE_ROLE_INPUT_SCALE))
                .expect("checked input scale plan");
            let binding = owned
                .iter()
                .find(|binding| binding.storage_role == crate::flm::STORAGE_ROLE_INPUT_SCALE)
                .ok_or_else(|| {
                    Error::Other(format!(
                        "FLM Stage 3 {format_tag} tensor {} missing input_scale binding",
                        logical.name
                    ))
                })?;
            let meta = index.get(&binding.tensor_name).cloned().ok_or_else(|| {
                Error::Other(format!(
                    "FLM Stage 3 sidecar tensor {} missing from index",
                    binding.tensor_name
                ))
            })?;
            let dtype = flm_dtype_name(step.target_dtype)?;
            let shape = flm_plan_target_shape(step)?;
            let alias = format!("{alias_base}.weight_{format_tag}_input_scale");
            index.entry(alias.clone()).or_insert_with(|| TensorMeta {
                name: alias.clone(),
                shape: shape.clone(),
                dtype: dtype.to_string(),
                layout: LayoutTag::Raw,
                offset: meta.offset,
                byte_len: meta.byte_len,
            });
            upload_views
                .entry(alias)
                .or_insert_with(|| TensorUploadView {
                    dtype: dtype.to_string(),
                    shape,
                });
        }
    }
    Ok(())
}

fn flm_logical_shape(logical: &crate::flm::FlmLogicalTensor) -> Result<Vec<usize>, Error> {
    let rank = usize::from(logical.rank);
    if rank > logical.shape.len() {
        return Err(Error::Other(format!(
            "FLM logical tensor {} rank {} exceeds stored shape rank {}",
            logical.name,
            logical.rank,
            logical.shape.len()
        )));
    }
    logical.shape[..rank]
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                Error::Other(format!(
                    "FLM logical tensor {} dimension {dim} does not fit usize",
                    logical.name
                ))
            })
        })
        .collect()
}

fn flm_plan_target_shape(step: &crate::flm::FlmPlanStep) -> Result<Vec<usize>, Error> {
    let rank = usize::from(step.target_rank);
    if rank > step.target_shape.len() {
        return Err(Error::Other(format!(
            "FLM plan step target rank {} exceeds stored shape rank {}",
            step.target_rank,
            step.target_shape.len()
        )));
    }
    step.target_shape[..rank]
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                Error::Other(format!(
                    "FLM plan step target dimension {dim} does not fit usize"
                ))
            })
        })
        .collect()
}

impl BakedStore {
    /// Open a baked package from a bake directory.
    /// Reads manifest.json and mmaps weights.bin.
    pub fn open(bake_dir: &Path) -> Result<Self, Error> {
        let manifest_text = std::fs::read_to_string(crate::manifest_path(bake_dir))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)?;

        let weights_path = crate::weights_bin_path(bake_dir);
        let weights_file = File::open(&weights_path)?;
        let source_path =
            std::fs::canonicalize(&weights_path).unwrap_or_else(|_| weights_path.clone());
        let mmap = unsafe { Mmap::map(&weights_file)? };

        let data = mmap.as_ptr();
        let data_len = mmap.len();

        let mut index = HashMap::with_capacity(manifest.tensors.len());
        for entry in manifest.tensors {
            index.insert(entry.name.clone(), entry);
        }

        Ok(Self {
            _mmap: mmap,
            data,
            data_len,
            source_kind: TensorStorageSourceKind::BakedWeights,
            source_path,
            index,
            synthetic: HashMap::new(),
            upload_views: HashMap::new(),
            ct_int4_bf16_fallbacks: HashMap::new(),
            runtime: None,
        })
    }

    /// Open an FLM container directly as a BakedStore-compatible byte index.
    pub fn open_flm(path: &Path) -> Result<Self, Error> {
        Self::open_flm_with_options(path, FlmLoadOptions::default())
    }

    /// Open an FLM container with optional compatibility aliases.
    pub fn open_flm_with_options(path: &Path, options: FlmLoadOptions) -> Result<Self, Error> {
        let file = File::open(path)?;
        let source_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mmap = unsafe { Mmap::map(&file)? };
        let data = mmap.as_ptr();
        let data_len = mmap.len();
        let sb = flm_parse_superblock(&mmap)?;
        let strings = flm_read_string_table(&mmap, &sb)?;
        let shards = flm_read_shards(&mmap, &sb)?;
        let index_entries = flm_read_index_entries(&mmap, &sb, &strings, &shards)?;
        let mut index = flm_build_index(&index_entries)?;
        let runtime = match (sb.runtime_dir_offset, sb.runtime_dir_len) {
            (0, 0) => None,
            (0, len) => {
                return Err(Error::Other(format!(
                    "FLM runtime directory length is {len} but offset is zero"
                )));
            }
            (offset, 0) => {
                return Err(Error::Other(format!(
                    "FLM runtime directory offset is {offset} but length is zero"
                )));
            }
            (offset, len) => {
                let runtime = read_exact_range(&mmap, offset, len, "FLM runtime directory")?;
                Some(crate::flm::FlmRuntimeDirectory::parse(runtime)?)
            }
        };
        if options.verify_block_hashes {
            flm_verify_block_hashes(&mmap, &sb, &strings, &shards)?;
        }
        if let Some(runtime) = runtime.as_ref() {
            validate_flm_manifest_against_index(runtime, &index_entries)?;
        }
        let mut synthetic = HashMap::new();
        let mut upload_views = HashMap::new();
        let mut ct_int4_bf16_fallbacks = HashMap::new();
        if options.flm_int4_logical_aliases {
            if let Some(runtime) = runtime.as_ref() {
                if !runtime.logical_tensors().is_empty() {
                    add_stage3_raw_value_aliases(runtime, &mut index, &mut upload_views)?;
                    add_stage3_int4_aliases(
                        runtime,
                        &mut index,
                        &mut upload_views,
                        &mut ct_int4_bf16_fallbacks,
                    )?;
                    add_stage3_lowbit_aliases(runtime, &mut index, &mut upload_views)?;
                } else {
                    add_manifest_int4_aliases(runtime, &mut index, &mut synthetic)?;
                }
            }
        }
        Ok(Self {
            _mmap: mmap,
            data,
            data_len,
            source_kind: TensorStorageSourceKind::FlmContainer,
            source_path,
            index,
            synthetic,
            upload_views,
            ct_int4_bf16_fallbacks,
            runtime,
        })
    }

    /// Check if a tensor exists in the store.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Get the shape of a tensor without loading it.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|m| m.shape.as_slice())
    }

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.index.get(name)
    }

    pub fn layout(&self, name: &str) -> Option<&LayoutTag> {
        self.index.get(name).map(|m| &m.layout)
    }

    pub fn flm_runtime(&self) -> Option<&crate::flm::FlmRuntimeDirectory> {
        self.runtime.as_ref()
    }

    #[cfg(test)]
    fn upload_view(&self, name: &str) -> Option<&TensorUploadView> {
        self.upload_views.get(name)
    }

    fn materialize_ct_int4_bf16_fallback(
        &self,
        name: &str,
        fallback: &CtSymInt4Bf16Fallback,
    ) -> Result<Vec<u8>, Error> {
        let packed_meta = self
            .index
            .get(&fallback.packed_tensor)
            .ok_or_else(|| Error::NotFound(fallback.packed_tensor.clone()))?;
        let scale_meta = self
            .index
            .get(&fallback.scale_tensor)
            .ok_or_else(|| Error::NotFound(fallback.scale_tensor.clone()))?;
        let packed_bytes = self.tensor_bytes(&fallback.packed_tensor, packed_meta)?;
        let scale_bytes = self.tensor_bytes(&fallback.scale_tensor, scale_meta)?;
        flm_ct_sym_int4_to_bf16(
            packed_bytes,
            &packed_meta.shape,
            scale_bytes,
            &scale_meta.shape,
            &fallback.shape,
            fallback.group_size,
            name,
        )
    }

    fn upload_payload<'a>(
        &'a self,
        name: &str,
        meta: &'a TensorMeta,
    ) -> Result<(ScalarType, Vec<usize>, Cow<'a, [u8]>), Error> {
        if let Some(fallback) = self.ct_int4_bf16_fallbacks.get(name) {
            let bytes = self.materialize_ct_int4_bf16_fallback(name, fallback)?;
            return Ok((ScalarType::BF16, fallback.shape.clone(), Cow::Owned(bytes)));
        }

        let slice = self.tensor_bytes(name, meta)?;
        let (dtype, upload_shape) = if let Some(view) = self.upload_views.get(name) {
            (parse_dtype(&view.dtype)?, view.shape.clone())
        } else {
            (parse_dtype(&meta.dtype)?, gpu_upload_shape(meta)?)
        };
        Ok((dtype, upload_shape, Cow::Borrowed(slice)))
    }

    #[cfg(test)]
    fn materialize_upload_for_test(
        &self,
        name: &str,
    ) -> Result<(ScalarType, Vec<usize>, Vec<u8>), Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let (dtype, shape, bytes) = self.upload_payload(name, meta)?;
        Ok((dtype, shape, bytes.into_owned()))
    }

    fn tensor_bytes(&self, name: &str, meta: &TensorMeta) -> Result<&[u8], Error> {
        if let Some(bytes) = self.synthetic.get(name) {
            if bytes.len() != meta.byte_len as usize {
                return Err(Error::Other(format!(
                    "synthetic tensor '{}' byte_len={} does not match metadata byte_len={}",
                    name,
                    bytes.len(),
                    meta.byte_len
                )));
            }
            return Ok(bytes);
        }
        let start = meta.offset as usize;
        let end = start.checked_add(meta.byte_len as usize).ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' byte range overflows (offset={}, len={})",
                meta.offset, meta.byte_len
            ))
        })?;
        if end > self.data_len {
            return Err(Error::Other(format!(
                "tensor '{}' extends past end of weight store (offset={}, len={}, file_len={})",
                name, meta.offset, meta.byte_len, self.data_len,
            )));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.data.add(start), meta.byte_len as usize) })
    }

    /// Return the raw mmap-backed bytes of a tensor. Useful for tensors that
    /// are too large to upload to GPU in full (e.g. Gemma 4's
    /// `embed_tokens_per_layer`, which is row-accessed per-token). The slice
    /// lives as long as the `BakedStore`'s mmap.
    pub fn raw_bytes(&self, name: &str) -> Option<&[u8]> {
        if self.ct_int4_bf16_fallbacks.contains_key(name) {
            return None;
        }
        let meta = self.index.get(name)?;
        self.tensor_bytes(name, meta).ok()
    }

    pub fn raw_byte_range(
        &self,
        name: &str,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<&[u8], Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if self.ct_int4_bf16_fallbacks.contains_key(name) {
            return Err(Error::Other(format!(
                "tensor '{name}' is a transformed FLM logical alias and has no raw byte range"
            )));
        }
        let bytes = self.tensor_bytes(name, meta)?;
        let range_end = byte_offset.checked_add(byte_len).ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' raw range overflows: offset={byte_offset} len={byte_len}"
            ))
        })?;
        if range_end > meta.byte_len as usize {
            return Err(Error::Other(format!(
                "tensor '{name}' raw range [{byte_offset}, {range_end}) exceeds byte_len={}",
                meta.byte_len
            )));
        }
        Ok(&bytes[byte_offset..range_end])
    }

    /// Return the concrete file extent for a direct file-backed tensor.
    ///
    /// This is the storage-side descriptor future transfer backends need: the
    /// source file identity plus byte extent, alongside both the bytes-on-disk
    /// metadata and the runtime upload view. Synthesized and transformed
    /// aliases return `Ok(None)` because there is no single source-file extent
    /// that can be transferred directly.
    pub fn tensor_storage_extent(&self, name: &str) -> Result<Option<TensorStorageExtent>, Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if self.synthetic.contains_key(name) || self.ct_int4_bf16_fallbacks.contains_key(name) {
            return Ok(None);
        }
        self.tensor_bytes(name, meta)?;
        let (upload_dtype, upload_shape) = if let Some(view) = self.upload_views.get(name) {
            (view.dtype.clone(), view.shape.clone())
        } else {
            (meta.dtype.clone(), gpu_upload_shape(meta)?)
        };
        Ok(Some(TensorStorageExtent {
            source_kind: self.source_kind,
            source_path: self.source_path.clone(),
            name: name.to_string(),
            file_offset: meta.offset,
            byte_len: meta.byte_len,
            storage_dtype: meta.dtype.clone(),
            storage_shape: meta.shape.clone(),
            layout: meta.layout.clone(),
            upload_dtype,
            upload_shape,
        }))
    }

    /// Return the concrete file range for a direct tensor subrange.
    ///
    /// Virtual allocation loaders use this as the source-side transfer
    /// descriptor: current backends copy from the mmap, while future
    /// storage-to-GPU backends can use `file_offset` and `byte_len` directly.
    pub fn tensor_storage_range(
        &self,
        name: &str,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Option<TensorStorageRange>, Error> {
        let Some(extent) = self.tensor_storage_extent(name)? else {
            return Ok(None);
        };
        let tensor_byte_offset = u64::try_from(byte_offset).map_err(|_| {
            Error::Other(format!(
                "tensor '{name}' storage range offset does not fit u64: {byte_offset}"
            ))
        })?;
        let byte_len_u64 = u64::try_from(byte_len).map_err(|_| {
            Error::Other(format!(
                "tensor '{name}' storage range length does not fit u64: {byte_len}"
            ))
        })?;
        let range_end = tensor_byte_offset
            .checked_add(byte_len_u64)
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{name}' storage range overflows: offset={byte_offset} len={byte_len}"
                ))
            })?;
        if range_end > extent.byte_len {
            return Err(Error::Other(format!(
                "tensor '{name}' storage range [{byte_offset}, {range_end}) exceeds byte_len={}",
                extent.byte_len
            )));
        }
        let file_offset = extent
            .file_offset
            .checked_add(tensor_byte_offset)
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{name}' storage range file offset overflows: base={} offset={byte_offset}",
                    extent.file_offset
                ))
            })?;
        Ok(Some(TensorStorageRange {
            extent,
            tensor_byte_offset,
            file_offset,
            byte_len: byte_len_u64,
        }))
    }

    /// Load a tensor from the baked store to GPU memory.
    /// Direct tensors borrow mmap bytes; transformed FLM logical aliases
    /// materialize a temporary host fallback payload before upload.
    pub fn load_to_gpu(&self, name: &str, ordinal: usize) -> Result<GpuBuffer, Error> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let (dtype, upload_shape, payload) = self.upload_payload(name, meta)?;
        let buf = GpuBuffer::from_host_bytes(ordinal, dtype, &upload_shape, payload.as_ref())?;
        Ok(buf)
    }

    /// Load a direct mmap-backed tensor through backend host registration.
    ///
    /// This is an opt-in diagnostic/experimental path for callers that have
    /// already determined registration is worthwhile for the tensor
    /// shape/layout. It is not the first-class FLM fast-load target; direct FLM
    /// plans should still prefer GPU-resident layouts such as virtual expert
    /// slabs. The registered path deliberately rejects transformed or
    /// synthetic payloads because those are not backed by this store's mmap.
    pub fn load_to_gpu_registered_mmap(
        &self,
        name: &str,
        ordinal: usize,
    ) -> Result<GpuBuffer, Error> {
        if current_backend() != Backend::Hip {
            return Err(Error::Gpu(gpu_hal::GpuError::Unsupported(
                "registered mmap upload is currently implemented for HIP only".into(),
            )));
        }
        let extent = self.tensor_storage_extent(name)?.ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' is not a direct file-backed extent and cannot use registered upload"
            ))
        })?;
        let byte_offset = u64_to_usize(extent.file_offset, "registered upload file offset")?;
        let byte_len = u64_to_usize(extent.byte_len, "registered upload byte length")?;
        let dtype = parse_dtype(&extent.upload_dtype)?;
        let upload_shape = extent.upload_shape;
        let bytes = self.raw_byte_range(name, 0, byte_len)?;
        let data_start = (self.data as usize)
            .checked_add(byte_offset)
            .ok_or_else(|| Error::Other("registered upload data pointer overflows".into()))?;
        let range = host_registration_range_for_mmap_slice(
            self.data as usize,
            self.data_len,
            data_start,
            byte_len,
            host_page_size(),
        )?;
        let elems = upload_shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{name}' upload shape {:?} overflows element count",
                    upload_shape
                ))
            })?;
        let expected_bytes = elems.checked_mul(dtype.size_in_bytes()).ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' upload shape {:?} overflows byte count",
                upload_shape
            ))
        })?;
        if bytes.len() != expected_bytes {
            return Err(Error::Other(format!(
                "tensor '{name}' registered upload expected {expected_bytes} bytes from dtype={dtype:?} shape={upload_shape:?}, got {}",
                bytes.len()
            )));
        }

        let stream = GpuStream::new_nonblocking(ordinal)?;
        let mut buffer = GpuBuffer::alloc(ordinal, dtype, &upload_shape)?;
        let registered = unsafe { RegisteredHostBuffer::new(ordinal, range.ptr, range.len)? };
        copy_h2d_async(
            ordinal,
            &stream,
            buffer.as_mut_ptr(),
            bytes.as_ptr() as *const c_void,
            bytes.len(),
        )?;
        stream.synchronize()?;
        drop(registered);
        Ok(buffer)
    }

    /// Load a tensor into a role-tagged virtual allocation.
    ///
    /// This is the low-level entry point for virtual weights and MoE expert
    /// islands: the allocation keeps a stable virtual address, exposes logical
    /// versus page-resident accounting through `VirtualArena`, and can later be
    /// evicted/restored by the HAL policy layer.
    pub fn load_to_virtual_arena(
        &self,
        arena: &mut VirtualArena,
        name: &str,
        role: VirtualAllocationRole,
    ) -> Result<usize, Error> {
        let id = self.reserve_virtual_arena(arena, name, role)?;
        let len = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?
            .byte_len as usize;
        self.load_range_to_virtual_arena(arena, id, name, 0, len)?;
        Ok(id)
    }

    /// Reserve a stable virtual address range for one baked tensor without
    /// making any physical pages resident yet.
    pub fn reserve_virtual_arena(
        &self,
        arena: &mut VirtualArena,
        name: &str,
        role: VirtualAllocationRole,
    ) -> Result<usize, Error> {
        if self.ct_int4_bf16_fallbacks.contains_key(name) {
            return Err(Error::Other(format!(
                "tensor '{name}' is a transformed FLM logical alias and cannot be reserved in a virtual arena"
            )));
        }
        let extent = self.tensor_storage_extent(name)?.ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' is not a direct file-backed extent and cannot be reserved in a virtual arena"
            ))
        })?;
        let dtype = parse_dtype(&extent.storage_dtype)?;
        let expected_len = extent
            .storage_shape
            .iter()
            .try_fold(dtype.size_in_bytes(), |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| {
                Error::Other(format!(
                    "tensor '{}' shape {:?} overflows byte-size calculation",
                    name, extent.storage_shape
                ))
            })?;
        if extent.byte_len != expected_len as u64 {
            return Err(Error::Other(format!(
                "tensor '{}' manifest byte_len={} does not match dtype={} shape={:?} expected_bytes={}",
                name,
                extent.byte_len,
                extent.storage_dtype,
                extent.storage_shape,
                expected_len,
            )));
        }
        arena
            .reserve(name.to_string(), role, dtype, &extent.storage_shape)
            .map_err(Error::from)
    }

    /// Upload a byte range from a baked tensor into an existing virtual
    /// allocation. The allocation must have been reserved from the same tensor.
    pub fn load_range_to_virtual_arena(
        &self,
        arena: &mut VirtualArena,
        allocation_id: usize,
        name: &str,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), Error> {
        let transfer_range = self.tensor_storage_range(name, byte_offset, byte_len)?.ok_or_else(|| {
            Error::Other(format!(
                "tensor '{name}' is not a direct file-backed extent and cannot be range-loaded into a virtual arena"
            ))
        })?;
        let dtype = parse_dtype(&transfer_range.extent.storage_dtype)?;
        let ordinal = arena.device_ordinal();
        let allocation = arena.allocation_mut(allocation_id).ok_or_else(|| {
            Error::Other(format!("virtual allocation id {allocation_id} missing"))
        })?;
        let buffer = allocation.buffer_mut();
        if buffer.dtype() != dtype
            || buffer.shape() != transfer_range.extent.storage_shape.as_slice()
        {
            return Err(Error::Other(format!(
                "virtual allocation id {allocation_id} does not match tensor '{name}' \
                 (allocation dtype={:?} shape={:?}, tensor dtype={} shape={:?})",
                buffer.dtype(),
                buffer.shape(),
                transfer_range.extent.storage_dtype,
                transfer_range.extent.storage_shape
            )));
        }
        let src_byte_offset = u64_to_usize(
            transfer_range.tensor_byte_offset,
            "virtual arena source byte offset",
        )?;
        let src_byte_len =
            u64_to_usize(transfer_range.byte_len, "virtual arena source byte length")?;
        let src_bytes = self.raw_byte_range(name, src_byte_offset, src_byte_len)?;
        // The source tensor bytes are copied into the mapped range before the
        // allocation is observed as loaded, so clearing new pages first only
        // adds startup bandwidth.
        buffer.map_range_bytes_no_sync(byte_offset, src_byte_len)?;
        let src = src_bytes.as_ptr() as *const _;
        copy_h2d(
            ordinal,
            buffer.offset_mut_ptr(byte_offset),
            src,
            src_byte_len,
        )?;
        sync(ordinal)?;
        Ok(())
    }

    /// Convenience constructor for a virtual arena using this store's common
    /// CPU-backup residency policy.
    pub fn virtual_weight_arena(ordinal: usize) -> VirtualArena {
        VirtualArena::new(ordinal, VirtualBacking::CpuBackup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{LayoutTag, Manifest, TensorMeta, FORMAT_VERSION};
    use std::io::Write;

    const TEST_FLM_SUPERBLOCK_SIZE: usize = 4096;
    const TEST_FLM_INDEX_RECORD_SIZE: usize = 64;
    const TEST_FLM_SHARD_DESC_SIZE: usize = 24;
    const TEST_FLM_HASH_RECORD_SIZE: usize = 40;

    struct TestFlmTensor {
        name: &'static str,
        shape: Vec<u32>,
        dtype: u16,
        codec: u8,
        payload: Vec<u8>,
    }

    fn put_u16(buf: &mut [u8], off: usize, value: u16) {
        buf[off..off + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(buf: &mut [u8], off: usize, value: u32) {
        buf[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(buf: &mut [u8], off: usize, value: u64) {
        buf[off..off + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn align_len(buf: &mut Vec<u8>, alignment: usize) {
        let rem = buf.len() % alignment;
        if rem != 0 {
            buf.resize(buf.len() + (alignment - rem), 0);
        }
    }

    fn put_head_crc64(buf: &mut [u8]) {
        let shard_table_offset =
            read_u64(buf, 88, "test FLM shard table offset").expect("shard table offset") as usize;
        put_u64(buf, 144, 0);
        let crc = flm_head_crc64(&buf[..shard_table_offset]).expect("head CRC64");
        put_u64(buf, 144, crc);
    }

    fn build_test_flm(tensors: &[TestFlmTensor]) -> Vec<u8> {
        build_test_flm_inner(tensors, None)
    }

    fn build_test_flm_with_hashes(tensors: &[TestFlmTensor], hashes: &[[u8; 32]]) -> Vec<u8> {
        assert_eq!(tensors.len(), hashes.len());
        build_test_flm_inner(tensors, Some(hashes))
    }

    fn build_test_flm_inner(tensors: &[TestFlmTensor], hashes: Option<&[[u8; 32]]>) -> Vec<u8> {
        let alignment = 256usize;
        let mut out = vec![0u8; TEST_FLM_SUPERBLOCK_SIZE];

        let index_offset = out.len();
        let mut shard = Vec::new();
        let mut records = Vec::new();
        for (idx, tensor) in tensors.iter().enumerate() {
            assert!(tensor.shape.len() <= 4);
            let shard_offset = shard.len();
            shard.extend_from_slice(&tensor.payload);

            let mut rec = [0u8; TEST_FLM_INDEX_RECORD_SIZE];
            put_u32(&mut rec, 0, idx as u32);
            put_u16(&mut rec, 4, 0);
            put_u16(&mut rec, 6, 0);
            put_u32(&mut rec, 8, 0);
            put_u64(&mut rec, 12, shard_offset as u64);
            put_u64(&mut rec, 20, tensor.payload.len() as u64);
            put_u64(&mut rec, 28, tensor.payload.len() as u64);
            put_u16(&mut rec, 36, tensor.dtype);
            put_u16(&mut rec, 38, tensor.dtype);
            rec[40] = tensor.codec;
            rec[41] = tensor.shape.len() as u8;
            for (dim_idx, dim) in tensor.shape.iter().enumerate() {
                put_u32(&mut rec, 42 + dim_idx * 4, *dim);
            }
            if hashes.is_some() {
                put_u32(&mut rec, 58, (idx + 1) as u32);
            }
            records.extend_from_slice(&rec);
        }
        out.extend_from_slice(&records);
        let index_len = records.len();

        let metadata_offset = out.len();
        put_u32_at_vec(&mut out, tensors.len() as u32);
        for tensor in tensors {
            put_u32_at_vec(&mut out, tensor.name.len() as u32);
            out.extend_from_slice(tensor.name.as_bytes());
        }
        put_u32_at_vec(&mut out, 0);
        let metadata_len = out.len() - metadata_offset;

        let mut hashtable_offset = 0usize;
        let mut hashtable_len = 0usize;
        if let Some(hashes) = hashes {
            hashtable_offset = out.len();
            hashtable_len = hashes.len() * TEST_FLM_HASH_RECORD_SIZE;
            for (tensor, digest) in tensors.iter().zip(hashes.iter()) {
                out.extend_from_slice(digest);
                out.extend_from_slice(&(tensor.payload.len() as u64).to_le_bytes());
            }
        }

        let shard_table_offset = out.len();
        out.resize(out.len() + TEST_FLM_SHARD_DESC_SIZE, 0);
        align_len(&mut out, alignment);
        let shard_offset = out.len();
        out.extend_from_slice(&shard);

        put_u32(&mut out, shard_table_offset, 0);
        put_u64(&mut out, shard_table_offset + 4, shard_offset as u64);
        put_u64(&mut out, shard_table_offset + 12, shard.len() as u64);
        out[shard_table_offset + 20] = 5;

        out[0..8].copy_from_slice(b"FLM1\0\0\0\0");
        put_u32(&mut out, 8, 1);
        put_u64(&mut out, 16, tensors.len() as u64);
        put_u64(&mut out, 24, index_offset as u64);
        put_u64(&mut out, 32, index_len as u64);
        put_u64(&mut out, 40, metadata_offset as u64);
        put_u64(&mut out, 48, metadata_len as u64);
        put_u64(&mut out, 72, hashtable_offset as u64);
        put_u64(&mut out, 80, hashtable_len as u64);
        put_u64(&mut out, 88, shard_table_offset as u64);
        put_u32(&mut out, 96, 1);
        put_u32(&mut out, 100, alignment as u32);
        put_u64(&mut out, 104, shard.len() as u64);
        put_u16(&mut out, 156, 1);
        put_head_crc64(&mut out);
        out
    }

    fn put_u32_at_vec(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u16(buf: &mut Vec<u8>, value: u16) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_f64(buf: &mut Vec<u8>, value: f64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, value: &str) {
        push_u32(buf, value.len() as u32);
        buf.extend_from_slice(value.as_bytes());
    }

    fn test_bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
        values
            .into_iter()
            .flat_map(|value| half::bf16::from_f32(value).to_le_bytes())
            .collect()
    }

    fn test_ct_int4_packed_bytes(
        rows: usize,
        cols: usize,
        mut code_at: impl FnMut(usize, usize) -> i32,
    ) -> Vec<u8> {
        assert_eq!(cols % 8, 0);
        let mut out = Vec::with_capacity(rows * cols / 2);
        for row in 0..rows {
            for packed_col in 0..(cols / 8) {
                let mut word = 0u32;
                for lane in 0..8 {
                    let col = packed_col * 8 + lane;
                    let code = code_at(row, col);
                    assert!((-8..=7).contains(&code));
                    word |= ((code + 8) as u32) << (lane * 4);
                }
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    }

    struct TestManifestRow {
        name: &'static str,
        role_id: u32,
        group_id: u32,
        companion_kind: u8,
        rank: u8,
        dtype: u16,
        logical_dtype: u16,
        codec_id: u8,
        flags: u8,
        shape: [u32; 4],
    }

    fn runtime_config_section() -> Vec<u8> {
        let mut out = Vec::new();
        for value in [
            151_936u32, 5120, 27_648, 62, 40, 8, 128, 262_144, 128, 256, 256, 16, 32,
        ] {
            push_u32(&mut out, value);
        }
        push_f64(&mut out, 1e-6);
        push_f64(&mut out, 10_000_000.0);
        out.push(1);
        out.push(0);
        push_u32(&mut out, 2);
        push_f64(&mut out, 0.25);
        push_u32(&mut out, 151_645);
        push_u32(&mut out, 151_643);
        push_u32(&mut out, 3);
        push_u32(&mut out, 3);
        push_u32(&mut out, 7);
        push_u32(&mut out, 11);
        out
    }

    fn runtime_tokenizer_section() -> Vec<u8> {
        let mut out = Vec::new();
        for value in [
            0u32,
            crate::flm::TOKENIZER_QWEN_BPE_V1,
            151_936,
            1,
            2,
            3,
            4,
            0,
        ] {
            push_u32(&mut out, value);
        }
        out
    }

    fn runtime_codec_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, 10);
        for (codec_id, semantic_id, layout_id, decoder_id, flags) in [
            (0u8, crate::flm::CODEC_RAW_BF16 as u8, 0u16, 0u16, 0u32),
            (
                1u8,
                crate::flm::CODEC_SYM_INT4_G128_BF16 as u8,
                0u16,
                1u16,
                0u32,
            ),
            (2u8, crate::flm::CODEC_RAW_I64 as u8, 0u16, 0u16, 0u32),
            (
                3u8,
                crate::flm::CODEC_NVFP4_E2M1_B16_E4M3_F32 as u8,
                0u16,
                1u16,
                0u32,
            ),
            (
                4u8,
                crate::flm::CODEC_MXFP4_E2M1_B32_E8M0 as u8,
                0u16,
                1u16,
                0u32,
            ),
            (
                5u8,
                crate::flm::CODEC_MXFP8_E4M3_B32_E8M0 as u8,
                0u16,
                1u16,
                0u32,
            ),
            (6u8, crate::flm::CODEC_FP8_E4M3_F32 as u8, 0u16, 1u16, 0u32),
            (
                7u8,
                crate::flm::CODEC_FP8_E4M3_B128_BF16_INV as u8,
                0u16,
                1u16,
                0u32,
            ),
            (
                8u8,
                crate::flm::CODEC_FP8_E4M3_B64_BF16 as u8,
                0u16,
                1u16,
                0u32,
            ),
            (
                9u8,
                crate::flm::CODEC_SUPERSONIC_NATIVE_INT4_G128_BF16 as u8,
                0u16,
                1u16,
                0u32,
            ),
        ] {
            out.push(codec_id);
            out.push(semantic_id);
            push_u16(&mut out, layout_id);
            push_u16(&mut out, decoder_id);
            push_u32(&mut out, flags);
        }
        out
    }

    fn runtime_tensor_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, crate::flm::TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1);
        push_string(&mut out, "model.language_model");
        push_string(&mut out, ".weight_packed");
        push_string(&mut out, ".weight_scale");
        push_string(&mut out, ".weight_shape");
        out
    }

    fn runtime_asset_sections() -> (Vec<u8>, Vec<u8>) {
        let assets = [
            (
                1u32,
                crate::flm::ASSET_TOKENIZER_VOCAB,
                crate::flm::ASSET_FLAG_REQUIRED_FOR_RUNTIME,
                "tokenizer_vocab",
                b"vocab".as_slice(),
            ),
            (
                2u32,
                crate::flm::ASSET_TOKENIZER_MERGES,
                crate::flm::ASSET_FLAG_REQUIRED_FOR_RUNTIME,
                "tokenizer_merges",
                b"merges".as_slice(),
            ),
            (
                3u32,
                crate::flm::ASSET_TOKENIZER_ADDED_TOKENS,
                0,
                "tokenizer_added_tokens",
                b"[]".as_slice(),
            ),
            (
                4u32,
                crate::flm::ASSET_TOKENIZER_REGEX,
                crate::flm::ASSET_FLAG_REQUIRED_FOR_RUNTIME,
                "tokenizer_regex",
                br#"\p{L}+"#.as_slice(),
            ),
        ];
        let mut table = Vec::new();
        let mut payloads = Vec::new();
        push_u32(&mut table, assets.len() as u32);
        for (asset_id, kind_id, flags, name, payload) in assets {
            push_u32(&mut table, asset_id);
            push_u32(&mut table, payloads.len() as u32);
            push_u32(&mut table, payload.len() as u32);
            push_u16(&mut table, kind_id);
            push_u16(&mut table, flags);
            push_u32(&mut table, name.len() as u32);
            table.extend_from_slice(name.as_bytes());
            payloads.extend_from_slice(payload);
        }
        (table, payloads)
    }

    fn runtime_model_descriptor_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, crate::flm::MODEL_QWEN3_6_DENSE_V1);
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        push_u32(&mut out, crate::flm::TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1);
        push_u32(
            &mut out,
            crate::flm::QUANT_PROFILE_QWEN3_6_DENSE_CT_INT4_G128_BF16_V1,
        );
        push_u32(&mut out, 0);
        out
    }

    fn runtime_tensor_manifest_section(rows: &[TestManifestRow]) -> Vec<u8> {
        let mut string_pool = Vec::new();
        let mut names = Vec::with_capacity(rows.len());
        for row in rows {
            let offset = string_pool.len() as u32;
            string_pool.extend_from_slice(row.name.as_bytes());
            names.push((offset, row.name.len() as u16));
        }

        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 40);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, string_pool.len() as u32);
        for (row, (name_offset, name_len)) in rows.iter().zip(names) {
            push_u32(&mut out, row.role_id);
            push_u32(&mut out, row.group_id);
            out.push(row.companion_kind);
            out.push(row.rank);
            push_u16(&mut out, row.dtype);
            push_u16(&mut out, row.logical_dtype);
            out.push(row.codec_id);
            out.push(row.flags);
            for dim in row.shape {
                push_u32(&mut out, dim);
            }
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name_len);
            push_u16(&mut out, 0);
        }
        out.extend_from_slice(&string_pool);
        out
    }

    fn build_test_runtime_directory_with_manifest(rows: &[TestManifestRow]) -> Vec<u8> {
        let (asset_table, asset_payloads) = runtime_asset_sections();
        let sections = [
            (1u32, runtime_config_section()),
            (2u32, runtime_tokenizer_section()),
            (3u32, runtime_codec_section()),
            (4u32, runtime_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, runtime_model_descriptor_section()),
            (8u32, runtime_tensor_manifest_section(rows)),
        ];
        let header_len = 16 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        push_u16(&mut out, 4);
        push_u16(&mut out, sections.len() as u16);
        push_u32(&mut out, crate::flm::ARCH_QWEN3_6_DENSE);
        for (section_id, data) in &sections {
            push_u32(&mut out, *section_id);
            push_u32(&mut out, offset);
            push_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    fn build_test_runtime_directory() -> Vec<u8> {
        build_test_runtime_directory_with_manifest(&[])
    }

    fn runtime_stage3_storage_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 21);
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        push_u16(&mut out, 1);
        push_u16(&mut out, crate::flm::STORAGE_ABI_KIND_GROUP_QUANT);
        push_u16(&mut out, crate::flm::CODEC_SYM_INT4_G128_BF16);
        push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
        out.push(4);
        push_u16(&mut out, 128);
        push_u16(&mut out, crate::flm::QUANT_FLAG_SYMMETRIC);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        out
    }

    fn runtime_stage3_empty_storage_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 21);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        out
    }

    fn runtime_stage3_logical_tensor_section() -> Vec<u8> {
        let name = b"model.language_model.layers.0.mlp.gate_proj.weight";
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 44);
        push_u32(&mut out, 1);
        push_u32(&mut out, name.len() as u32);
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        push_u16(&mut out, name.len() as u16);
        push_u16(&mut out, crate::flm::LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT);
        out.push(2);
        out.push(0);
        for dim in [128u32, 64, 0, 0] {
            push_u32(&mut out, dim);
        }
        push_u16(&mut out, crate::flm::VALUE_FORMAT_SYM_INT4);
        push_u16(&mut out, FLM_DTYPE_BF16);
        push_u32(&mut out, 0);
        push_u16(&mut out, 3);
        push_u16(&mut out, crate::flm::LOGICAL_TENSOR_FLAG_REQUIRED);
        push_u16(&mut out, 0);
        out.extend_from_slice(name);
        out
    }

    fn runtime_stage3_raw_value_logical_tensor_section() -> Vec<u8> {
        let name = b"model.language_model.layers.0.linear_attn.A_log";
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 44);
        push_u32(&mut out, 1);
        push_u32(&mut out, name.len() as u32);
        push_u32(&mut out, 2);
        push_u32(&mut out, 0);
        push_u16(&mut out, name.len() as u16);
        push_u16(&mut out, crate::flm::LOGICAL_TENSOR_ROLE_WEIGHT);
        out.push(1);
        out.push(0);
        for dim in [4u32, 0, 0, 0] {
            push_u32(&mut out, dim);
        }
        push_u16(&mut out, crate::flm::VALUE_FORMAT_RAW_DENSE);
        push_u16(&mut out, FLM_DTYPE_FP32);
        push_u32(&mut out, 0);
        push_u16(&mut out, 1);
        push_u16(&mut out, crate::flm::LOGICAL_TENSOR_FLAG_REQUIRED);
        push_u16(&mut out, 0);
        out.extend_from_slice(name);
        out
    }

    fn runtime_stage3_storage_binding_section() -> Vec<u8> {
        let rows: [(&[u8], u16, u16); 3] = [
            (
                b"storage/l0_gate_packed",
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_INT32,
            ),
            (
                b"storage/l0_gate_scale",
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_BF16,
            ),
            (
                b"storage/l0_gate_shape",
                crate::flm::STORAGE_ROLE_SHAPE,
                FLM_DTYPE_INT64,
            ),
        ];
        let pool_len: usize = rows.iter().map(|(name, _, _)| name.len()).sum();
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 20);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, pool_len as u32);
        let mut pool = Vec::new();
        let mut name_offset = 0u32;
        for (name, role, dtype) in rows {
            push_u32(&mut out, 1);
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, role);
            push_u16(&mut out, dtype);
            push_u16(&mut out, 1);
            push_u16(&mut out, crate::flm::STORAGE_BINDING_FLAG_REQUIRED);
            push_u16(&mut out, 0);
            pool.extend_from_slice(name);
            name_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn runtime_stage3_raw_value_storage_binding_section() -> Vec<u8> {
        let name = b"storage/l0_a_log";
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 20);
        push_u32(&mut out, 1);
        push_u32(&mut out, name.len() as u32);
        push_u32(&mut out, 2);
        push_u32(&mut out, 0);
        push_u16(&mut out, name.len() as u16);
        push_u16(&mut out, crate::flm::STORAGE_ROLE_VALUE);
        push_u16(&mut out, FLM_DTYPE_FP32);
        push_u16(&mut out, crate::flm::STORAGE_ABI_ID_NONE);
        push_u16(&mut out, crate::flm::STORAGE_BINDING_FLAG_REQUIRED);
        push_u16(&mut out, 0);
        out.extend_from_slice(name);
        out
    }

    fn runtime_stage3_plan_step_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 38);
        push_u32(&mut out, 2);

        push_u32(&mut out, 1);
        push_u16(&mut out, crate::flm::STORAGE_ROLE_PACKED);
        push_u16(&mut out, crate::flm::CONSUME_STRATEGY_DIRECT);
        push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
        push_u16(&mut out, FLM_DTYPE_UINT8);
        out.push(2);
        out.push(0);
        for dim in [128u32, 32, 0, 0] {
            push_u32(&mut out, dim);
        }
        push_u16(&mut out, crate::flm::PLAN_STREAM_DEFAULT);
        push_u16(&mut out, crate::flm::PLAN_PRIORITY_DEFAULT);
        push_u32(&mut out, crate::flm::PLAN_STEP_FLAG_NONE);

        push_u32(&mut out, 1);
        push_u16(&mut out, crate::flm::STORAGE_ROLE_SCALE);
        push_u16(&mut out, crate::flm::CONSUME_STRATEGY_DIRECT);
        push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
        push_u16(&mut out, FLM_DTYPE_BF16);
        out.push(2);
        out.push(0);
        for dim in [128u32, 1, 0, 0] {
            push_u32(&mut out, dim);
        }
        push_u16(&mut out, crate::flm::PLAN_STREAM_DEFAULT);
        push_u16(&mut out, crate::flm::PLAN_PRIORITY_DEFAULT);
        push_u32(&mut out, crate::flm::PLAN_STEP_FLAG_NONE);
        out
    }

    fn runtime_stage3_raw_value_plan_step_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 38);
        push_u32(&mut out, 1);

        push_u32(&mut out, 2);
        push_u16(&mut out, crate::flm::STORAGE_ROLE_VALUE);
        push_u16(&mut out, crate::flm::CONSUME_STRATEGY_DIRECT);
        push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
        push_u16(&mut out, FLM_DTYPE_FP32);
        out.push(1);
        out.push(0);
        for dim in [4u32, 0, 0, 0] {
            push_u32(&mut out, dim);
        }
        push_u16(&mut out, crate::flm::PLAN_STREAM_DEFAULT);
        push_u16(&mut out, crate::flm::PLAN_PRIORITY_DEFAULT);
        push_u32(&mut out, crate::flm::PLAN_STEP_FLAG_NONE);
        out
    }

    fn runtime_stage3_native_int4_storage_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 21);
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        push_u16(&mut out, 8);
        push_u16(&mut out, crate::flm::STORAGE_ABI_KIND_GROUP_QUANT);
        push_u16(&mut out, crate::flm::CODEC_SUPERSONIC_NATIVE_INT4_G128_BF16);
        push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
        out.push(4);
        push_u16(&mut out, 128);
        push_u16(&mut out, crate::flm::QUANT_FLAG_SYMMETRIC);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
        out
    }

    fn runtime_stage3_native_int4_logical_tensor_section() -> Vec<u8> {
        let name = b"model.language_model.layers.0.mlp.experts.gate_up_proj";
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 44);
        push_u32(&mut out, 1);
        push_u32(&mut out, name.len() as u32);
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        push_u16(&mut out, name.len() as u16);
        push_u16(&mut out, crate::flm::LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT);
        out.push(3);
        out.push(0);
        for dim in [2u32, 256, 128, 0] {
            push_u32(&mut out, dim);
        }
        push_u16(&mut out, crate::flm::VALUE_FORMAT_SYM_INT4);
        push_u16(&mut out, FLM_DTYPE_BF16);
        push_u32(&mut out, 0);
        push_u16(&mut out, 3);
        push_u16(&mut out, crate::flm::LOGICAL_TENSOR_FLAG_REQUIRED);
        push_u16(&mut out, 0);
        out.extend_from_slice(name);
        out
    }

    fn runtime_stage3_native_int4_storage_binding_section() -> Vec<u8> {
        let rows: [(&[u8], u16, u16); 3] = [
            (
                b"model.language_model.layers.0.mlp.experts.gate_up_proj",
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
            ),
            (
                b"model.language_model.layers.0.mlp.experts.gate_up_proj_int4_scale",
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_BF16,
            ),
            (
                b"model.language_model.layers.0.mlp.experts.gate_up_proj_int4_zero",
                crate::flm::STORAGE_ROLE_ZERO,
                FLM_DTYPE_BF16,
            ),
        ];
        let pool_len: usize = rows.iter().map(|(name, _, _)| name.len()).sum();
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 20);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, pool_len as u32);
        let mut pool = Vec::new();
        let mut name_offset = 0u32;
        for (name, role, dtype) in rows {
            push_u32(&mut out, 1);
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, role);
            push_u16(&mut out, dtype);
            push_u16(&mut out, 8);
            push_u16(&mut out, crate::flm::STORAGE_BINDING_FLAG_REQUIRED);
            push_u16(&mut out, 0);
            pool.extend_from_slice(name);
            name_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn runtime_stage3_native_int4_plan_step_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 38);
        push_u32(&mut out, 3);

        for (role, dtype, rank, shape) in [
            (
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                3u8,
                [2u32, 256, 64, 0],
            ),
            (
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_BF16,
                3u8,
                [2u32, 2, 1, 0],
            ),
            (
                crate::flm::STORAGE_ROLE_ZERO,
                FLM_DTYPE_BF16,
                3u8,
                [2u32, 2, 1, 0],
            ),
        ] {
            push_u32(&mut out, 1);
            push_u16(&mut out, role);
            push_u16(&mut out, crate::flm::CONSUME_STRATEGY_DIRECT);
            push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
            push_u16(&mut out, dtype);
            out.push(rank);
            out.push(0);
            for dim in shape {
                push_u32(&mut out, dim);
            }
            push_u16(&mut out, crate::flm::PLAN_STREAM_DEFAULT);
            push_u16(&mut out, crate::flm::PLAN_PRIORITY_DEFAULT);
            push_u32(&mut out, crate::flm::PLAN_STEP_FLAG_NONE);
        }
        out
    }

    fn build_test_runtime_directory_with_stage3_tables() -> Vec<u8> {
        let (asset_table, asset_payloads) = runtime_asset_sections();
        let sections = [
            (1u32, runtime_config_section()),
            (2u32, runtime_tokenizer_section()),
            (3u32, runtime_codec_section()),
            (4u32, runtime_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, runtime_model_descriptor_section()),
            (8u32, runtime_tensor_manifest_section(&[])),
            (9u32, runtime_stage3_storage_abi_section()),
            (10u32, runtime_stage3_logical_tensor_section()),
            (11u32, runtime_stage3_storage_binding_section()),
            (12u32, runtime_stage3_plan_step_section()),
        ];
        let header_len = 16 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        push_u16(&mut out, 4);
        push_u16(&mut out, sections.len() as u16);
        push_u32(&mut out, crate::flm::ARCH_QWEN3_6_DENSE);
        for (section_id, data) in &sections {
            push_u32(&mut out, *section_id);
            push_u32(&mut out, offset);
            push_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    fn build_test_runtime_directory_with_native_int4_stage3_tables() -> Vec<u8> {
        let (asset_table, asset_payloads) = runtime_asset_sections();
        let sections = [
            (1u32, runtime_config_section()),
            (2u32, runtime_tokenizer_section()),
            (3u32, runtime_codec_section()),
            (4u32, runtime_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, runtime_model_descriptor_section()),
            (8u32, runtime_tensor_manifest_section(&[])),
            (9u32, runtime_stage3_native_int4_storage_abi_section()),
            (10u32, runtime_stage3_native_int4_logical_tensor_section()),
            (11u32, runtime_stage3_native_int4_storage_binding_section()),
            (12u32, runtime_stage3_native_int4_plan_step_section()),
        ];
        let header_len = 16 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        push_u16(&mut out, 4);
        push_u16(&mut out, sections.len() as u16);
        push_u32(&mut out, crate::flm::ARCH_QWEN3_6_DENSE);
        for (section_id, data) in &sections {
            push_u32(&mut out, *section_id);
            push_u32(&mut out, offset);
            push_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    fn build_test_runtime_directory_with_raw_value_stage3_tables() -> Vec<u8> {
        let (asset_table, asset_payloads) = runtime_asset_sections();
        let sections = [
            (1u32, runtime_config_section()),
            (2u32, runtime_tokenizer_section()),
            (3u32, runtime_codec_section()),
            (4u32, runtime_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, runtime_model_descriptor_section()),
            (8u32, runtime_tensor_manifest_section(&[])),
            (9u32, runtime_stage3_empty_storage_abi_section()),
            (10u32, runtime_stage3_raw_value_logical_tensor_section()),
            (11u32, runtime_stage3_raw_value_storage_binding_section()),
            (12u32, runtime_stage3_raw_value_plan_step_section()),
        ];
        let header_len = 16 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        push_u16(&mut out, 4);
        push_u16(&mut out, sections.len() as u16);
        push_u32(&mut out, crate::flm::ARCH_QWEN3_6_DENSE);
        for (section_id, data) in &sections {
            push_u32(&mut out, *section_id);
            push_u32(&mut out, offset);
            push_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    fn runtime_stage3_lowbit_storage_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 21);
        push_u32(&mut out, 5);
        push_u32(&mut out, 0);
        for (storage_abi_id, abi_kind, codec_semantic_id, bits, group_size, flags) in [
            (
                2u16,
                crate::flm::STORAGE_ABI_KIND_GROUP_QUANT,
                crate::flm::CODEC_NVFP4_E2M1_B16_E4M3_F32,
                4u8,
                16u16,
                crate::flm::QUANT_FLAG_SYMMETRIC,
            ),
            (
                3u16,
                crate::flm::STORAGE_ABI_KIND_GROUP_QUANT,
                crate::flm::CODEC_MXFP4_E2M1_B32_E8M0,
                4u8,
                32u16,
                crate::flm::QUANT_FLAG_SYMMETRIC,
            ),
            (
                4u16,
                crate::flm::STORAGE_ABI_KIND_GROUP_QUANT,
                crate::flm::CODEC_MXFP8_E4M3_B32_E8M0,
                8u8,
                32u16,
                crate::flm::QUANT_FLAG_SYMMETRIC,
            ),
            (
                5u16,
                crate::flm::STORAGE_ABI_KIND_SCALED_FLOAT,
                crate::flm::CODEC_FP8_E4M3_F32,
                8u8,
                0u16,
                0u16,
            ),
            (
                6u16,
                crate::flm::STORAGE_ABI_KIND_SCALED_FLOAT,
                crate::flm::CODEC_FP8_E4M3_B128_BF16_INV,
                8u8,
                128u16,
                0u16,
            ),
        ] {
            push_u16(&mut out, storage_abi_id);
            push_u16(&mut out, abi_kind);
            push_u16(&mut out, codec_semantic_id);
            push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
            out.push(bits);
            push_u16(&mut out, group_size);
            push_u16(&mut out, flags);
            push_u32(&mut out, 0);
            push_u32(&mut out, 0);
        }
        out
    }

    fn runtime_stage3_lowbit_logical_tensor_section() -> Vec<u8> {
        let rows = [
            (
                3u32,
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                crate::flm::VALUE_FORMAT_NVFP4_E2M1,
                [128u32, 128, 0, 0],
                0u32,
                3u16,
            ),
            (
                4u32,
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                crate::flm::VALUE_FORMAT_MXFP4_E2M1,
                [64u32, 128, 0, 0],
                3u32,
                2u16,
            ),
            (
                5u32,
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                crate::flm::VALUE_FORMAT_MXFP8_E4M3,
                [32u32, 128, 0, 0],
                5u32,
                2u16,
            ),
            (
                6u32,
                "model.language_model.layers.0.self_attn.q_proj.weight",
                crate::flm::VALUE_FORMAT_FP8_E4M3_B128_BF16_INV,
                [96u32, 128, 0, 0],
                7u32,
                2u16,
            ),
        ];
        let pool_len: usize = rows.iter().map(|(_, name, _, _, _, _)| name.len()).sum();
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 44);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, pool_len as u32);
        let mut pool = Vec::new();
        let mut name_offset = 0u32;
        for (tensor_id, name, value_format, shape, binding_start, binding_count) in rows {
            push_u32(&mut out, tensor_id);
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, crate::flm::LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT);
            out.push(2);
            out.push(0);
            for dim in shape {
                push_u32(&mut out, dim);
            }
            push_u16(&mut out, value_format);
            push_u16(&mut out, FLM_DTYPE_BF16);
            push_u32(&mut out, binding_start);
            push_u16(&mut out, binding_count);
            push_u16(&mut out, crate::flm::LOGICAL_TENSOR_FLAG_REQUIRED);
            push_u16(&mut out, 0);
            pool.extend_from_slice(name.as_bytes());
            name_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn runtime_stage3_lowbit_storage_binding_section() -> Vec<u8> {
        let rows: [(u32, &str, u16, u16, u16); 9] = [
            (
                3,
                "storage/nv_packed",
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                2,
            ),
            (
                3,
                "storage/nv_scale",
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_FP8_E4M3,
                2,
            ),
            (
                3,
                "storage/nv_global",
                crate::flm::STORAGE_ROLE_GLOBAL_SCALE,
                FLM_DTYPE_FP32,
                2,
            ),
            (
                4,
                "storage/mx4_packed",
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                3,
            ),
            (
                4,
                "storage/mx4_scale",
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_UINT8,
                3,
            ),
            (
                5,
                "storage/mx8_value",
                crate::flm::STORAGE_ROLE_VALUE,
                FLM_DTYPE_FP8_E4M3,
                4,
            ),
            (
                5,
                "storage/mx8_scale",
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_UINT8,
                4,
            ),
            (
                6,
                "storage/qwen_fp8_value",
                crate::flm::STORAGE_ROLE_VALUE,
                FLM_DTYPE_FP8_E4M3,
                6,
            ),
            (
                6,
                "storage/qwen_fp8_scale_inv",
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_BF16,
                6,
            ),
        ];
        let pool_len: usize = rows.iter().map(|(_, name, _, _, _)| name.len()).sum();
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 20);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, pool_len as u32);
        let mut pool = Vec::new();
        let mut name_offset = 0u32;
        for (logical_tensor_id, name, role, dtype, abi_id) in rows {
            push_u32(&mut out, logical_tensor_id);
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, role);
            push_u16(&mut out, dtype);
            push_u16(&mut out, abi_id);
            push_u16(&mut out, crate::flm::STORAGE_BINDING_FLAG_REQUIRED);
            push_u16(&mut out, 0);
            pool.extend_from_slice(name.as_bytes());
            name_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn runtime_stage3_lowbit_plan_step_section() -> Vec<u8> {
        let rows: [(u32, u16, u16, u8, [u32; 4]); 9] = [
            (
                3,
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                2,
                [128, 64, 0, 0],
            ),
            (
                3,
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_FP8_E4M3,
                2,
                [128, 8, 0, 0],
            ),
            (
                3,
                crate::flm::STORAGE_ROLE_GLOBAL_SCALE,
                FLM_DTYPE_FP32,
                1,
                [1, 0, 0, 0],
            ),
            (
                4,
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                2,
                [64, 64, 0, 0],
            ),
            (
                4,
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_UINT8,
                2,
                [64, 4, 0, 0],
            ),
            (
                5,
                crate::flm::STORAGE_ROLE_VALUE,
                FLM_DTYPE_FP8_E4M3,
                2,
                [32, 128, 0, 0],
            ),
            (
                5,
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_UINT8,
                2,
                [32, 4, 0, 0],
            ),
            (
                6,
                crate::flm::STORAGE_ROLE_VALUE,
                FLM_DTYPE_FP8_E4M3,
                2,
                [96, 128, 0, 0],
            ),
            (
                6,
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_BF16,
                2,
                [1, 1, 0, 0],
            ),
        ];
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 38);
        push_u32(&mut out, rows.len() as u32);
        for (logical_tensor_id, role, dtype, rank, shape) in rows {
            push_u32(&mut out, logical_tensor_id);
            push_u16(&mut out, role);
            push_u16(&mut out, crate::flm::CONSUME_STRATEGY_DIRECT);
            push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
            push_u16(&mut out, dtype);
            out.push(rank);
            out.push(0);
            for dim in shape {
                push_u32(&mut out, dim);
            }
            push_u16(&mut out, crate::flm::PLAN_STREAM_DEFAULT);
            push_u16(&mut out, crate::flm::PLAN_PRIORITY_DEFAULT);
            push_u32(&mut out, crate::flm::PLAN_STEP_FLAG_NONE);
        }
        out
    }

    fn build_test_runtime_directory_with_lowbit_stage3_tables() -> Vec<u8> {
        let (asset_table, asset_payloads) = runtime_asset_sections();
        let sections = [
            (1u32, runtime_config_section()),
            (2u32, runtime_tokenizer_section()),
            (3u32, runtime_codec_section()),
            (4u32, runtime_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, runtime_model_descriptor_section()),
            (8u32, runtime_tensor_manifest_section(&[])),
            (9u32, runtime_stage3_lowbit_storage_abi_section()),
            (10u32, runtime_stage3_lowbit_logical_tensor_section()),
            (11u32, runtime_stage3_lowbit_storage_binding_section()),
            (12u32, runtime_stage3_lowbit_plan_step_section()),
        ];
        let header_len = 16 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        push_u16(&mut out, 4);
        push_u16(&mut out, sections.len() as u16);
        push_u32(&mut out, crate::flm::ARCH_QWEN3_6_DENSE);
        for (section_id, data) in &sections {
            push_u32(&mut out, *section_id);
            push_u32(&mut out, offset);
            push_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    fn runtime_stage3_modelopt_nvfp4_logical_tensor_section() -> Vec<u8> {
        let rows = [
            (
                3u32,
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                crate::flm::VALUE_FORMAT_NVFP4_E2M1,
                [128u32, 128, 0, 0],
                0u32,
                4u16,
            ),
            (
                4u32,
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                crate::flm::VALUE_FORMAT_FP8_E4M3_F32,
                [32u32, 128, 0, 0],
                4u32,
                3u16,
            ),
        ];
        let pool_len: usize = rows.iter().map(|(_, name, _, _, _, _)| name.len()).sum();
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 44);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, pool_len as u32);
        let mut pool = Vec::new();
        let mut name_offset = 0u32;
        for (tensor_id, name, value_format, shape, binding_start, binding_count) in rows {
            push_u32(&mut out, tensor_id);
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, crate::flm::LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT);
            out.push(2);
            out.push(0);
            for dim in shape {
                push_u32(&mut out, dim);
            }
            push_u16(&mut out, value_format);
            push_u16(&mut out, FLM_DTYPE_BF16);
            push_u32(&mut out, binding_start);
            push_u16(&mut out, binding_count);
            push_u16(&mut out, crate::flm::LOGICAL_TENSOR_FLAG_REQUIRED);
            push_u16(&mut out, 0);
            pool.extend_from_slice(name.as_bytes());
            name_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn runtime_stage3_modelopt_nvfp4_storage_binding_section() -> Vec<u8> {
        let nv_base = "model.language_model.layers.0.linear_attn.out_proj";
        let fp8_base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let rows: [(u32, String, u16, u16, u16); 7] = [
            (
                3,
                format!("{nv_base}.weight"),
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                2,
            ),
            (
                3,
                format!("{nv_base}.weight_scale"),
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_FP8_E4M3,
                2,
            ),
            (
                3,
                format!("{nv_base}.weight_scale_2"),
                crate::flm::STORAGE_ROLE_GLOBAL_SCALE,
                FLM_DTYPE_FP32,
                2,
            ),
            (
                3,
                format!("{nv_base}.input_scale"),
                crate::flm::STORAGE_ROLE_INPUT_SCALE,
                FLM_DTYPE_FP32,
                2,
            ),
            (
                4,
                format!("{fp8_base}.weight"),
                crate::flm::STORAGE_ROLE_VALUE,
                FLM_DTYPE_FP8_E4M3,
                5,
            ),
            (
                4,
                format!("{fp8_base}.weight_scale"),
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_FP32,
                5,
            ),
            (
                4,
                format!("{fp8_base}.input_scale"),
                crate::flm::STORAGE_ROLE_INPUT_SCALE,
                FLM_DTYPE_FP32,
                5,
            ),
        ];
        let pool_len: usize = rows.iter().map(|(_, name, _, _, _)| name.len()).sum();
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 20);
        push_u32(&mut out, rows.len() as u32);
        push_u32(&mut out, pool_len as u32);
        let mut pool = Vec::new();
        let mut name_offset = 0u32;
        for (logical_tensor_id, name, role, dtype, abi_id) in rows {
            push_u32(&mut out, logical_tensor_id);
            push_u32(&mut out, name_offset);
            push_u16(&mut out, name.len() as u16);
            push_u16(&mut out, role);
            push_u16(&mut out, dtype);
            push_u16(&mut out, abi_id);
            push_u16(&mut out, crate::flm::STORAGE_BINDING_FLAG_REQUIRED);
            push_u16(&mut out, 0);
            pool.extend_from_slice(name.as_bytes());
            name_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn runtime_stage3_modelopt_nvfp4_plan_step_section() -> Vec<u8> {
        let rows: [(u32, u16, u16, u8, [u32; 4]); 7] = [
            (
                3,
                crate::flm::STORAGE_ROLE_PACKED,
                FLM_DTYPE_UINT8,
                2,
                [128, 64, 0, 0],
            ),
            (
                3,
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_FP8_E4M3,
                2,
                [128, 8, 0, 0],
            ),
            (
                3,
                crate::flm::STORAGE_ROLE_GLOBAL_SCALE,
                FLM_DTYPE_FP32,
                1,
                [1, 0, 0, 0],
            ),
            (
                3,
                crate::flm::STORAGE_ROLE_INPUT_SCALE,
                FLM_DTYPE_FP32,
                1,
                [1, 0, 0, 0],
            ),
            (
                4,
                crate::flm::STORAGE_ROLE_VALUE,
                FLM_DTYPE_FP8_E4M3,
                2,
                [32, 128, 0, 0],
            ),
            (
                4,
                crate::flm::STORAGE_ROLE_SCALE,
                FLM_DTYPE_FP32,
                1,
                [1, 0, 0, 0],
            ),
            (
                4,
                crate::flm::STORAGE_ROLE_INPUT_SCALE,
                FLM_DTYPE_FP32,
                1,
                [1, 0, 0, 0],
            ),
        ];
        let mut out = Vec::new();
        push_u16(&mut out, 1);
        push_u16(&mut out, 38);
        push_u32(&mut out, rows.len() as u32);
        for (logical_tensor_id, role, dtype, rank, shape) in rows {
            push_u32(&mut out, logical_tensor_id);
            push_u16(&mut out, role);
            push_u16(&mut out, crate::flm::CONSUME_STRATEGY_DIRECT);
            push_u16(&mut out, crate::flm::LAYOUT_ID_DEFAULT);
            push_u16(&mut out, dtype);
            out.push(rank);
            out.push(0);
            for dim in shape {
                push_u32(&mut out, dim);
            }
            push_u16(&mut out, crate::flm::PLAN_STREAM_DEFAULT);
            push_u16(&mut out, crate::flm::PLAN_PRIORITY_DEFAULT);
            push_u32(&mut out, crate::flm::PLAN_STEP_FLAG_NONE);
        }
        out
    }

    fn build_test_runtime_directory_with_modelopt_nvfp4_stage3_tables() -> Vec<u8> {
        let (asset_table, asset_payloads) = runtime_asset_sections();
        let sections = [
            (1u32, runtime_config_section()),
            (2u32, runtime_tokenizer_section()),
            (3u32, runtime_codec_section()),
            (4u32, runtime_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, runtime_model_descriptor_section()),
            (8u32, runtime_tensor_manifest_section(&[])),
            (9u32, runtime_stage3_lowbit_storage_abi_section()),
            (
                10u32,
                runtime_stage3_modelopt_nvfp4_logical_tensor_section(),
            ),
            (
                11u32,
                runtime_stage3_modelopt_nvfp4_storage_binding_section(),
            ),
            (12u32, runtime_stage3_modelopt_nvfp4_plan_step_section()),
        ];
        let header_len = 16 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        push_u16(&mut out, 4);
        push_u16(&mut out, sections.len() as u16);
        push_u32(&mut out, crate::flm::ARCH_QWEN3_6_DENSE);
        for (section_id, data) in &sections {
            push_u32(&mut out, *section_id);
            push_u32(&mut out, offset);
            push_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    fn build_test_flm_with_modelopt_nvfp4_stage3_bindings() -> Vec<u8> {
        let mut data = build_test_flm(&[
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.out_proj.weight",
                shape: vec![128, 64],
                dtype: FLM_DTYPE_UINT8,
                codec: 3,
                payload: vec![0x11; 128 * 64],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
                shape: vec![128, 8],
                dtype: FLM_DTYPE_FP8_E4M3,
                codec: 3,
                payload: vec![0x3f; 128 * 8],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.out_proj.weight_scale_2",
                shape: vec![1],
                dtype: FLM_DTYPE_FP32,
                codec: 0,
                payload: vec![0, 0, 128, 63],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.out_proj.input_scale",
                shape: vec![1],
                dtype: FLM_DTYPE_FP32,
                codec: 0,
                payload: vec![0, 0, 0, 64],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                shape: vec![32, 128],
                dtype: FLM_DTYPE_FP8_E4M3,
                codec: 6,
                payload: vec![0x33; 32 * 128],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
                shape: vec![1],
                dtype: FLM_DTYPE_FP32,
                codec: 6,
                payload: vec![0, 0, 0, 63],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale",
                shape: vec![1],
                dtype: FLM_DTYPE_FP32,
                codec: 0,
                payload: vec![0, 0, 64, 64],
            },
        ]);
        let runtime = build_test_runtime_directory_with_modelopt_nvfp4_stage3_tables();
        append_runtime_directory(&mut data, &runtime);
        data
    }

    fn build_test_flm_with_stage3_lowbit_bindings() -> Vec<u8> {
        let mut data = build_test_flm(&[
            TestFlmTensor {
                name: "storage/nv_packed",
                shape: vec![128, 64],
                dtype: FLM_DTYPE_UINT8,
                codec: 3,
                payload: vec![0x11; 128 * 64],
            },
            TestFlmTensor {
                name: "storage/nv_scale",
                shape: vec![128, 8],
                dtype: FLM_DTYPE_FP8_E4M3,
                codec: 3,
                payload: vec![0x3f; 128 * 8],
            },
            TestFlmTensor {
                name: "storage/nv_global",
                shape: vec![1],
                dtype: FLM_DTYPE_FP32,
                codec: 0,
                payload: vec![0, 0, 128, 63],
            },
            TestFlmTensor {
                name: "storage/mx4_packed",
                shape: vec![64, 64],
                dtype: FLM_DTYPE_UINT8,
                codec: 4,
                payload: vec![0x22; 64 * 64],
            },
            TestFlmTensor {
                name: "storage/mx4_scale",
                shape: vec![64, 4],
                dtype: FLM_DTYPE_UINT8,
                codec: 4,
                payload: vec![127; 64 * 4],
            },
            TestFlmTensor {
                name: "storage/mx8_value",
                shape: vec![32, 128],
                dtype: FLM_DTYPE_FP8_E4M3,
                codec: 5,
                payload: vec![0x33; 32 * 128],
            },
            TestFlmTensor {
                name: "storage/mx8_scale",
                shape: vec![32, 4],
                dtype: FLM_DTYPE_UINT8,
                codec: 5,
                payload: vec![127; 32 * 4],
            },
            TestFlmTensor {
                name: "storage/qwen_fp8_value",
                shape: vec![96, 128],
                dtype: FLM_DTYPE_FP8_E4M3,
                codec: 7,
                payload: vec![0x44; 96 * 128],
            },
            TestFlmTensor {
                name: "storage/qwen_fp8_scale_inv",
                shape: vec![1, 1],
                dtype: FLM_DTYPE_BF16,
                codec: 7,
                payload: vec![0, 63],
            },
        ]);
        let runtime = build_test_runtime_directory_with_lowbit_stage3_tables();
        append_runtime_directory(&mut data, &runtime);
        data
    }

    fn build_test_flm_with_stage3_logical_bindings() -> Vec<u8> {
        let mut data = build_test_flm(&[
            TestFlmTensor {
                name: "storage/l0_gate_packed",
                shape: vec![128, 8],
                dtype: FLM_DTYPE_INT32,
                codec: 1,
                payload: test_ct_int4_packed_bytes(128, 64, |_row, col| (col % 16) as i32 - 8),
            },
            TestFlmTensor {
                name: "storage/l0_gate_scale",
                shape: vec![128, 1],
                dtype: FLM_DTYPE_BF16,
                codec: 1,
                payload: test_bf16_bytes((0..128).map(|row| if row % 2 == 0 { 1.0 } else { 0.5 })),
            },
            TestFlmTensor {
                name: "storage/l0_gate_shape",
                shape: vec![2],
                dtype: FLM_DTYPE_INT64,
                codec: 2,
                payload: vec![0; 16],
            },
        ]);
        let runtime = build_test_runtime_directory_with_stage3_tables();
        append_runtime_directory(&mut data, &runtime);
        data
    }

    fn build_test_flm_with_stage3_native_int4_bindings() -> Vec<u8> {
        let mut data = build_test_flm(&[
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.experts.gate_up_proj",
                shape: vec![2, 256, 64],
                dtype: FLM_DTYPE_UINT8,
                codec: 9,
                payload: vec![0xab; 2 * 256 * 64],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.experts.gate_up_proj_int4_scale",
                shape: vec![2, 2, 1],
                dtype: FLM_DTYPE_BF16,
                codec: 0,
                payload: test_bf16_bytes([1.0, 2.0, 3.0, 4.0]),
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.experts.gate_up_proj_int4_zero",
                shape: vec![2, 2, 1],
                dtype: FLM_DTYPE_BF16,
                codec: 0,
                payload: test_bf16_bytes([8.0; 4]),
            },
        ]);
        let runtime = build_test_runtime_directory_with_native_int4_stage3_tables();
        append_runtime_directory(&mut data, &runtime);
        data
    }

    fn build_test_flm_with_stage3_raw_value_binding() -> Vec<u8> {
        let mut data = build_test_flm(&[TestFlmTensor {
            name: "storage/l0_a_log",
            shape: vec![4],
            dtype: FLM_DTYPE_FP32,
            codec: 0,
            payload: vec![0; 16],
        }]);
        let runtime = build_test_runtime_directory_with_raw_value_stage3_tables();
        append_runtime_directory(&mut data, &runtime);
        data
    }

    fn append_runtime_directory(data: &mut Vec<u8>, runtime: &[u8]) {
        let runtime_offset = data.len();
        data.extend_from_slice(runtime);
        put_u64(data, 168, runtime_offset as u64);
        put_u64(data, 176, runtime.len() as u64);
        put_head_crc64(data);
    }

    fn build_test_flm_with_runtime_manifest_group() -> Vec<u8> {
        let mut data = build_test_flm(&[
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_packed",
                shape: vec![128, 16],
                dtype: FLM_DTYPE_INT32,
                codec: 1,
                payload: vec![0; 128],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_scale",
                shape: vec![128, 1],
                dtype: FLM_DTYPE_BF16,
                codec: 1,
                payload: vec![0; 256],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_shape",
                shape: vec![2],
                dtype: FLM_DTYPE_INT64,
                codec: 2,
                payload: vec![0; 16],
            },
        ]);
        let runtime = build_test_runtime_directory_with_manifest(&[
            TestManifestRow {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_packed",
                role_id: 1,
                group_id: 7,
                companion_kind: crate::flm::MANIFEST_COMPANION_PACKED,
                rank: 2,
                dtype: FLM_DTYPE_INT32,
                logical_dtype: FLM_DTYPE_UINT8,
                codec_id: 1,
                flags: crate::flm::MANIFEST_FLAG_REQUIRED,
                shape: [128, 16, 0, 0],
            },
            TestManifestRow {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_scale",
                role_id: 1,
                group_id: 7,
                companion_kind: crate::flm::MANIFEST_COMPANION_SCALE,
                rank: 2,
                dtype: FLM_DTYPE_BF16,
                logical_dtype: FLM_DTYPE_BF16,
                codec_id: 1,
                flags: crate::flm::MANIFEST_FLAG_REQUIRED,
                shape: [128, 1, 0, 0],
            },
            TestManifestRow {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_shape",
                role_id: 1,
                group_id: 7,
                companion_kind: crate::flm::MANIFEST_COMPANION_SHAPE,
                rank: 1,
                dtype: FLM_DTYPE_INT64,
                logical_dtype: FLM_DTYPE_INT64,
                codec_id: 2,
                flags: crate::flm::MANIFEST_FLAG_REQUIRED,
                shape: [2, 0, 0, 0],
            },
        ]);
        append_runtime_directory(&mut data, &runtime);
        data
    }

    fn corrupt_first_manifest_shape(data: &mut [u8], value: u32) {
        let manifest_offset = runtime_section_offset(data, 8);
        let shape_offset = manifest_offset + 12 + 16;
        put_u32(data, shape_offset, value);
    }

    fn runtime_section_offset(data: &[u8], wanted_section_id: u32) -> usize {
        let runtime_offset =
            read_u64(data, 168, "test FLM runtime offset").expect("runtime offset") as usize;
        let runtime = &data[runtime_offset..];
        let section_count =
            read_u16(runtime, 10, "test FLM section count").expect("section count") as usize;
        for idx in 0..section_count {
            let record = 16 + idx * 12;
            let section_id = read_u32(runtime, record, "test FLM section id").expect("section id");
            if section_id == wanted_section_id {
                let section_offset = read_u32(runtime, record + 4, "test FLM section offset")
                    .expect("section offset") as usize;
                return runtime_offset + section_offset;
            }
        }
        panic!("missing runtime section {wanted_section_id}");
    }

    fn put_first_storage_abi_codec_semantic(data: &mut [u8], codec_semantic_id: u16) {
        let storage_abi_offset = runtime_section_offset(data, 9);
        let row_offset = storage_abi_offset + 12;
        let codec_semantic_offset = row_offset + 4;
        put_u16(data, codec_semantic_offset, codec_semantic_id);
    }

    fn write_temp_flm(data: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temp FLM file");
        file.write_all(data).expect("write FLM fixture");
        file.flush().expect("flush FLM fixture");
        file
    }

    #[test]
    fn flm_crc64_ecma_matches_standard_check_vector() {
        assert_eq!(flm_crc64_ecma(b"123456789"), 0x6C40_DF5F_0B49_7347);
    }

    #[test]
    fn open_flm_exposes_mmap_backed_tensor_bytes() {
        let data = build_test_flm(&[
            TestFlmTensor {
                name: "model.embed_tokens.weight",
                shape: vec![2, 4],
                dtype: 4,
                codec: 0,
                payload: (0u8..8).collect(),
            },
            TestFlmTensor {
                name: "model.norm.weight",
                shape: vec![2],
                dtype: 2,
                codec: 0,
                payload: vec![0x00, 0x3f, 0x00, 0x40],
            },
        ]);
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm(file.path()).expect("open FLM");

        assert!(store.flm_runtime().is_none());
        assert!(store.contains("model.embed_tokens.weight"));
        let meta = store.meta("model.embed_tokens.weight").unwrap();
        assert_eq!(meta.shape, vec![2, 4]);
        assert_eq!(meta.dtype, "u8");
        assert_eq!(meta.layout, LayoutTag::Raw);
        assert_eq!(
            store.raw_bytes("model.embed_tokens.weight").unwrap(),
            &(0u8..8).collect::<Vec<_>>()
        );
        assert_eq!(
            store
                .raw_byte_range("model.embed_tokens.weight", 2, 3)
                .unwrap(),
            &[2, 3, 4]
        );

        let norm = store.meta("model.norm.weight").unwrap();
        assert_eq!(norm.dtype, "bf16");
        assert_eq!(norm.shape, vec![2]);
    }

    #[test]
    fn open_flm_exposes_file_storage_extent_for_direct_tensor() {
        let data = build_test_flm(&[
            TestFlmTensor {
                name: "model.embed_tokens.weight",
                shape: vec![2, 4],
                dtype: 4,
                codec: 0,
                payload: (0u8..8).collect(),
            },
            TestFlmTensor {
                name: "model.norm.weight",
                shape: vec![2],
                dtype: 2,
                codec: 0,
                payload: vec![0x00, 0x3f, 0x00, 0x40],
            },
        ]);
        let file = write_temp_flm(&data);
        let store = BakedStore::open_flm(file.path()).expect("open FLM");
        let meta = store.meta("model.norm.weight").expect("norm meta");

        let extent = store
            .tensor_storage_extent("model.norm.weight")
            .expect("storage extent")
            .expect("direct FLM tensor should expose a file extent");

        assert_eq!(extent.source_kind, TensorStorageSourceKind::FlmContainer);
        assert_eq!(extent.source_path, file.path());
        assert_eq!(extent.name, "model.norm.weight");
        assert_eq!(extent.file_offset, meta.offset);
        assert_eq!(extent.byte_len, meta.byte_len);
        assert_eq!(extent.storage_dtype, "bf16");
        assert_eq!(extent.storage_shape, vec![2]);
        assert_eq!(extent.layout, LayoutTag::Raw);
        assert_eq!(extent.upload_dtype, "bf16");
        assert_eq!(extent.upload_shape, vec![2]);
    }

    #[test]
    fn open_flm_exposes_file_storage_range_for_direct_tensor_slice() {
        let data = build_test_flm(&[TestFlmTensor {
            name: "model.norm.weight",
            shape: vec![4],
            dtype: 2,
            codec: 0,
            payload: vec![0x00, 0x3f, 0x00, 0x40, 0x00, 0x41, 0x00, 0x42],
        }]);
        let file = write_temp_flm(&data);
        let store = BakedStore::open_flm(file.path()).expect("open FLM");
        let meta = store.meta("model.norm.weight").expect("norm meta");

        let range = store
            .tensor_storage_range("model.norm.weight", 2, 4)
            .expect("storage range")
            .expect("direct FLM tensor slice should expose a file range");

        assert_eq!(
            range.extent.source_kind,
            TensorStorageSourceKind::FlmContainer
        );
        assert_eq!(range.extent.source_path, file.path());
        assert_eq!(range.extent.name, "model.norm.weight");
        assert_eq!(range.extent.file_offset, meta.offset);
        assert_eq!(range.extent.byte_len, meta.byte_len);
        assert_eq!(range.tensor_byte_offset, 2);
        assert_eq!(range.byte_len, 4);
        assert_eq!(range.file_offset, meta.offset + 2);
        assert_eq!(range.extent.storage_dtype, "bf16");
        assert_eq!(range.extent.storage_shape, vec![4]);
        assert_eq!(range.extent.layout, LayoutTag::Raw);
        assert_eq!(range.extent.upload_dtype, "bf16");
        assert_eq!(range.extent.upload_shape, vec![4]);
    }

    #[test]
    fn host_registration_range_is_bounded_by_page_rounded_mmap() {
        let range = host_registration_range_for_mmap_slice(0x1000, 0x2500, 0x2234, 0x40, 4096)
            .expect("registration range");

        assert_eq!(range.ptr as usize, 0x2000);
        assert_eq!(range.len, 0x1000);
        assert_eq!(range.data_offset, 0x234);

        let err = host_registration_range_for_mmap_slice(0x1000, 0x100, 0x0fff, 0x20, 4096)
            .expect_err("slice outside mmap should fail");
        assert!(
            err.to_string().contains("outside mmap backing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_flm_rejects_bad_head_crc64() {
        let mut data = build_test_flm(&[TestFlmTensor {
            name: "model.embed_tokens.weight",
            shape: vec![4],
            dtype: 4,
            codec: 0,
            payload: b"good".to_vec(),
        }]);
        put_u64(&mut data, 144, 1);
        let file = write_temp_flm(&data);

        let err = match BakedStore::open_flm(file.path()) {
            Ok(_) => panic!("bad FLM head CRC64 should fail verification"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("head CRC"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_flm_parses_runtime_directory_when_offsets_present() {
        let mut data = build_test_flm(&[TestFlmTensor {
            name: "model.embed_tokens.weight",
            shape: vec![2, 4],
            dtype: 4,
            codec: 0,
            payload: (0u8..8).collect(),
        }]);
        let runtime = build_test_runtime_directory();
        let runtime_offset = data.len();
        data.extend_from_slice(&runtime);
        put_u64(&mut data, 168, runtime_offset as u64);
        put_u64(&mut data, 176, runtime.len() as u64);
        put_head_crc64(&mut data);
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm(file.path()).expect("open FLM with runtime");
        let runtime = store.flm_runtime().expect("parsed runtime");

        assert_eq!(runtime.qwen36_config().unwrap().hidden_size, 5120);
        assert_eq!(runtime.tokenizer().unwrap().tokenizer_id, 0);
        assert_eq!(runtime.codec_by_id(0).unwrap().layout_id, 0);
        assert_eq!(runtime.codec_by_id(0).unwrap().decoder_id, 0);
        assert_eq!(runtime.codec_by_id(1).unwrap().layout_id, 0);
        assert_eq!(runtime.codec_by_id(1).unwrap().decoder_id, 1);
        assert_eq!(runtime.codec_by_id(2).unwrap().layout_id, 0);
        assert_eq!(runtime.codec_by_id(2).unwrap().decoder_id, 0);
        assert_eq!(runtime.tensor_abi().weight_prefix, "model.language_model");
        assert_eq!(
            runtime.asset_by_kind("tokenizer_regex").unwrap().asset_id,
            4
        );
    }

    #[test]
    fn read_flm_runtime_identity_reads_model_descriptor_without_store_open() {
        let mut data = build_test_flm(&[TestFlmTensor {
            name: "model.embed_tokens.weight",
            shape: vec![2, 4],
            dtype: 4,
            codec: 0,
            payload: (0u8..8).collect(),
        }]);
        let runtime = build_test_runtime_directory();
        append_runtime_directory(&mut data, &runtime);
        let file = write_temp_flm(&data);

        let identity = read_flm_runtime_identity(file.path())
            .expect("read runtime identity")
            .unwrap();

        assert_eq!(identity.architecture_id, crate::flm::ARCH_QWEN3_6_DENSE);
        assert_eq!(identity.model_id, crate::flm::MODEL_QWEN3_6_DENSE_V1);
    }

    #[test]
    fn open_flm_verify_hashes_rejects_tampered_payload() {
        let mut data = build_test_flm_with_hashes(
            &[TestFlmTensor {
                name: "model.embed_tokens.weight",
                shape: vec![4],
                dtype: 4,
                codec: 0,
                payload: b"good".to_vec(),
            }],
            &[[
                0x4a, 0xe7, 0x5b, 0x23, 0x49, 0xdb, 0x30, 0x92, 0xa6, 0xa3, 0x34, 0x36, 0x25, 0xa6,
                0xaa, 0x7b, 0xa4, 0xbb, 0x32, 0x98, 0x49, 0x1a, 0xca, 0xc2, 0x02, 0xc4, 0x68, 0x47,
                0xb9, 0xef, 0x31, 0xc8,
            ]],
        );
        let shard_table_offset = read_u64(&data, 88, "test FLM shard table offset")
            .expect("shard table offset") as usize;
        let payload_offset = read_u64(&data, shard_table_offset + 4, "test FLM shard file offset")
            .expect("shard file offset") as usize;
        data[payload_offset] ^= 0xff;
        let file = write_temp_flm(&data);

        let err = match BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: false,
                verify_block_hashes: true,
            },
        ) {
            Ok(_) => panic!("tampered FLM payload should fail verification"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("hash mismatch"),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string().contains("model.embed_tokens.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_flm_rejects_half_zero_runtime_directory_fields() {
        let mut offset_only = build_test_flm(&[TestFlmTensor {
            name: "model.embed_tokens.weight",
            shape: vec![2, 4],
            dtype: 4,
            codec: 0,
            payload: (0u8..8).collect(),
        }]);
        put_u64(&mut offset_only, 168, 4096);
        put_head_crc64(&mut offset_only);
        let file = write_temp_flm(&offset_only);
        let err = match BakedStore::open_flm(file.path()) {
            Ok(_) => panic!("zero runtime len should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("length is zero"),
            "unexpected error: {err}"
        );

        let mut len_only = build_test_flm(&[TestFlmTensor {
            name: "model.embed_tokens.weight",
            shape: vec![2, 4],
            dtype: 4,
            codec: 0,
            payload: (0u8..8).collect(),
        }]);
        put_u64(&mut len_only, 176, 16);
        put_head_crc64(&mut len_only);
        let file = write_temp_flm(&len_only);
        let err = match BakedStore::open_flm(file.path()) {
            Ok(_) => panic!("zero runtime offset should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("offset is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_flm_builds_int4_aliases_from_manifest_groups() {
        let data = build_test_flm_with_runtime_manifest_group();
        let file = write_temp_flm(&data);
        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with manifest aliases");

        let alias = store
            .meta("model.language_model.layers.0.mlp.gate_proj.weight")
            .expect("native alias");
        assert_eq!(alias.layout, LayoutTag::Int4Quantized);
        assert_eq!(alias.dtype, "u8");
        assert_eq!(alias.shape, vec![128, 64]);
        let scale = store
            .meta("model.language_model.layers.0.mlp.gate_proj.weight_int4_scale")
            .expect("native scale alias");
        assert_eq!(scale.dtype, "bf16");
        assert_eq!(scale.shape, vec![128, 1]);
    }

    #[test]
    fn open_flm_builds_ct_int4_stage3_binding_as_bf16_fallback() {
        let data = build_test_flm_with_stage3_logical_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with Stage 3 bindings");

        let alias = store
            .meta("model.language_model.layers.0.mlp.gate_proj.weight")
            .expect("logical INT4 fallback alias");
        assert_eq!(alias.layout, LayoutTag::Raw);
        assert_eq!(alias.dtype, "bf16");
        assert_eq!(alias.shape, vec![128, 64]);
        assert_eq!(alias.byte_len, 128 * 64 * 2);

        let upload_view = store
            .upload_view("model.language_model.layers.0.mlp.gate_proj.weight")
            .expect("logical INT4 fallback upload view");
        assert_eq!(upload_view.dtype, "bf16");
        assert_eq!(upload_view.shape, vec![128, 64]);
    }

    #[test]
    fn open_flm_builds_native_int4_stage3_binding_with_zero_plane() {
        let data = build_test_flm_with_stage3_native_int4_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with native INT4 Stage 3 bindings");

        let name = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let weight = store.meta(name).expect("native INT4 logical tensor");
        assert_eq!(weight.layout, LayoutTag::Int4Quantized);
        assert_eq!(weight.dtype, "u8");
        assert_eq!(weight.shape, vec![2, 256, 64]);
        assert_eq!(weight.byte_len, (2 * 256 * 64) as u64);

        let upload_view = store.upload_view(name).expect("native INT4 upload view");
        assert_eq!(upload_view.dtype, "u8");
        assert_eq!(upload_view.shape, vec![2, 256, 64]);

        let scale_name = format!("{name}_int4_scale");
        let scale = store.meta(&scale_name).expect("native INT4 scale plane");
        assert_eq!(scale.dtype, "bf16");
        assert_eq!(scale.shape, vec![2, 2, 1]);

        let zero_name = format!("{name}_int4_zero");
        let zero = store.meta(&zero_name).expect("native INT4 zero plane");
        assert_eq!(zero.dtype, "bf16");
        assert_eq!(zero.shape, vec![2, 2, 1]);
        assert_eq!(
            store.raw_bytes(&zero_name).expect("native INT4 zero bytes"),
            test_bf16_bytes([8.0; 4]).as_slice()
        );
    }

    #[test]
    fn flm_runtime_classifies_ct_int4_stage3_plan_as_bf16_fallback() {
        let data = build_test_flm_with_stage3_logical_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open CT INT4 Stage 3 FLM");

        let kind = store
            .flm_runtime()
            .expect("runtime directory")
            .stage3_direct_weight_kind("model.language_model.layers.0.mlp.gate_proj.weight")
            .expect("classify direct weight kind");

        assert_eq!(
            kind,
            Some(crate::flm::FlmStage3DirectWeightKind::CtInt4Bf16Fallback)
        );
        assert!(
            store
                .tensor_storage_extent("model.language_model.layers.0.mlp.gate_proj.weight")
                .expect("CT fallback storage extent")
                .is_none(),
            "CT fallback aliases are materialized transforms and must not expose a single file extent"
        );
    }

    #[test]
    fn flm_runtime_classifies_native_int4_stage3_plan() {
        let data = build_test_flm_with_stage3_native_int4_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open native INT4 Stage 3 FLM");

        let kind = store
            .flm_runtime()
            .expect("runtime directory")
            .stage3_direct_weight_kind("model.language_model.layers.0.mlp.experts.gate_up_proj")
            .expect("classify direct weight kind");

        assert_eq!(
            kind,
            Some(crate::flm::FlmStage3DirectWeightKind::NativeInt4)
        );
        let extent = store
            .tensor_storage_extent("model.language_model.layers.0.mlp.experts.gate_up_proj")
            .expect("native INT4 extent")
            .expect("native INT4 direct tensor should expose file extent");
        assert_eq!(extent.source_kind, TensorStorageSourceKind::FlmContainer);
        assert_eq!(extent.source_path, file.path());
        assert_eq!(extent.storage_dtype, "u8");
        assert_eq!(extent.storage_shape, vec![2, 256, 64]);
        assert_eq!(extent.layout, LayoutTag::Int4Quantized);
        assert_eq!(extent.upload_dtype, "u8");
        assert_eq!(extent.upload_shape, vec![2, 256, 64]);
    }

    #[test]
    fn flm_runtime_classifies_raw_dense_stage3_plan() {
        let data = build_test_flm_with_stage3_raw_value_binding();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open raw dense Stage 3 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let kind = runtime
            .stage3_direct_weight_kind("model.language_model.layers.0.linear_attn.A_log")
            .expect("classify direct weight kind");

        assert_eq!(kind, Some(crate::flm::FlmStage3DirectWeightKind::RawDense));
        assert_eq!(
            runtime
                .stage3_direct_weight_kind("model.language_model.layers.0.missing.weight")
                .expect("missing logical tensor is not an error"),
            None
        );
    }

    #[test]
    fn open_flm_rejects_native_int4_stage3_binding_with_non_native_abi() {
        let mut data = build_test_flm_with_stage3_native_int4_bindings();
        put_first_storage_abi_codec_semantic(&mut data, crate::flm::CODEC_SYM_INT4_G128_BF16);
        let file = write_temp_flm(&data);

        let err = match BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        ) {
            Ok(_) => panic!("native INT4 Stage 3 binding with non-native ABI should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("unsupported native INT4 ABI"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_flm_materializes_ct_int4_stage3_fallback_upload_bytes() {
        let data = build_test_flm_with_stage3_logical_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with Stage 3 bindings");

        let logical_name = "model.language_model.layers.0.mlp.gate_proj.weight";
        let (dtype, shape, bytes) = store
            .materialize_upload_for_test(logical_name)
            .expect("materialize CT INT4 BF16 fallback");
        assert_eq!(dtype, ScalarType::BF16);
        assert_eq!(shape, vec![128, 64]);

        let expected = test_bf16_bytes((0..128).flat_map(|row| {
            (0..64).map(move |col| {
                let code = (col % 16) as i32 - 8;
                let scale = if row % 2 == 0 { 1.0 } else { 0.5 };
                code as f32 * scale
            })
        }));
        assert_eq!(bytes, expected);
    }

    #[test]
    fn open_flm_rejects_virtual_arena_for_ct_int4_stage3_fallback_alias() {
        let data = build_test_flm_with_stage3_logical_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with Stage 3 bindings");

        let mut arena = BakedStore::virtual_weight_arena(0);
        let err = store
            .reserve_virtual_arena(
                &mut arena,
                "model.language_model.layers.0.mlp.gate_proj.weight",
                VirtualAllocationRole::Weights,
            )
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("transformed FLM logical alias"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_flm_rejects_virtual_arena_for_synthesized_manifest_alias() {
        let data = build_test_flm_with_runtime_manifest_group();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            crate::store::FlmLoadOptions {
                flm_int4_logical_aliases: true,
                ..Default::default()
            },
        )
        .expect("open FLM with manifest aliases");

        let mut arena = BakedStore::virtual_weight_arena(0);
        let err = store
            .reserve_virtual_arena(
                &mut arena,
                "model.language_model.layers.0.mlp.gate_proj.weight_int4_zero",
                VirtualAllocationRole::Weights,
            )
            .expect_err("synthetic aliases must not reserve virtual direct storage")
            .to_string();

        assert!(
            err.contains("not a direct file-backed extent"),
            "unexpected error: {err}"
        );
        assert_eq!(arena.stats().allocations, 0);
    }

    #[test]
    fn open_flm_builds_raw_fp32_value_alias_from_stage3_binding() {
        let data = build_test_flm_with_stage3_raw_value_binding();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with raw Stage 3 value binding");

        let storage = store.meta("storage/l0_a_log").expect("storage tensor");
        assert_eq!(storage.dtype, "f32");
        assert_eq!(storage.shape, vec![4]);

        let logical_name = "model.language_model.layers.0.linear_attn.A_log";
        let alias = store.meta(logical_name).expect("raw logical value alias");
        assert_eq!(alias.layout, LayoutTag::Raw);
        assert_eq!(alias.dtype, "f32");
        assert_eq!(alias.shape, vec![4]);
        assert_eq!(alias.offset, storage.offset);
        assert_eq!(alias.byte_len, 16);

        let upload_view = store
            .upload_view(logical_name)
            .expect("raw logical value upload view");
        assert_eq!(upload_view.dtype, "f32");
        assert_eq!(upload_view.shape, vec![4]);
    }

    #[test]
    fn stage3_raw_value_alias_layout_marks_qwen_linear_attention_runtime_shapes() {
        assert_eq!(
            stage3_raw_value_alias_layout(
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                &[8192, 4],
            ),
            LayoutTag::DepthwiseConvSqueezed
        );
        assert_eq!(
            stage3_raw_value_alias_layout(
                "model.language_model.layers.0.linear_attn.dt_bias",
                &[1, 1, 32],
            ),
            LayoutTag::HeadBiasReshaped
        );
        assert_eq!(
            stage3_raw_value_alias_layout(
                "model.language_model.layers.0.linear_attn.A_log",
                &[1, 1, 32],
            ),
            LayoutTag::HeadExpReshaped
        );
        assert_eq!(
            stage3_raw_value_alias_layout("model.language_model.layers.0.linear_attn.A_log", &[32],),
            LayoutTag::Raw
        );
    }

    #[test]
    fn open_flm_builds_nvfp4_and_mxfp_stage3_direct_views() {
        let data = build_test_flm_with_stage3_lowbit_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with low-bit Stage 3 bindings");

        let nv_name = "model.language_model.layers.0.linear_attn.out_proj.weight";
        let nv = store.meta(nv_name).expect("NVFP4 logical alias");
        let nv_storage = store.meta("storage/nv_packed").expect("NVFP4 storage");
        assert_eq!(nv.layout, LayoutTag::Raw);
        assert_eq!(nv.dtype, "u8");
        assert_eq!(nv.shape, vec![128, 128]);
        assert_eq!(nv.offset, nv_storage.offset);
        assert_eq!(nv.byte_len, nv_storage.byte_len);
        assert_eq!(
            store.upload_view(nv_name).expect("NVFP4 upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![128, 64],
            }
        );
        let nv_scale = store
            .meta("model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_scale")
            .expect("NVFP4 scale alias");
        assert_eq!(nv_scale.dtype, "f8_e4m3");
        assert_eq!(nv_scale.shape, vec![128, 8]);
        let nv_global = store
            .meta("model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_global_scale")
            .expect("NVFP4 global scale alias");
        assert_eq!(nv_global.dtype, "f32");
        assert_eq!(nv_global.shape, vec![1]);

        let mx4_name = "model.language_model.layers.0.linear_attn.in_proj_z.weight";
        let mx4 = store.meta(mx4_name).expect("MXFP4 logical alias");
        assert_eq!(mx4.layout, LayoutTag::Raw);
        assert_eq!(mx4.dtype, "u8");
        assert_eq!(mx4.shape, vec![64, 128]);
        assert_eq!(
            store.upload_view(mx4_name).expect("MXFP4 upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![64, 64],
            }
        );
        let mx4_scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_z.weight_mxfp4_scale")
            .expect("MXFP4 scale alias");
        assert_eq!(mx4_scale.dtype, "u8");
        assert_eq!(mx4_scale.shape, vec![64, 4]);

        let mx8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let mx8 = store.meta(mx8_name).expect("MXFP8 logical alias");
        assert_eq!(mx8.layout, LayoutTag::Raw);
        assert_eq!(mx8.dtype, "f8_e4m3");
        assert_eq!(mx8.shape, vec![32, 128]);
        assert_eq!(
            store.upload_view(mx8_name).expect("MXFP8 upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![32, 128],
            }
        );
        let mx8_scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_mxfp8_scale")
            .expect("MXFP8 scale alias");
        assert_eq!(mx8_scale.dtype, "u8");
        assert_eq!(mx8_scale.shape, vec![32, 4]);

        let qwen_fp8_name = "model.language_model.layers.0.self_attn.q_proj.weight";
        let qwen_fp8 = store
            .meta(qwen_fp8_name)
            .expect("Qwen FP8 block-scale logical alias");
        assert_eq!(qwen_fp8.layout, LayoutTag::Raw);
        assert_eq!(qwen_fp8.dtype, "f8_e4m3");
        assert_eq!(qwen_fp8.shape, vec![96, 128]);
        assert_eq!(
            store
                .upload_view(qwen_fp8_name)
                .expect("Qwen FP8 block-scale upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![96, 128],
            }
        );
        let qwen_fp8_scale = store
            .meta("model.language_model.layers.0.self_attn.q_proj.weight_fp8_e4m3_b128_bf16_scale_inv")
            .expect("Qwen FP8 block-scale inverse scale alias");
        assert_eq!(qwen_fp8_scale.dtype, "bf16");
        assert_eq!(qwen_fp8_scale.shape, vec![1, 1]);

        let runtime = store.flm_runtime().expect("runtime directory");
        let by_name = runtime
            .logical_tensors()
            .iter()
            .map(|logical| (logical.name.as_str(), logical.value_format_id))
            .collect::<HashMap<_, _>>();
        assert_eq!(by_name[nv_name], crate::flm::VALUE_FORMAT_NVFP4_E2M1);
        assert_eq!(by_name[mx4_name], crate::flm::VALUE_FORMAT_MXFP4_E2M1);
        assert_eq!(by_name[mx8_name], crate::flm::VALUE_FORMAT_MXFP8_E4M3);
        assert_eq!(
            by_name[qwen_fp8_name],
            crate::flm::VALUE_FORMAT_FP8_E4M3_B128_BF16_INV
        );
    }

    #[test]
    fn open_flm_builds_modelopt_nvfp4_same_name_packed_direct_view() {
        let data = build_test_flm_with_modelopt_nvfp4_stage3_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with ModelOpt NVFP4 Stage 3 bindings");

        let nv_name = "model.language_model.layers.0.linear_attn.out_proj.weight";
        let nv = store.meta(nv_name).expect("NVFP4 logical tensor");
        assert_eq!(nv.layout, LayoutTag::Raw);
        assert_eq!(nv.dtype, "u8");
        assert_eq!(nv.shape, vec![128, 128]);
        assert_eq!(nv.byte_len, 128 * 64);
        assert_eq!(
            store.upload_view(nv_name).expect("NVFP4 upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![128, 64],
            }
        );
        let input = store
            .meta("model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_input_scale")
            .expect("NVFP4 input scale alias");
        assert_eq!(input.dtype, "f32");
        assert_eq!(input.shape, vec![1]);
        assert_eq!(
            store
                .upload_view(
                    "model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_input_scale"
                )
                .expect("NVFP4 input scale upload view"),
            &TensorUploadView {
                dtype: "f32".to_string(),
                shape: vec![1],
            }
        );
    }

    #[test]
    fn open_flm_builds_fp8_scalar_stage3_direct_view() {
        let data = build_test_flm_with_modelopt_nvfp4_stage3_bindings();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with FP8 scalar Stage 3 bindings");

        let fp8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let fp8 = store.meta(fp8_name).expect("FP8 scalar logical tensor");
        assert_eq!(fp8.layout, LayoutTag::Raw);
        assert_eq!(fp8.dtype, "f8_e4m3");
        assert_eq!(fp8.shape, vec![32, 128]);
        assert_eq!(
            store.upload_view(fp8_name).expect("FP8 scalar upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![32, 128],
            }
        );
        let scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_scale")
            .expect("FP8 scalar scale alias");
        assert_eq!(scale.dtype, "f32");
        assert_eq!(scale.shape, vec![1]);
        let input = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_input_scale")
            .expect("FP8 scalar input scale alias");
        assert_eq!(input.dtype, "f32");
        assert_eq!(input.shape, vec![1]);
    }

    #[test]
    fn gpu_upload_shape_uses_packed_row_bytes_for_logical_int4_aliases() {
        let meta = TensorMeta {
            name: "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            shape: vec![17408, 5120],
            dtype: "u8".to_string(),
            layout: LayoutTag::Int4Quantized,
            offset: 0,
            byte_len: 44_564_480,
        };

        let upload_shape = gpu_upload_shape(&meta).expect("upload shape");

        assert_eq!(meta.shape, vec![17408, 5120]);
        assert_eq!(upload_shape, vec![17408, 2560]);
    }

    #[test]
    fn open_flm_rejects_required_manifest_shape_mismatch() {
        let mut data = build_test_flm_with_runtime_manifest_group();
        corrupt_first_manifest_shape(&mut data, 999);
        let file = write_temp_flm(&data);
        let err = match BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        ) {
            Ok(_) => panic!("manifest mismatch should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("manifest"));
    }

    #[test]
    fn open_flm_does_not_alias_suffix_tensors_absent_from_manifest_group() {
        let mut data = build_test_flm(&[
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_packed",
                shape: vec![128, 16],
                dtype: FLM_DTYPE_INT32,
                codec: 1,
                payload: vec![0; 128],
            },
            TestFlmTensor {
                name: "model.language_model.layers.0.mlp.gate_proj.weight_scale",
                shape: vec![128, 1],
                dtype: FLM_DTYPE_BF16,
                codec: 0,
                payload: vec![0; 256],
            },
        ]);
        let runtime = build_test_runtime_directory();
        append_runtime_directory(&mut data, &runtime);
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open FLM with empty manifest");

        assert!(store
            .meta("model.language_model.layers.0.mlp.gate_proj.weight")
            .is_none());
    }

    #[test]
    fn open_flm_can_synthesize_manifest_int4_aliases() {
        let data = build_test_flm_with_runtime_manifest_group();
        let file = write_temp_flm(&data);

        let store = BakedStore::open_flm_with_options(
            file.path(),
            crate::store::FlmLoadOptions {
                flm_int4_logical_aliases: true,
                ..Default::default()
            },
        )
        .expect("open FLM with manifest aliases");

        let weight = store
            .meta("model.language_model.layers.0.mlp.gate_proj.weight")
            .expect("aliased weight");
        assert_eq!(weight.dtype, "u8");
        assert_eq!(weight.shape, vec![128, 64]);
        assert_eq!(weight.layout, LayoutTag::Int4Quantized);
        assert_eq!(
            store
                .raw_bytes("model.language_model.layers.0.mlp.gate_proj.weight")
                .unwrap(),
            &[0; 128]
        );
        assert_eq!(
            store
                .raw_bytes("model.language_model.layers.0.mlp.gate_proj.weight_int4_scale")
                .unwrap(),
            &[0; 256]
        );
        assert_eq!(
            store
                .raw_bytes("model.language_model.layers.0.mlp.gate_proj.weight_int4_zero")
                .unwrap(),
            vec![0x00, 0x41].repeat(128).as_slice()
        );
        assert!(store
            .tensor_storage_range("model.language_model.layers.0.mlp.gate_proj.weight", 0, 64)
            .expect("packed alias storage range")
            .is_some());
        assert!(store
            .tensor_storage_range(
                "model.language_model.layers.0.mlp.gate_proj.weight_int4_scale",
                0,
                64,
            )
            .expect("scale alias storage range")
            .is_some());
        assert!(
            store
                .tensor_storage_range(
                    "model.language_model.layers.0.mlp.gate_proj.weight_int4_zero",
                    0,
                    64,
                )
                .expect("synthetic zero storage range")
                .is_none(),
            "synthesized zero aliases must not expose direct file-backed transfer ranges"
        );
    }

    /// End-to-end loadability test against a real Qwen3.6-MoE bake.
    /// Skipped when `SUPERSONIC_QWEN36_MOE_BAKE_DIR` is unset so CI / non-bake
    /// machines stay green. Exercises mmap, manifest parse, and per-tensor
    /// shape/layout/byte-range invariants the runtime relies on.
    #[test]
    fn qwen36_moe_bake_loadable() {
        let Ok(bake_dir_str) = std::env::var("SUPERSONIC_QWEN36_MOE_BAKE_DIR") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_MOE_BAKE_DIR not set. Point it at a bake \
                 directory like .supersonic/v2-int4-gptq to validate end-to-end \
                 loadability of a real Qwen3.6-MoE INT4 GPTQ bake."
            );
            return;
        };
        let bake_dir = Path::new(&bake_dir_str);
        let store = BakedStore::open(bake_dir).expect("open bake");

        // 1. Vocab/output sanity. lm_head is INT4-packed so its column count
        //    is hidden/2 (2 nibbles per byte). Companion scale + zero must
        //    both be present in the index.
        let lm = store
            .meta("lm_head.weight")
            .expect("lm_head.weight missing");
        assert_eq!(
            lm.layout,
            LayoutTag::Int4Quantized,
            "lm_head should be INT4 in this bake"
        );
        assert_eq!(lm.shape.len(), 2, "lm_head shape should be 2D");
        assert_eq!(
            lm.shape[1],
            2048 / 2,
            "lm_head INT4 column count = hidden/2 (2 nibbles per byte)"
        );
        assert!(
            store.contains("lm_head.weight_int4_scale"),
            "lm_head.weight_int4_scale missing"
        );
        assert!(
            store.contains("lm_head.weight_int4_zero"),
            "lm_head.weight_int4_zero missing"
        );

        // 2. Per-layer MoE expert presence. The bake must have all 40 layers
        //    of fused expert weight, each with packed nibbles + scale + zero.
        for li in 0..40 {
            let lp = format!("model.language_model.layers.{li}.mlp.experts");
            for kind in ["gate_up_proj", "down_proj"] {
                let base = format!("{lp}.{kind}");
                let scale = format!("{base}_int4_scale");
                let zero = format!("{base}_int4_zero");
                assert!(store.contains(&base), "missing fused expert tensor: {base}");
                assert!(store.contains(&scale), "missing scale sidecar: {scale}");
                assert!(store.contains(&zero), "missing zero sidecar: {zero}");
                let m = store.meta(&base).unwrap();
                assert_eq!(
                    m.layout,
                    LayoutTag::Int4Quantized,
                    "{base} should be Int4Quantized"
                );
                assert_eq!(m.shape.len(), 3, "{base} should be 3D [E, rows, cols/2]");
                assert_eq!(m.shape[0], 256, "{base} num_experts must be 256");
            }
        }

        // 3. Norm + gate raw tensors per layer.
        for li in 0..40 {
            let lp = format!("model.language_model.layers.{li}");
            for n in [
                format!("{lp}.input_layernorm.weight"),
                format!("{lp}.post_attention_layernorm.weight"),
                format!("{lp}.mlp.gate.weight"),
                format!("{lp}.mlp.shared_expert_gate.weight"),
            ] {
                let m = store
                    .meta(&n)
                    .unwrap_or_else(|| panic!("missing raw tensor: {n}"));
                assert_eq!(m.layout, LayoutTag::Raw, "{n} should be Raw layout");
                assert_eq!(m.dtype, "bf16", "{n} should be bf16");
            }
        }

        // 4. Each tensor's [offset, offset+byte_len) must lie strictly
        //    within weights.bin and never exceed it. Catches an
        //    integer-overflow / off-by-one in the writer.
        // 4a. Pull the file size via a normal stat to avoid relying on the
        //     internal mmap accessor.
        let weights_path = crate::weights_bin_path(bake_dir);
        let weights_len = std::fs::metadata(&weights_path)
            .expect("stat weights.bin")
            .len();
        for (name, _) in store.index.iter().take(20) {
            let m = store.meta(name).unwrap();
            let end = m.offset + m.byte_len;
            assert!(
                end <= weights_len,
                "{name}: offset+len {end} > weights.bin len {weights_len}"
            );
            // raw_bytes() should succeed and have the right length.
            let bytes = store
                .raw_bytes(name)
                .unwrap_or_else(|| panic!("raw_bytes returned None for {name}"));
            assert_eq!(
                bytes.len() as u64,
                m.byte_len,
                "{name}: raw_bytes length disagrees with manifest"
            );
        }

        // 5. Quick bake-quality smoke: lm_head's INT4 scale must not be all
        //    zero. A run that died mid-quant could leave us with a stub
        //    scale tensor that would silently produce zero logits.
        let scale_bytes = store
            .raw_bytes("lm_head.weight_int4_scale")
            .expect("lm_head scale bytes");
        let nonzero = scale_bytes.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > scale_bytes.len() / 4,
            "lm_head scale looks suspicious: {nonzero}/{} bytes nonzero",
            scale_bytes.len()
        );

        eprintln!(
            "[bake-validate] OK — {} tensors, weights.bin {} MiB",
            store.index.len(),
            weights_len / (1024 * 1024),
        );
    }

    #[test]
    fn flm_qwen36_27b_loadable_with_ct_aliases() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_FLM not set. Point it at an FLM \
                 file like qwen36-27b-int4.flm to validate full-artifact FLM loadability."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-27b FLM");

        let embed = store
            .meta("model.language_model.embed_tokens.weight")
            .expect("embed_tokens missing");
        assert_eq!(embed.dtype, "bf16");
        assert_eq!(embed.shape, vec![248320, 5120]);
        assert_eq!(embed.byte_len, 2_542_796_800);

        let packed_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed";
        let alias_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let packed = store.meta(packed_name).expect("packed CT tensor missing");
        let alias = store.meta(alias_name).expect("native INT4 alias missing");
        assert_eq!(packed.dtype, "u8");
        assert_eq!(alias.dtype, "u8");
        assert_eq!(alias.layout, LayoutTag::Int4Quantized);
        assert_eq!(alias.shape, vec![10240, 5120]);
        assert_eq!(alias.offset, packed.offset);
        assert_eq!(alias.byte_len, packed.byte_len);

        let scale = store
            .raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_int4_scale")
            .expect("aliased scale bytes");
        assert_eq!(scale.len(), 819_200);
        let scale_meta = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_int4_scale")
            .expect("aliased scale meta");
        assert_eq!(scale_meta.dtype, "bf16");
        let zero = store
            .raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_int4_zero")
            .expect("synthetic zero bytes");
        assert_eq!(zero.len(), scale.len());
        assert!(zero.chunks_exact(2).all(|pair| pair == [0x00, 0x41]));

        eprintln!(
            "[flm-validate] OK — {} tensors including FLM logical aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_27b_fp8_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_FP8_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_FP8_FLM not set. Point it at a \
                 preserve-lowbit Qwen3.6-27B-FP8 FLM to validate Stage 3 FP8 \
                 block-scale aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-27b FP8 FLM");

        let embed = store
            .meta("model.language_model.embed_tokens.weight")
            .expect("embed_tokens missing");
        assert_eq!(embed.dtype, "bf16");
        assert_eq!(embed.shape, vec![248320, 5120]);
        assert_eq!(embed.byte_len, 2_542_796_800);

        let fp8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let fp8 = store
            .meta(fp8_name)
            .expect("Qwen FP8 logical alias missing");
        assert_eq!(fp8.layout, LayoutTag::Raw);
        assert_eq!(fp8.dtype, "f8_e4m3");
        assert_eq!(fp8.shape, vec![10240, 5120]);
        assert_eq!(fp8.byte_len, 52_428_800);
        assert_eq!(
            store
                .upload_view(fp8_name)
                .expect("Qwen FP8 direct upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![10240, 5120],
            }
        );

        let scale_alias = concat!(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.",
            "weight_fp8_e4m3_b128_bf16_scale_inv"
        );
        let scale = store
            .meta(scale_alias)
            .expect("Qwen FP8 inverse-scale alias missing");
        assert_eq!(scale.dtype, "bf16");
        assert_eq!(scale.shape, vec![80, 40]);
        assert_eq!(scale.byte_len, 6_400);

        let runtime = store.flm_runtime().expect("runtime directory");
        let fp8_logical_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| {
                logical.value_format_id == crate::flm::VALUE_FORMAT_FP8_E4M3_B128_BF16_INV
            })
            .count();
        assert_eq!(fp8_logical_count, 407);

        eprintln!(
            "[flm-validate] OK — {} tensors including Qwen FP8 block-scale aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_27b_nvfp4_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_NVFP4_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_NVFP4_FLM not set. Point it at a \
                 preserve-lowbit nvidia/Qwen3.6-27B-NVFP4 FLM to validate dense \
                 NVFP4/FP8 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-27b NVFP4 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let dense = runtime.qwen36_config().expect("dense runtime config");
        assert_eq!(dense.hidden_size, 5120);
        assert_eq!(dense.num_hidden_layers, 64);
        assert_eq!(dense.intermediate_size, 17408);

        let raw_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_RAW_DENSE)
            .count();
        let nvfp4_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_NVFP4_E2M1)
            .count();
        let fp8_scalar_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_FP8_E4M3_F32)
            .count();
        assert_eq!(raw_count, 450);
        assert_eq!(nvfp4_count, 193);
        assert_eq!(fp8_scalar_count, 208);

        let embed = store
            .meta("model.language_model.embed_tokens.weight")
            .expect("embed_tokens missing");
        assert_eq!(embed.dtype, "bf16");
        assert_eq!(embed.shape, vec![248320, 5120]);
        assert_eq!(embed.byte_len, 2_542_796_800);

        let lm_head = store.meta("lm_head.weight").expect("lm_head missing");
        assert_eq!(lm_head.dtype, "u8");
        assert_eq!(lm_head.shape, vec![248320, 5120]);
        assert_eq!(lm_head.byte_len, 635_699_200);
        assert_eq!(
            store
                .upload_view("lm_head.weight")
                .expect("NVFP4 lm_head upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![248320, 2560],
            }
        );
        assert!(store.contains("lm_head.weight_nvfp4_scale"));
        assert!(store.contains("lm_head.weight_nvfp4_global_scale"));
        assert!(store.contains("lm_head.weight_nvfp4_input_scale"));

        let mlp_name = "model.language_model.layers.0.mlp.down_proj.weight";
        let mlp = store.meta(mlp_name).expect("NVFP4 MLP logical alias");
        assert_eq!(mlp.dtype, "u8");
        assert_eq!(mlp.shape, vec![5120, 17408]);
        assert_eq!(mlp.byte_len, 44_564_480);
        assert_eq!(
            store.upload_view(mlp_name).expect("NVFP4 MLP upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![5120, 8704],
            }
        );
        assert!(store.contains("model.language_model.layers.0.mlp.down_proj.weight_nvfp4_scale"));
        assert!(
            store.contains("model.language_model.layers.0.mlp.down_proj.weight_nvfp4_global_scale")
        );
        assert!(
            store.contains("model.language_model.layers.0.mlp.down_proj.weight_nvfp4_input_scale")
        );

        let fp8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        assert_eq!(
            store.upload_view(fp8_name).expect("FP8 scalar upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![10240, 5120],
            }
        );
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_scale"
        ));
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_input_scale"
        ));

        eprintln!(
            "[flm-validate] OK — {} tensors including 27B NVFP4/FP8 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_27b_mxfp4_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_MXFP4_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_MXFP4_FLM not set. Point it at a \
                 preserve-lowbit OsaurusAI/Qwen3.6-27B-MXFP4 FLM to validate \
                 dense MXFP4 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-27b MXFP4 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let dense = runtime.qwen36_config().expect("dense runtime config");
        assert_eq!(dense.hidden_size, 5120);
        assert_eq!(dense.num_hidden_layers, 64);
        assert_eq!(dense.intermediate_size, 17408);

        let raw_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_RAW_DENSE)
            .count();
        let mxfp4_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_MXFP4_E2M1)
            .count();
        assert_eq!(raw_count, 353);
        assert_eq!(mxfp4_count, 498);

        let embed_name = "model.language_model.embed_tokens.weight";
        let embed = store.meta(embed_name).expect("MXFP4 embed logical alias");
        assert_eq!(embed.dtype, "u8");
        assert_eq!(embed.shape, vec![248320, 5120]);
        assert_eq!(embed.byte_len, 635_699_200);
        assert_eq!(
            store
                .upload_view(embed_name)
                .expect("MXFP4 embed upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![248320, 2560],
            }
        );
        let embed_scale = store
            .meta("model.language_model.embed_tokens.weight_mxfp4_scale")
            .expect("MXFP4 embed scale alias");
        assert_eq!(embed_scale.dtype, "u8");
        assert_eq!(embed_scale.shape, vec![248320, 160]);

        let lm_head = store.meta("lm_head.weight").expect("lm_head missing");
        assert_eq!(lm_head.dtype, "u8");
        assert_eq!(lm_head.shape, vec![248320, 5120]);
        assert_eq!(lm_head.byte_len, 635_699_200);
        assert_eq!(
            store
                .upload_view("lm_head.weight")
                .expect("MXFP4 lm_head upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![248320, 2560],
            }
        );
        let lm_head_scale = store
            .meta("lm_head.weight_mxfp4_scale")
            .expect("MXFP4 lm_head scale alias");
        assert_eq!(lm_head_scale.dtype, "u8");
        assert_eq!(lm_head_scale.shape, vec![248320, 160]);

        let qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let qkv = store.meta(qkv_name).expect("MXFP4 qkv logical alias");
        assert_eq!(qkv.dtype, "u8");
        assert_eq!(qkv.shape, vec![10240, 5120]);
        assert_eq!(qkv.byte_len, 26_214_400);
        assert_eq!(
            store.upload_view(qkv_name).expect("MXFP4 qkv upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![10240, 2560],
            }
        );
        let qkv_scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_mxfp4_scale")
            .expect("MXFP4 qkv scale alias");
        assert_eq!(qkv_scale.dtype, "u8");
        assert_eq!(qkv_scale.shape, vec![10240, 160]);

        let mlp_name = "model.language_model.layers.0.mlp.down_proj.weight";
        let mlp = store.meta(mlp_name).expect("MXFP4 MLP logical alias");
        assert_eq!(mlp.dtype, "u8");
        assert_eq!(mlp.shape, vec![5120, 17408]);
        assert_eq!(mlp.byte_len, 44_564_480);
        assert_eq!(
            store.upload_view(mlp_name).expect("MXFP4 MLP upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![5120, 8704],
            }
        );
        let mlp_scale = store
            .meta("model.language_model.layers.0.mlp.down_proj.weight_mxfp4_scale")
            .expect("MXFP4 MLP scale alias");
        assert_eq!(mlp_scale.dtype, "u8");
        assert_eq!(mlp_scale.shape, vec![5120, 544]);

        let conv = store
            .meta("model.language_model.layers.0.linear_attn.conv1d.weight")
            .expect("conv1d raw tensor");
        assert_eq!(conv.dtype, "bf16");
        assert_eq!(conv.shape, vec![10240, 4, 1]);

        eprintln!(
            "[flm-validate] OK — {} tensors including 27B MXFP4 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_27b_mxfp8_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_MXFP8_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_MXFP8_FLM not set. Point it at a \
                 preserve-lowbit mlx-community/Qwen3.6-27B-mxfp8 FLM to validate \
                 dense MXFP8 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-27b MXFP8 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let dense = runtime.qwen36_config().expect("dense runtime config");
        assert_eq!(dense.hidden_size, 5120);
        assert_eq!(dense.num_hidden_layers, 64);
        assert_eq!(dense.intermediate_size, 17408);

        let raw_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_RAW_DENSE)
            .count();
        let mxfp8_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_MXFP8_E4M3)
            .count();
        assert_eq!(raw_count, 353);
        assert_eq!(mxfp8_count, 498);

        let embed_name = "model.language_model.embed_tokens.weight";
        let embed = store.meta(embed_name).expect("MXFP8 embed logical alias");
        assert_eq!(embed.dtype, "f8_e4m3");
        assert_eq!(embed.shape, vec![248320, 5120]);
        assert_eq!(embed.byte_len, 1_271_398_400);
        assert_eq!(
            store
                .upload_view(embed_name)
                .expect("MXFP8 embed upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![248320, 5120],
            }
        );
        let embed_scale = store
            .meta("model.language_model.embed_tokens.weight_mxfp8_scale")
            .expect("MXFP8 embed scale alias");
        assert_eq!(embed_scale.dtype, "u8");
        assert_eq!(embed_scale.shape, vec![248320, 160]);

        let lm_head = store.meta("lm_head.weight").expect("lm_head missing");
        assert_eq!(lm_head.dtype, "f8_e4m3");
        assert_eq!(lm_head.shape, vec![248320, 5120]);
        assert_eq!(lm_head.byte_len, 1_271_398_400);
        assert_eq!(
            store
                .upload_view("lm_head.weight")
                .expect("MXFP8 lm_head upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![248320, 5120],
            }
        );
        let lm_head_scale = store
            .meta("lm_head.weight_mxfp8_scale")
            .expect("MXFP8 lm_head scale alias");
        assert_eq!(lm_head_scale.dtype, "u8");
        assert_eq!(lm_head_scale.shape, vec![248320, 160]);

        let qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let qkv = store.meta(qkv_name).expect("MXFP8 qkv logical alias");
        assert_eq!(qkv.dtype, "f8_e4m3");
        assert_eq!(qkv.shape, vec![10240, 5120]);
        assert_eq!(qkv.byte_len, 52_428_800);
        assert_eq!(
            store.upload_view(qkv_name).expect("MXFP8 qkv upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![10240, 5120],
            }
        );
        let qkv_scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_mxfp8_scale")
            .expect("MXFP8 qkv scale alias");
        assert_eq!(qkv_scale.dtype, "u8");
        assert_eq!(qkv_scale.shape, vec![10240, 160]);

        let mlp_name = "model.language_model.layers.0.mlp.down_proj.weight";
        let mlp = store.meta(mlp_name).expect("MXFP8 MLP logical alias");
        assert_eq!(mlp.dtype, "f8_e4m3");
        assert_eq!(mlp.shape, vec![5120, 17408]);
        assert_eq!(mlp.byte_len, 89_128_960);
        assert_eq!(
            store.upload_view(mlp_name).expect("MXFP8 MLP upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![5120, 17408],
            }
        );
        let mlp_scale = store
            .meta("model.language_model.layers.0.mlp.down_proj.weight_mxfp8_scale")
            .expect("MXFP8 MLP scale alias");
        assert_eq!(mlp_scale.dtype, "u8");
        assert_eq!(mlp_scale.shape, vec![5120, 544]);

        let conv = store
            .meta("model.language_model.layers.0.linear_attn.conv1d.weight")
            .expect("conv1d raw tensor");
        assert_eq!(conv.dtype, "bf16");
        assert_eq!(conv.shape, vec![10240, 4, 1]);

        eprintln!(
            "[flm-validate] OK — {} tensors including 27B MXFP8 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_tiny_qwen36_raw_fp32_and_int4_aliases_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_TINY_QWEN36_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_TINY_QWEN36_FLM not set. Point it at a tiny \
                 converter-produced Qwen3.6 FLM to validate Stage 3 raw VALUE and \
                 INT4 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open tiny qwen3.6 FLM");

        let raw_name = "model.language_model.layers.0.linear_attn.A_log";
        let raw = store.meta(raw_name).expect("raw FP32 logical tensor");
        assert_eq!(raw.layout, LayoutTag::Raw);
        assert_eq!(raw.dtype, "f32");
        assert_eq!(raw.shape, vec![4]);
        assert_eq!(raw.byte_len, 16);
        let raw_view = store
            .upload_view(raw_name)
            .expect("raw FP32 direct upload view");
        assert_eq!(raw_view.dtype, "f32");
        assert_eq!(raw_view.shape, vec![4]);

        let packed_name = "model.language_model.layers.0.linear_attn.out_proj.weight_packed";
        let alias_name = "model.language_model.layers.0.linear_attn.out_proj.weight";
        let packed = store.meta(packed_name).expect("packed INT4 storage tensor");
        let alias = store.meta(alias_name).expect("logical INT4 alias");
        assert_eq!(alias.layout, LayoutTag::Int4Quantized);
        assert_eq!(alias.dtype, "u8");
        assert_eq!(alias.shape, vec![8, 128]);
        assert_eq!(alias.offset, packed.offset);
        assert_eq!(alias.byte_len, packed.byte_len);
        let alias_view = store
            .upload_view(alias_name)
            .expect("logical INT4 direct upload view");
        assert_eq!(alias_view.dtype, "u8");
        assert_eq!(alias_view.shape, vec![8, 64]);

        let runtime = store.flm_runtime().expect("runtime directory");
        let raw_logical = runtime
            .logical_tensors()
            .iter()
            .find(|logical| logical.name == raw_name)
            .expect("raw logical runtime row");
        assert_eq!(
            raw_logical.value_format_id,
            crate::flm::VALUE_FORMAT_RAW_DENSE
        );
    }

    #[test]
    fn flm_tiny_qwen36_lowbit_aliases_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_TINY_QWEN36_LOWBIT_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_TINY_QWEN36_LOWBIT_FLM not set. Point it at a tiny \
                 descriptor-produced Qwen3.6 FLM to validate NVFP4/MXFP Stage 3 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open tiny low-bit qwen3.6 FLM");

        let nv_name = "model.language_model.layers.0.linear_attn.out_proj.weight";
        let nv = store.meta(nv_name).expect("NVFP4 logical tensor");
        assert_eq!(nv.dtype, "u8");
        assert_eq!(nv.shape, vec![128, 128]);
        assert_eq!(
            store.upload_view(nv_name).expect("NVFP4 upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![128, 64],
            }
        );
        assert!(
            store.contains("model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_scale")
        );
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_global_scale"
        ));

        let mx4_name = "model.language_model.layers.0.linear_attn.in_proj_z.weight";
        assert_eq!(
            store.upload_view(mx4_name).expect("MXFP4 upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![64, 64],
            }
        );
        assert!(store
            .contains("model.language_model.layers.0.linear_attn.in_proj_z.weight_mxfp4_scale"));

        let mx8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        assert_eq!(
            store.upload_view(mx8_name).expect("MXFP8 upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![32, 128],
            }
        );
        assert!(store
            .contains("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_mxfp8_scale"));

        let qwen_fp8_name = "model.language_model.layers.0.self_attn.q_proj.weight";
        assert_eq!(
            store
                .upload_view(qwen_fp8_name)
                .expect("Qwen FP8 block-scale upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![8, 128],
            }
        );
        assert!(store.contains(
            "model.language_model.layers.0.self_attn.q_proj.weight_fp8_e4m3_b128_bf16_scale_inv"
        ));
    }

    #[test]
    fn flm_tiny_qwen36_modelopt_lowbit_aliases_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_TINY_QWEN36_MODELOPT_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_TINY_QWEN36_MODELOPT_FLM not set. Point it at a tiny \
                 descriptor-produced Qwen3.6 FLM to validate ModelOpt NVFP4/FP8 Stage 3 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open tiny ModelOpt low-bit qwen3.6 FLM");

        let nv_name = "model.language_model.layers.0.linear_attn.out_proj.weight";
        let nv = store.meta(nv_name).expect("ModelOpt NVFP4 logical tensor");
        assert_eq!(nv.dtype, "u8");
        assert_eq!(nv.shape, vec![128, 128]);
        assert_eq!(
            store
                .upload_view(nv_name)
                .expect("ModelOpt NVFP4 upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![128, 64],
            }
        );
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.out_proj.weight_nvfp4_input_scale"
        ));

        let fp8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        assert_eq!(
            store.upload_view(fp8_name).expect("FP8 scalar upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![32, 128],
            }
        );
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_scale"
        ));
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_input_scale"
        ));
    }

    #[test]
    fn flm_tiny_qwen36_moe_modelopt_lowbit_aliases_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_TINY_QWEN36_MOE_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_TINY_QWEN36_MOE_FLM not set. Point it at a tiny \
                 descriptor-produced Qwen3.6 MoE FLM to validate runtime parsing and \
                 ModelOpt expert aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open tiny ModelOpt low-bit qwen3.6 MoE FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        assert!(runtime.qwen36_config().is_none());
        let moe = runtime.qwen36_moe_config().expect("MoE runtime config");
        assert_eq!(moe.hidden_size, 4);
        assert_eq!(moe.num_experts, 2);
        assert_eq!(moe.num_experts_per_tok, 1);
        assert_eq!(moe.mrope_section, [11, 11, 10]);

        let expert_name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
        let expert = store
            .meta(expert_name)
            .expect("ModelOpt expert logical tensor");
        assert_eq!(expert.dtype, "u8");
        assert_eq!(expert.shape, vec![2, 2]);
        assert_eq!(
            store
                .upload_view(expert_name)
                .expect("ModelOpt expert upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![2, 1],
            }
        );
        assert!(store.contains(
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_nvfp4_input_scale"
        ));
    }

    #[test]
    fn flm_qwen36_35b_native_int4_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_NATIVE_INT4_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_NATIVE_INT4_FLM not set. Point it at a \
                 SuperSonic-native INT4 Qwen3.6-35B-A3B FLM to validate full MoE \
                 native aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-35b native INT4 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        assert!(runtime.qwen36_config().is_none());
        let moe = runtime.qwen36_moe_config().expect("MoE runtime config");
        assert_eq!(moe.hidden_size, 2048);
        assert_eq!(moe.num_hidden_layers, 40);
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.num_experts_per_tok, 8);

        let expert_name = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let expert = store
            .meta(expert_name)
            .expect("native INT4 fused expert logical tensor");
        assert_eq!(expert.layout, LayoutTag::Int4Quantized);
        assert_eq!(expert.dtype, "u8");
        assert_eq!(expert.shape, vec![256, 1024, 1024]);
        assert_eq!(expert.byte_len, 268_435_456);
        assert_eq!(
            store
                .upload_view(expert_name)
                .expect("native INT4 fused expert upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![256, 1024, 1024],
            }
        );

        let expert_scale = store
            .meta("model.language_model.layers.0.mlp.experts.gate_up_proj_int4_scale")
            .expect("native INT4 fused expert scale");
        assert_eq!(expert_scale.dtype, "bf16");
        assert_eq!(expert_scale.shape, vec![256, 8, 16]);
        let expert_zero = store
            .meta("model.language_model.layers.0.mlp.experts.gate_up_proj_int4_zero")
            .expect("native INT4 fused expert zero plane");
        assert_eq!(expert_zero.dtype, "bf16");
        assert_eq!(expert_zero.shape, vec![256, 8, 16]);

        let qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let qkv = store.meta(qkv_name).expect("native INT4 linear qkv tensor");
        assert_eq!(qkv.layout, LayoutTag::Int4Quantized);
        assert_eq!(qkv.dtype, "u8");
        assert_eq!(qkv.shape, vec![8192, 1024]);
        assert_eq!(qkv.byte_len, 8_388_608);

        let embed = store
            .meta("model.language_model.embed_tokens.weight")
            .expect("embed_tokens missing");
        assert_eq!(embed.dtype, "bf16");
        assert_eq!(embed.shape, vec![248320, 2048]);
        assert_eq!(embed.byte_len, 1_017_118_720);

        eprintln!(
            "[flm-validate] OK — {} tensors including 35B native INT4 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_35b_nvfp4_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_NVFP4_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_NVFP4_FLM not set. Point it at a \
                 preserve-lowbit nvidia/Qwen3.6-35B-A3B-NVFP4 FLM to validate \
                 full MoE ModelOpt aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-35b NVFP4 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let moe = runtime.qwen36_moe_config().expect("MoE runtime config");
        assert_eq!(moe.hidden_size, 2048);
        assert_eq!(moe.num_hidden_layers, 40);
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.num_experts_per_tok, 8);

        let nvfp4_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_NVFP4_E2M1)
            .count();
        let fp8_scalar_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_FP8_E4M3_F32)
            .count();
        assert_eq!(nvfp4_count, 30_841);
        assert_eq!(fp8_scalar_count, 130);

        let embed = store
            .meta("model.language_model.embed_tokens.weight")
            .expect("embed_tokens missing");
        assert_eq!(embed.dtype, "bf16");
        assert_eq!(embed.shape, vec![248320, 2048]);
        assert_eq!(embed.byte_len, 1_017_118_720);

        let lm_head = store.meta("lm_head.weight").expect("lm_head missing");
        assert_eq!(lm_head.dtype, "u8");
        assert_eq!(lm_head.shape, vec![248320, 2048]);
        assert_eq!(lm_head.byte_len, 254_279_680);
        assert_eq!(
            store
                .upload_view("lm_head.weight")
                .expect("NVFP4 lm_head upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![248320, 1024],
            }
        );
        assert!(store.contains("lm_head.weight_nvfp4_scale"));
        assert!(store.contains("lm_head.weight_nvfp4_global_scale"));
        assert!(store.contains("lm_head.weight_nvfp4_input_scale"));

        let fp8_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        assert_eq!(
            store.upload_view(fp8_name).expect("FP8 scalar upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![8192, 2048],
            }
        );
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_scale"
        ));
        assert!(store.contains(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_fp8_e4m3_f32_input_scale"
        ));

        eprintln!(
            "[flm-validate] OK — {} tensors including 35B NVFP4/FP8 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_35b_mxfp4_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_MXFP4_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_MXFP4_FLM not set. Point it at a \
                 preserve-lowbit Qwen3.6-35B-A3B-MXFP4 FLM to validate full MoE \
                 MXFP4 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-35b MXFP4 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let moe = runtime.qwen36_moe_config().expect("MoE runtime config");
        assert_eq!(moe.hidden_size, 2048);
        assert_eq!(moe.num_hidden_layers, 40);
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.num_experts_per_tok, 8);

        let raw_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_RAW_DENSE)
            .count();
        let mxfp4_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_MXFP4_E2M1)
            .count();
        assert_eq!(raw_count, 463);
        assert_eq!(mxfp4_count, 30_870);

        let embed = store
            .meta("model.language_model.embed_tokens.weight")
            .expect("embed_tokens missing");
        assert_eq!(embed.dtype, "bf16");
        assert_eq!(embed.shape, vec![248320, 2048]);
        assert_eq!(embed.byte_len, 1_017_118_720);

        let lm_head = store.meta("lm_head.weight").expect("lm_head missing");
        assert_eq!(lm_head.dtype, "bf16");
        assert_eq!(lm_head.shape, vec![248320, 2048]);
        assert_eq!(lm_head.byte_len, 1_017_118_720);

        let qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let qkv = store.meta(qkv_name).expect("MXFP4 qkv logical alias");
        assert_eq!(qkv.dtype, "u8");
        assert_eq!(qkv.shape, vec![8192, 2048]);
        assert_eq!(qkv.byte_len, 8_388_608);
        assert_eq!(
            store.upload_view(qkv_name).expect("MXFP4 qkv upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![8192, 1024],
            }
        );
        let qkv_scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_mxfp4_scale")
            .expect("MXFP4 qkv scale alias");
        assert_eq!(qkv_scale.dtype, "u8");
        assert_eq!(qkv_scale.shape, vec![8192, 64]);

        let expert_name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
        let expert = store.meta(expert_name).expect("MXFP4 expert logical alias");
        assert_eq!(expert.dtype, "u8");
        assert_eq!(expert.shape, vec![512, 2048]);
        assert_eq!(expert.byte_len, 524_288);
        assert_eq!(
            store
                .upload_view(expert_name)
                .expect("MXFP4 expert upload view"),
            &TensorUploadView {
                dtype: "u8".to_string(),
                shape: vec![512, 1024],
            }
        );
        let expert_scale = store
            .meta("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_mxfp4_scale")
            .expect("MXFP4 expert scale alias");
        assert_eq!(expert_scale.dtype, "u8");
        assert_eq!(expert_scale.shape, vec![512, 64]);

        eprintln!(
            "[flm-validate] OK — {} tensors including 35B MXFP4 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_35b_mxfp8_preserve_lowbit_loadable() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_MXFP8_FLM") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_MXFP8_FLM not set. Point it at a \
                 preserve-lowbit mlx-community/Qwen3.6-35B-A3B-mxfp8 FLM to \
                 validate full MoE MXFP8 aliases."
            );
            return;
        };
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: true,
            },
        )
        .expect("open qwen3.6-35b MXFP8 FLM");

        let runtime = store.flm_runtime().expect("runtime directory");
        let moe = runtime.qwen36_moe_config().expect("MoE runtime config");
        assert_eq!(moe.hidden_size, 2048);
        assert_eq!(moe.num_hidden_layers, 40);
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.num_experts_per_tok, 8);

        let raw_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_RAW_DENSE)
            .count();
        let mxfp8_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_MXFP8_E4M3)
            .count();
        let fp8_b64_count = runtime
            .logical_tensors()
            .iter()
            .filter(|logical| logical.value_format_id == crate::flm::VALUE_FORMAT_FP8_E4M3_B64_BF16)
            .count();
        assert_eq!(raw_count, 221);
        assert_eq!(mxfp8_count, 432);
        assert_eq!(fp8_b64_count, 80);

        let embed_name = "model.language_model.embed_tokens.weight";
        let embed = store.meta(embed_name).expect("MXFP8 embed logical alias");
        assert_eq!(embed.dtype, "f8_e4m3");
        assert_eq!(embed.shape, vec![248320, 2048]);
        assert_eq!(embed.byte_len, 508_559_360);
        assert_eq!(
            store
                .upload_view(embed_name)
                .expect("MXFP8 embed upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![248320, 2048],
            }
        );
        let embed_scale = store
            .meta("model.language_model.embed_tokens.weight_mxfp8_scale")
            .expect("MXFP8 embed scale alias");
        assert_eq!(embed_scale.dtype, "u8");
        assert_eq!(embed_scale.shape, vec![248320, 64]);

        let qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let qkv = store.meta(qkv_name).expect("MXFP8 qkv logical alias");
        assert_eq!(qkv.dtype, "f8_e4m3");
        assert_eq!(qkv.shape, vec![8192, 2048]);
        assert_eq!(qkv.byte_len, 16_777_216);
        let qkv_scale = store
            .meta("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_mxfp8_scale")
            .expect("MXFP8 qkv scale alias");
        assert_eq!(qkv_scale.dtype, "u8");
        assert_eq!(qkv_scale.shape, vec![8192, 64]);

        let switch_name = "model.language_model.layers.0.mlp.switch_mlp.down_proj.weight";
        let switch = store
            .meta(switch_name)
            .expect("rank-3 MXFP8 switch MLP logical alias");
        assert_eq!(switch.dtype, "f8_e4m3");
        assert_eq!(switch.shape, vec![256, 2048, 512]);
        assert_eq!(switch.byte_len, 268_435_456);
        assert_eq!(
            store
                .upload_view(switch_name)
                .expect("rank-3 MXFP8 switch MLP upload view"),
            &TensorUploadView {
                dtype: "f8_e4m3".to_string(),
                shape: vec![256, 2048, 512],
            }
        );
        let switch_scale = store
            .meta("model.language_model.layers.0.mlp.switch_mlp.down_proj.weight_mxfp8_scale")
            .expect("rank-3 MXFP8 switch MLP scale alias");
        assert_eq!(switch_scale.dtype, "u8");
        assert_eq!(switch_scale.shape, vec![256, 2048, 16]);

        let gate_name = "model.language_model.layers.0.mlp.gate.weight";
        let gate = store
            .meta(gate_name)
            .expect("FP8-B64-BF16 gate logical alias");
        assert_eq!(gate.dtype, "f8_e4m3");
        assert_eq!(gate.shape, vec![256, 2048]);
        assert_eq!(gate.byte_len, 524_288);
        let gate_scale = store
            .meta("model.language_model.layers.0.mlp.gate.weight_fp8_e4m3_b64_bf16_scale")
            .expect("FP8-B64-BF16 gate scale alias");
        assert_eq!(gate_scale.dtype, "bf16");
        assert_eq!(gate_scale.shape, vec![256, 32]);

        let shared_gate_scale = store
            .meta(
                "model.language_model.layers.0.mlp.shared_expert_gate.weight_fp8_e4m3_b64_bf16_scale",
            )
            .expect("FP8-B64-BF16 shared expert gate scale alias");
        assert_eq!(shared_gate_scale.dtype, "bf16");
        assert_eq!(shared_gate_scale.shape, vec![1, 32]);

        eprintln!(
            "[flm-validate] OK — {} tensors including 35B MXFP8/FP8-B64 aliases",
            store.index.len()
        );
    }

    #[test]
    fn flm_qwen36_27b_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_FLM_HIP_UPLOAD not set. Point it at an FLM \
                 file like qwen36-27b-int4.flm to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-27b FLM");

        let weight = store
            .load_to_gpu(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                0,
            )
            .expect("upload INT4 logical alias with direct view");
        assert_eq!(weight.dtype(), ScalarType::U8);
        assert_eq!(weight.shape(), &[10240, 2560]);

        let scale = store
            .load_to_gpu(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_int4_scale",
                0,
            )
            .expect("upload INT4 scale alias with direct view");
        assert_eq!(scale.dtype(), ScalarType::BF16);
        assert_eq!(scale.shape(), &[10240, 40]);

        eprintln!("[flm-upload] OK — FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_27b_fp8_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_FP8_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_FP8_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit Qwen3.6-27B-FP8 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-27b FP8 FLM");

        let weight = store
            .load_to_gpu(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                0,
            )
            .expect("upload Qwen FP8 logical alias with direct view");
        assert_eq!(weight.dtype(), ScalarType::F8E4M3);
        assert_eq!(weight.shape(), &[10240, 5120]);

        let scale = store
            .load_to_gpu(
                concat!(
                    "model.language_model.layers.0.linear_attn.in_proj_qkv.",
                    "weight_fp8_e4m3_b128_bf16_scale_inv"
                ),
                0,
            )
            .expect("upload Qwen FP8 scale_inv alias with direct view");
        assert_eq!(scale.dtype(), ScalarType::BF16);
        assert_eq!(scale.shape(), &[80, 40]);

        eprintln!("[flm-upload] OK — Qwen FP8 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_27b_nvfp4_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_NVFP4_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_NVFP4_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit nvidia/Qwen3.6-27B-NVFP4 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-27b NVFP4 FLM");

        let mlp = store
            .load_to_gpu("model.language_model.layers.0.mlp.down_proj.weight", 0)
            .expect("upload NVFP4 MLP logical alias with direct view");
        assert_eq!(mlp.dtype(), ScalarType::U8);
        assert_eq!(mlp.shape(), &[5120, 8704]);

        let fp8 = store
            .load_to_gpu(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                0,
            )
            .expect("upload FP8 scalar logical alias with direct view");
        assert_eq!(fp8.dtype(), ScalarType::F8E4M3);
        assert_eq!(fp8.shape(), &[10240, 5120]);

        eprintln!("[flm-upload] OK — 27B NVFP4 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_27b_mxfp4_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_MXFP4_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_MXFP4_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit OsaurusAI/Qwen3.6-27B-MXFP4 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-27b MXFP4 FLM");

        let mlp = store
            .load_to_gpu("model.language_model.layers.0.mlp.down_proj.weight", 0)
            .expect("upload MXFP4 MLP logical alias with direct view");
        assert_eq!(mlp.dtype(), ScalarType::U8);
        assert_eq!(mlp.shape(), &[5120, 8704]);

        let mlp_scale = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.down_proj.weight_mxfp4_scale",
                0,
            )
            .expect("upload MXFP4 MLP scale alias with direct view");
        assert_eq!(mlp_scale.dtype(), ScalarType::U8);
        assert_eq!(mlp_scale.shape(), &[5120, 544]);

        eprintln!("[flm-upload] OK — 27B MXFP4 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_27b_mxfp8_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_27B_MXFP8_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_27B_MXFP8_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit mlx-community/Qwen3.6-27B-mxfp8 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-27b MXFP8 FLM");

        let mlp = store
            .load_to_gpu("model.language_model.layers.0.mlp.down_proj.weight", 0)
            .expect("upload MXFP8 MLP logical alias with direct view");
        assert_eq!(mlp.dtype(), ScalarType::F8E4M3);
        assert_eq!(mlp.shape(), &[5120, 17408]);

        let mlp_scale = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.down_proj.weight_mxfp8_scale",
                0,
            )
            .expect("upload MXFP8 MLP scale alias with direct view");
        assert_eq!(mlp_scale.dtype(), ScalarType::U8);
        assert_eq!(mlp_scale.shape(), &[5120, 544]);

        eprintln!("[flm-upload] OK — 27B MXFP8 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_35b_nvfp4_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_NVFP4_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_NVFP4_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit nvidia/Qwen3.6-35B-A3B-NVFP4 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-35b NVFP4 FLM");

        let expert = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                0,
            )
            .expect("upload NVFP4 expert logical alias with direct view");
        assert_eq!(expert.dtype(), ScalarType::U8);
        assert_eq!(expert.shape(), &[512, 1024]);

        let fp8 = store
            .load_to_gpu(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                0,
            )
            .expect("upload FP8 scalar logical alias with direct view");
        assert_eq!(fp8.dtype(), ScalarType::F8E4M3);
        assert_eq!(fp8.shape(), &[8192, 2048]);

        eprintln!("[flm-upload] OK — 35B NVFP4 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_35b_mxfp4_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_MXFP4_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_MXFP4_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit Qwen3.6-35B-A3B-MXFP4 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-35b MXFP4 FLM");

        let expert = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                0,
            )
            .expect("upload MXFP4 expert logical alias with direct view");
        assert_eq!(expert.dtype(), ScalarType::U8);
        assert_eq!(expert.shape(), &[512, 1024]);

        let expert_scale = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_mxfp4_scale",
                0,
            )
            .expect("upload MXFP4 expert scale alias with direct view");
        assert_eq!(expert_scale.dtype(), ScalarType::U8);
        assert_eq!(expert_scale.shape(), &[512, 64]);

        eprintln!("[flm-upload] OK — 35B MXFP4 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_35b_mxfp8_direct_views_upload_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_MXFP8_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_MXFP8_FLM_HIP_UPLOAD not set. Point it at a \
                 preserve-lowbit mlx-community/Qwen3.6-35B-A3B-mxfp8 FLM to validate HIP uploads."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-35b MXFP8 FLM");

        let shared = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                0,
            )
            .expect("upload MXFP8 shared expert logical alias with direct view");
        assert_eq!(shared.dtype(), ScalarType::F8E4M3);
        assert_eq!(shared.shape(), &[2048, 512]);

        let shared_scale = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight_mxfp8_scale",
                0,
            )
            .expect("upload MXFP8 shared expert scale alias with direct view");
        assert_eq!(shared_scale.dtype(), ScalarType::U8);
        assert_eq!(shared_scale.shape(), &[2048, 16]);

        let gate = store
            .load_to_gpu("model.language_model.layers.0.mlp.gate.weight", 0)
            .expect("upload FP8-B64-BF16 gate logical alias with direct view");
        assert_eq!(gate.dtype(), ScalarType::F8E4M3);
        assert_eq!(gate.shape(), &[256, 2048]);

        let gate_scale = store
            .load_to_gpu(
                "model.language_model.layers.0.mlp.gate.weight_fp8_e4m3_b64_bf16_scale",
                0,
            )
            .expect("upload FP8-B64-BF16 gate scale alias with direct view");
        assert_eq!(gate_scale.dtype(), ScalarType::BF16);
        assert_eq!(gate_scale.shape(), &[256, 32]);

        eprintln!("[flm-upload] OK — 35B MXFP8 FLM direct views uploaded to HIP");
    }

    #[test]
    fn flm_qwen36_35b_ct_int4_fallback_uploads_bf16_to_hip() {
        let Ok(flm_path_str) = std::env::var("SUPERSONIC_QWEN36_35B_CT_INT4_FLM_HIP_UPLOAD") else {
            eprintln!(
                "skip: SUPERSONIC_QWEN36_35B_CT_INT4_FLM_HIP_UPLOAD not set. Point it at a \
                 compressed-tensors INT4 Qwen3.6-35B-A3B FLM."
            );
            return;
        };
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        let store = BakedStore::open_flm_with_options(
            Path::new(&flm_path_str),
            FlmLoadOptions {
                flm_int4_logical_aliases: true,
                verify_block_hashes: false,
            },
        )
        .expect("open qwen3.6-35b CT INT4 FLM");

        let logical_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let meta = store
            .meta(logical_name)
            .expect("CT INT4 logical BF16 fallback alias");
        assert_eq!(meta.layout, LayoutTag::Raw);
        assert_eq!(meta.dtype, "bf16");
        assert_eq!(meta.shape, vec![8192, 2048]);
        assert!(
            store.raw_bytes(logical_name).is_none(),
            "transformed fallback aliases must not expose raw mmap bytes"
        );

        let weight = store
            .load_to_gpu(logical_name, 0)
            .expect("upload CT INT4 logical alias through BF16 fallback");
        assert_eq!(weight.dtype(), ScalarType::BF16);
        assert_eq!(weight.shape(), &[8192, 2048]);

        eprintln!("[flm-upload] OK — 35B CT INT4 fallback uploaded BF16 logical view to HIP");
    }

    #[test]
    fn virtual_arena_loads_baked_weight_and_expert_tensors() {
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        if !gpu_hal::vmm_is_supported(gpu_hal::Backend::Hip, 0) {
            eprintln!("skip: HIP VMM unsupported on this device/runtime");
            return;
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let bake_dir = tmp.path();
        let weights = (0..8192)
            .map(|idx| (idx as u8).wrapping_mul(13).wrapping_add(7))
            .collect::<Vec<_>>();
        std::fs::write(crate::weights_bin_path(bake_dir), &weights).expect("write weights.bin");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            converter_version: 1,
            model_family: "test".to_string(),
            quant_profile: None,
            source_format: None,
            source_quant: None,
            quant_method: None,
            tensors: vec![
                TensorMeta {
                    name: "lm_head.weight".to_string(),
                    shape: vec![4096],
                    dtype: "u8".to_string(),
                    layout: LayoutTag::Int4Quantized,
                    offset: 0,
                    byte_len: 4096,
                },
                TensorMeta {
                    name: "model.layers.0.mlp.experts.gate_up_proj".to_string(),
                    shape: vec![4096],
                    dtype: "u8".to_string(),
                    layout: LayoutTag::Int4Quantized,
                    offset: 4096,
                    byte_len: 4096,
                },
            ],
        };
        std::fs::write(
            crate::manifest_path(bake_dir),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let store = BakedStore::open(bake_dir).expect("open synthetic bake");
        let mut arena = BakedStore::virtual_weight_arena(0);
        let weight_id = store
            .load_to_virtual_arena(
                &mut arena,
                "lm_head.weight",
                gpu_hal::VirtualAllocationRole::Weights,
            )
            .expect("load virtual weight");
        let expert_id = store
            .load_to_virtual_arena(
                &mut arena,
                "model.layers.0.mlp.experts.gate_up_proj",
                gpu_hal::VirtualAllocationRole::MoeExpert,
            )
            .expect("load virtual expert");

        let stats = arena.stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.logical_bytes, 8192);
        assert_eq!(stats.logical_resident_bytes, 8192);
        assert!(stats.resident_bytes >= stats.logical_resident_bytes);

        let weight = arena
            .allocation(weight_id)
            .expect("weight allocation")
            .buffer()
            .to_host_bytes()
            .expect("read virtual weight");
        let expert = arena
            .allocation(expert_id)
            .expect("expert allocation")
            .buffer()
            .to_host_bytes()
            .expect("read virtual expert");
        assert_eq!(weight, weights[..4096]);
        assert_eq!(expert, weights[4096..]);
    }

    #[test]
    fn virtual_arena_load_initializes_mapped_pages_by_copy_without_clearing() {
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        if !gpu_hal::vmm_is_supported(gpu_hal::Backend::Hip, 0) {
            eprintln!("skip: HIP VMM unsupported on this device/runtime");
            return;
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let bake_dir = tmp.path();
        let weights = (0..4096)
            .map(|idx| (idx as u8).wrapping_mul(17).wrapping_add(3))
            .collect::<Vec<_>>();
        std::fs::write(crate::weights_bin_path(bake_dir), &weights).expect("write weights.bin");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            converter_version: 1,
            model_family: "test".to_string(),
            quant_profile: None,
            source_format: None,
            source_quant: None,
            quant_method: None,
            tensors: vec![TensorMeta {
                name: "model.layers.0.mlp.experts.gate_up_proj".to_string(),
                shape: vec![4096],
                dtype: "u8".to_string(),
                layout: LayoutTag::Int4Quantized,
                offset: 0,
                byte_len: 4096,
            }],
        };
        std::fs::write(
            crate::manifest_path(bake_dir),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let store = BakedStore::open(bake_dir).expect("open synthetic bake");
        let mut arena = BakedStore::virtual_weight_arena(0);

        gpu_hal::hal_profile_set_enabled(true);
        gpu_hal::hal_profile_reset();
        let allocation_id = store
            .load_to_virtual_arena(
                &mut arena,
                "model.layers.0.mlp.experts.gate_up_proj",
                gpu_hal::VirtualAllocationRole::MoeExpert,
            )
            .expect("load virtual expert");
        let profile = gpu_hal::hal_profile_snapshot();
        gpu_hal::hal_profile_set_enabled(false);

        assert!(
            profile
                .entries
                .iter()
                .any(|entry| entry.op == "vmm_map_no_sync"),
            "expected virtual upload to map pages with no-clear initialization, got {:?}",
            profile.entries
        );
        assert!(
            profile
                .entries
                .iter()
                .all(|entry| entry.op != "memset_zeros"),
            "virtual upload should not clear pages before immediately copying tensor bytes: {:?}",
            profile.entries
        );

        let loaded = arena
            .allocation(allocation_id)
            .expect("expert allocation")
            .buffer()
            .to_host_bytes()
            .expect("read virtual expert");
        assert_eq!(loaded, weights);
    }

    #[test]
    fn virtual_arena_rejects_manifest_byte_len_mismatch() {
        gpu_hal::set_backend(gpu_hal::Backend::Hip);
        if !gpu_hal::vmm_is_supported(gpu_hal::Backend::Hip, 0) {
            eprintln!("skip: HIP VMM unsupported on this device/runtime");
            return;
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let bake_dir = tmp.path();
        std::fs::write(crate::weights_bin_path(bake_dir), vec![0u8; 4]).expect("write weights.bin");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            converter_version: 1,
            model_family: "test".to_string(),
            quant_profile: None,
            source_format: None,
            source_quant: None,
            quant_method: None,
            tensors: vec![TensorMeta {
                name: "bad.weight".to_string(),
                shape: vec![2],
                dtype: "bf16".to_string(),
                layout: LayoutTag::Raw,
                offset: 0,
                byte_len: 2,
            }],
        };
        std::fs::write(
            crate::manifest_path(bake_dir),
            serde_json::to_string(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let store = BakedStore::open(bake_dir).expect("open synthetic bake");
        let mut arena = BakedStore::virtual_weight_arena(0);
        let err = store
            .load_to_virtual_arena(
                &mut arena,
                "bad.weight",
                gpu_hal::VirtualAllocationRole::Weights,
            )
            .expect_err("byte length mismatch should be rejected");
        assert!(
            err.to_string().contains("does not match"),
            "unexpected error: {err}"
        );
        assert_eq!(arena.stats().allocations, 0);
    }
}
