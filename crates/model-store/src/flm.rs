use std::collections::HashMap;

use crate::Error;

pub const ARCH_QWEN3_6_DENSE: u32 = 1;
pub const TOKENIZER_QWEN_BPE_V1: u32 = 1;
pub const TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1: u32 = 1;
pub const CODEC_RAW_BF16: u16 = 1;
pub const CODEC_SYM_INT4_G128_BF16: u16 = 2;
pub const CODEC_RAW_I64: u16 = 3;

const RUNTIME_MAGIC: &[u8; 8] = b"FLMRUN1\0";
const RUNTIME_VERSION: u16 = 1;
const SECTION_CONFIG_QWEN36_DENSE: u32 = 1;
const SECTION_TOKENIZER: u32 = 2;
const SECTION_CODEC_TABLE: u32 = 3;
const SECTION_TENSOR_ABI: u32 = 4;
const SECTION_ASSET_TABLE: u32 = 5;
const SECTION_ASSET_PAYLOADS: u32 = 6;
const SECTION_RECORD_SIZE: usize = 12;
const HEADER_PREFIX_SIZE: usize = 12;
const CONFIG_FIXED_SIZE: usize = 13 * 4 + 2 * 8 + 2 + 4;
const TOKENIZER_SIZE: usize = 8 * 4;
const CODEC_RECORD_SIZE: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub struct FlmQwen36DenseConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub activation_id: u8,
    pub tie_word_embeddings: bool,
    pub eos_token_ids: Vec<u32>,
    pub full_attention_layers: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmTokenizerDescriptor {
    pub tokenizer_id: u32,
    pub algorithm_id: u32,
    pub vocab_size: u32,
    pub vocab_asset_id: u32,
    pub merges_asset_id: u32,
    pub added_tokens_asset_id: u32,
    pub regex_asset_id: u32,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmAsset {
    pub asset_id: u32,
    pub kind: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmCodecDescriptor {
    pub codec_id: u16,
    pub semantic_id: u8,
    pub layout_id: u16,
    pub decoder_id: u16,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmTensorAbiDescriptor {
    pub abi_id: u32,
    pub architecture: String,
    pub tensor_format: String,
    pub producer: String,
    pub version: String,
}

#[derive(Debug, Clone)]
pub struct FlmRuntimeDirectory {
    pub architecture_id: u32,
    pub config: FlmQwen36DenseConfig,
    pub tokenizer: FlmTokenizerDescriptor,
    pub assets: HashMap<u32, FlmAsset>,
    codecs: Vec<FlmCodecDescriptor>,
    tensor_abi: FlmTensorAbiDescriptor,
}

#[derive(Debug, Clone, Copy)]
struct SectionRange {
    offset: usize,
    len: usize,
}

impl FlmRuntimeDirectory {
    pub fn parse(buf: &[u8]) -> Result<Self, Error> {
        let sections = parse_section_table(buf)?;
        let config = parse_qwen36_config(section(buf, &sections, SECTION_CONFIG_QWEN36_DENSE)?)?;
        let tokenizer = parse_tokenizer(section(buf, &sections, SECTION_TOKENIZER)?)?;
        let codecs = parse_codec_table(section(buf, &sections, SECTION_CODEC_TABLE)?)?;
        let tensor_abi = parse_tensor_abi(section(buf, &sections, SECTION_TENSOR_ABI)?)?;
        let assets = parse_assets(
            section(buf, &sections, SECTION_ASSET_TABLE)?,
            section(buf, &sections, SECTION_ASSET_PAYLOADS)?,
        )?;

        Ok(Self {
            architecture_id: ARCH_QWEN3_6_DENSE,
            config,
            tokenizer,
            assets,
            codecs,
            tensor_abi,
        })
    }

    pub fn qwen36_config(&self) -> Option<&FlmQwen36DenseConfig> {
        (self.architecture_id == ARCH_QWEN3_6_DENSE).then_some(&self.config)
    }

    pub fn tokenizer(&self) -> Option<&FlmTokenizerDescriptor> {
        Some(&self.tokenizer)
    }

    pub fn asset(&self, id: u32) -> Option<&FlmAsset> {
        self.assets.get(&id)
    }

    pub fn asset_by_kind(&self, kind: &str) -> Option<&FlmAsset> {
        self.assets.values().find(|asset| asset.kind == kind)
    }

    pub fn codecs(&self) -> &[FlmCodecDescriptor] {
        &self.codecs
    }

    pub fn codec(&self, id: u16) -> Option<&FlmCodecDescriptor> {
        self.codecs.iter().find(|codec| codec.codec_id == id)
    }

    pub fn tensor_abi(&self) -> &FlmTensorAbiDescriptor {
        &self.tensor_abi
    }
}

fn parse_section_table(buf: &[u8]) -> Result<HashMap<u32, SectionRange>, Error> {
    let magic = read_exact_range(buf, 0, RUNTIME_MAGIC.len(), "FLM runtime magic")?;
    if magic != RUNTIME_MAGIC {
        return Err(Error::Other(format!(
            "bad FLM runtime magic: expected {:?}, got {:?}",
            RUNTIME_MAGIC, magic
        )));
    }

    let version = read_u16(buf, 8, "FLM runtime version")?;
    if version != RUNTIME_VERSION {
        return Err(Error::Other(format!(
            "unsupported FLM runtime version {version}; expected {RUNTIME_VERSION}"
        )));
    }

    let section_count = read_u16(buf, 10, "FLM runtime section count")? as usize;
    let header_len = HEADER_PREFIX_SIZE
        .checked_add(
            section_count
                .checked_mul(SECTION_RECORD_SIZE)
                .ok_or_else(|| {
                    Error::Other("FLM runtime section table length overflows".to_string())
                })?,
        )
        .ok_or_else(|| Error::Other("FLM runtime header length overflows".to_string()))?;
    read_exact_range(buf, 0, header_len, "FLM runtime section table")?;

    let mut sections = HashMap::with_capacity(section_count);
    for idx in 0..section_count {
        let off = HEADER_PREFIX_SIZE + idx * SECTION_RECORD_SIZE;
        let section_id = read_u32(buf, off, "FLM runtime section id")?;
        if !(SECTION_CONFIG_QWEN36_DENSE..=SECTION_ASSET_PAYLOADS).contains(&section_id) {
            return Err(Error::Other(format!(
                "FLM runtime section {idx} has unknown id {section_id}"
            )));
        }
        let offset = u32_to_usize(read_u32(buf, off + 4, "FLM runtime section offset")?)?;
        let len = u32_to_usize(read_u32(buf, off + 8, "FLM runtime section length")?)?;
        if offset < header_len {
            return Err(Error::Other(format!(
                "FLM runtime section {section_id} starts inside header (offset={offset}, header_len={header_len})"
            )));
        }
        read_exact_range(buf, offset, len, "FLM runtime section bytes")?;
        if sections
            .insert(section_id, SectionRange { offset, len })
            .is_some()
        {
            return Err(Error::Other(format!(
                "FLM runtime has duplicate section id {section_id}"
            )));
        }
    }

    for required in SECTION_CONFIG_QWEN36_DENSE..=SECTION_ASSET_PAYLOADS {
        if !sections.contains_key(&required) {
            return Err(Error::Other(format!(
                "FLM runtime missing required section {required}"
            )));
        }
    }
    let mut ranges: Vec<(u32, SectionRange)> = sections
        .iter()
        .map(|(section_id, range)| (*section_id, *range))
        .collect();
    ranges.sort_by_key(|(_, range)| range.offset);
    let mut previous_end = header_len;
    for (section_id, range) in ranges {
        if range.offset < previous_end {
            return Err(Error::Other(format!(
                "FLM runtime section {section_id} overlaps a previous section"
            )));
        }
        previous_end = range
            .offset
            .checked_add(range.len)
            .ok_or_else(|| Error::Other("FLM runtime section range overflows".to_string()))?;
    }
    Ok(sections)
}

fn section<'a>(
    buf: &'a [u8],
    sections: &HashMap<u32, SectionRange>,
    section_id: u32,
) -> Result<&'a [u8], Error> {
    let range = sections.get(&section_id).ok_or_else(|| {
        Error::Other(format!("FLM runtime missing required section {section_id}"))
    })?;
    read_exact_range(buf, range.offset, range.len, "FLM runtime section")
}

fn parse_qwen36_config(buf: &[u8]) -> Result<FlmQwen36DenseConfig, Error> {
    read_exact_range(buf, 0, CONFIG_FIXED_SIZE + 8, "FLM qwen config")?;
    let mut offset = 0usize;
    let vocab_size = read_usize(buf, &mut offset, "FLM qwen vocab_size")?;
    let hidden_size = read_usize(buf, &mut offset, "FLM qwen hidden_size")?;
    let intermediate_size = read_usize(buf, &mut offset, "FLM qwen intermediate_size")?;
    let num_hidden_layers = read_usize(buf, &mut offset, "FLM qwen num_hidden_layers")?;
    let num_attention_heads = read_usize(buf, &mut offset, "FLM qwen num_attention_heads")?;
    let num_key_value_heads = read_usize(buf, &mut offset, "FLM qwen num_key_value_heads")?;
    let head_dim = read_usize(buf, &mut offset, "FLM qwen head_dim")?;
    let max_position_embeddings = read_usize(buf, &mut offset, "FLM qwen max_position_embeddings")?;
    let linear_conv_kernel_dim = read_usize(buf, &mut offset, "FLM qwen linear_conv_kernel_dim")?;
    let linear_key_head_dim = read_usize(buf, &mut offset, "FLM qwen linear_key_head_dim")?;
    let linear_value_head_dim = read_usize(buf, &mut offset, "FLM qwen linear_value_head_dim")?;
    let linear_num_key_heads = read_usize(buf, &mut offset, "FLM qwen linear_num_key_heads")?;
    let linear_num_value_heads = read_usize(buf, &mut offset, "FLM qwen linear_num_value_heads")?;
    let rms_norm_eps = read_f64_advance(buf, &mut offset, "FLM qwen rms_norm_eps")?;
    let rope_theta = read_f64_advance(buf, &mut offset, "FLM qwen rope_theta")?;
    let activation_id = *read_exact_range(buf, offset, 1, "FLM qwen activation_id")?
        .first()
        .expect("slice length checked");
    offset += 1;
    let tie_raw = *read_exact_range(buf, offset, 1, "FLM qwen tie_word_embeddings")?
        .first()
        .expect("slice length checked");
    offset += 1;
    let tie_word_embeddings = match tie_raw {
        0 => false,
        1 => true,
        other => {
            return Err(Error::Other(format!(
                "FLM qwen tie_word_embeddings has invalid bool value {other}"
            )));
        }
    };
    let eos_count = read_count(buf, &mut offset, "FLM qwen eos_count")?;
    let partial_rotary_factor =
        read_f64_advance(buf, &mut offset, "FLM qwen partial_rotary_factor")?;

    let mut eos_token_ids = Vec::with_capacity(eos_count);
    for idx in 0..eos_count {
        eos_token_ids.push(read_u32_advance(
            buf,
            &mut offset,
            &format!("FLM qwen eos token id {idx}"),
        )?);
    }

    let layer_count = read_count(buf, &mut offset, "FLM qwen full attention layer count")?;
    let mut full_attention_layers = Vec::with_capacity(layer_count);
    for idx in 0..layer_count {
        full_attention_layers.push(read_usize(
            buf,
            &mut offset,
            &format!("FLM qwen full attention layer {idx}"),
        )?);
    }

    ensure_consumed(buf, offset, "FLM qwen config")?;
    Ok(FlmQwen36DenseConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        linear_conv_kernel_dim,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        rms_norm_eps,
        rope_theta,
        partial_rotary_factor,
        activation_id,
        tie_word_embeddings,
        eos_token_ids,
        full_attention_layers,
    })
}

fn parse_tokenizer(buf: &[u8]) -> Result<FlmTokenizerDescriptor, Error> {
    if buf.len() != TOKENIZER_SIZE {
        return Err(Error::Other(format!(
            "FLM tokenizer section has len {}; expected {TOKENIZER_SIZE}",
            buf.len()
        )));
    }
    let mut offset = 0usize;
    Ok(FlmTokenizerDescriptor {
        tokenizer_id: read_u32_advance(buf, &mut offset, "FLM tokenizer_id")?,
        algorithm_id: read_u32_advance(buf, &mut offset, "FLM tokenizer algorithm_id")?,
        vocab_size: read_u32_advance(buf, &mut offset, "FLM tokenizer vocab_size")?,
        vocab_asset_id: read_u32_advance(buf, &mut offset, "FLM tokenizer vocab_asset_id")?,
        merges_asset_id: read_u32_advance(buf, &mut offset, "FLM tokenizer merges_asset_id")?,
        added_tokens_asset_id: read_u32_advance(
            buf,
            &mut offset,
            "FLM tokenizer added_tokens_asset_id",
        )?,
        regex_asset_id: read_u32_advance(buf, &mut offset, "FLM tokenizer regex_asset_id")?,
        flags: read_u32_advance(buf, &mut offset, "FLM tokenizer flags")?,
    })
}

fn parse_codec_table(buf: &[u8]) -> Result<Vec<FlmCodecDescriptor>, Error> {
    let mut offset = 0usize;
    let count = read_count(buf, &mut offset, "FLM codec count")?;
    let expected_len = 4usize
        .checked_add(
            count
                .checked_mul(CODEC_RECORD_SIZE)
                .ok_or_else(|| Error::Other("FLM codec table length overflows".to_string()))?,
        )
        .ok_or_else(|| Error::Other("FLM codec table length overflows".to_string()))?;
    if buf.len() != expected_len {
        return Err(Error::Other(format!(
            "FLM codec table has len {}; expected {expected_len}",
            buf.len()
        )));
    }

    let mut codecs = Vec::with_capacity(count);
    for idx in 0..count {
        let codec_id = read_exact_range(buf, offset, 1, "FLM codec id")?[0] as u16;
        offset += 1;
        let semantic_id = read_exact_range(buf, offset, 1, "FLM codec semantic id")?[0];
        offset += 1;
        let layout_id = read_u16_advance(buf, &mut offset, "FLM codec layout_id")?;
        let decoder_id = read_u16_advance(buf, &mut offset, "FLM codec decoder_id")?;
        let flags = read_u32_advance(buf, &mut offset, "FLM codec flags")?;
        if codecs
            .iter()
            .any(|existing: &FlmCodecDescriptor| existing.codec_id == codec_id)
        {
            return Err(Error::Other(format!(
                "FLM codec table has duplicate codec id {codec_id}"
            )));
        }
        codecs.push(FlmCodecDescriptor {
            codec_id,
            semantic_id,
            layout_id,
            decoder_id,
            flags,
        });
        debug_assert_eq!(offset, 4 + (idx + 1) * CODEC_RECORD_SIZE);
    }
    Ok(codecs)
}

fn parse_tensor_abi(buf: &[u8]) -> Result<FlmTensorAbiDescriptor, Error> {
    let mut offset = 0usize;
    let abi_id = read_u32_advance(buf, &mut offset, "FLM tensor ABI id")?;
    let architecture = read_string_advance(buf, &mut offset, "FLM tensor ABI architecture")?;
    let tensor_format = read_string_advance(buf, &mut offset, "FLM tensor ABI format")?;
    let producer = read_string_advance(buf, &mut offset, "FLM tensor ABI producer")?;
    let version = read_string_advance(buf, &mut offset, "FLM tensor ABI version")?;
    ensure_consumed(buf, offset, "FLM tensor ABI")?;
    Ok(FlmTensorAbiDescriptor {
        abi_id,
        architecture,
        tensor_format,
        producer,
        version,
    })
}

fn parse_assets(table: &[u8], payloads: &[u8]) -> Result<HashMap<u32, FlmAsset>, Error> {
    let mut offset = 0usize;
    let count = read_count(table, &mut offset, "FLM asset count")?;
    let mut assets = HashMap::with_capacity(count);
    for idx in 0..count {
        let asset_id = read_u32_advance(table, &mut offset, "FLM asset_id")?;
        let payload_offset = u32_to_usize(read_u32_advance(
            table,
            &mut offset,
            "FLM asset payload_offset",
        )?)?;
        let payload_len = u32_to_usize(read_u32_advance(
            table,
            &mut offset,
            "FLM asset payload_len",
        )?)?;
        let kind_len = u32_to_usize(read_u32_advance(table, &mut offset, "FLM asset kind_len")?)?;
        let kind_bytes = read_exact_range(table, offset, kind_len, "FLM asset kind")?;
        offset += kind_len;
        let kind = std::str::from_utf8(kind_bytes)
            .map_err(|e| Error::Other(format!("FLM asset {asset_id} kind is not UTF-8: {e}")))?
            .to_string();
        let payload = read_exact_range(
            payloads,
            payload_offset,
            payload_len,
            &format!("FLM asset {asset_id} payload"),
        )?
        .to_vec();
        if assets
            .insert(
                asset_id,
                FlmAsset {
                    asset_id,
                    kind,
                    payload,
                },
            )
            .is_some()
        {
            return Err(Error::Other(format!(
                "FLM asset table has duplicate asset id {asset_id}"
            )));
        }
        debug_assert!(idx < count);
    }
    ensure_consumed(table, offset, "FLM asset table")?;
    Ok(assets)
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
            "{what}: range [{offset}, {end}) exceeds length {}",
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

fn read_f64(buf: &[u8], offset: usize, what: &str) -> Result<f64, Error> {
    let bytes: [u8; 8] = read_exact_range(buf, offset, 8, what)?
        .try_into()
        .expect("slice length checked");
    Ok(f64::from_le_bytes(bytes))
}

fn read_u16_advance(buf: &[u8], offset: &mut usize, what: &str) -> Result<u16, Error> {
    let value = read_u16(buf, *offset, what)?;
    *offset += 2;
    Ok(value)
}

fn read_u32_advance(buf: &[u8], offset: &mut usize, what: &str) -> Result<u32, Error> {
    let value = read_u32(buf, *offset, what)?;
    *offset += 4;
    Ok(value)
}

fn read_f64_advance(buf: &[u8], offset: &mut usize, what: &str) -> Result<f64, Error> {
    let value = read_f64(buf, *offset, what)?;
    *offset += 8;
    Ok(value)
}

fn read_count(buf: &[u8], offset: &mut usize, what: &str) -> Result<usize, Error> {
    u32_to_usize(read_u32_advance(buf, offset, what)?)
}

fn read_usize(buf: &[u8], offset: &mut usize, what: &str) -> Result<usize, Error> {
    u32_to_usize(read_u32_advance(buf, offset, what)?)
}

fn read_string_advance(buf: &[u8], offset: &mut usize, what: &str) -> Result<String, Error> {
    let len = read_count(buf, offset, what)?;
    let bytes = read_exact_range(buf, *offset, len, what)?;
    *offset += len;
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|e| Error::Other(format!("{what} is not UTF-8: {e}")))
}

fn ensure_consumed(buf: &[u8], offset: usize, what: &str) -> Result<(), Error> {
    if offset != buf.len() {
        return Err(Error::Other(format!(
            "{what}: trailing bytes at offset {offset} of {}",
            buf.len()
        )));
    }
    Ok(())
}

fn u32_to_usize(value: u32) -> Result<usize, Error> {
    usize::try_from(value)
        .map_err(|_| Error::Other(format!("FLM runtime value {value} does not fit usize")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_f64(out: &mut Vec<u8>, value: f64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(out: &mut Vec<u8>, value: &str) {
        write_u32(out, value.len() as u32);
        out.extend_from_slice(value.as_bytes());
    }

    fn build_qwen_config_section() -> Vec<u8> {
        let mut out = Vec::new();
        for value in [
            151_936u32, 5120, 27_648, 62, 40, 8, 128, 262_144, 128, 256, 256, 16, 32,
        ] {
            write_u32(&mut out, value);
        }
        write_f64(&mut out, 1e-6);
        write_f64(&mut out, 10_000_000.0);
        out.push(1);
        out.push(0);
        write_u32(&mut out, 2);
        write_f64(&mut out, 0.25);
        write_u32(&mut out, 151_645);
        write_u32(&mut out, 151_643);
        write_u32(&mut out, 3);
        write_u32(&mut out, 3);
        write_u32(&mut out, 7);
        write_u32(&mut out, 11);
        out
    }

    fn build_tokenizer_section() -> Vec<u8> {
        let mut out = Vec::new();
        for value in [1u32, TOKENIZER_QWEN_BPE_V1, 151_936, 1, 2, 3, 4, 0] {
            write_u32(&mut out, value);
        }
        out
    }

    fn build_codec_table_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, 3);
        for (codec_id, semantic_id, layout_id, decoder_id, flags) in [
            (CODEC_RAW_BF16 as u8, 1u8, 1u16, 1u16, 0u32),
            (CODEC_SYM_INT4_G128_BF16 as u8, 2u8, 2u16, 2u16, 0u32),
            (CODEC_RAW_I64 as u8, 3u8, 1u16, 1u16, 0u32),
        ] {
            out.push(codec_id);
            out.push(semantic_id);
            write_u16(&mut out, layout_id);
            write_u16(&mut out, decoder_id);
            write_u32(&mut out, flags);
        }
        out
    }

    fn build_tensor_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1);
        write_string(&mut out, "qwen3.6-dense");
        write_string(&mut out, "compressed-tensors-int4");
        write_string(&mut out, "geo-quant");
        write_string(&mut out, "v1");
        out
    }

    fn build_asset_sections() -> (Vec<u8>, Vec<u8>) {
        let assets = [
            (1u32, "tokenizer_vocab", br#"{"hello":0}"#.as_slice()),
            (2u32, "tokenizer_merges", b"#version: 0.2\n".as_slice()),
            (3u32, "tokenizer_added_tokens", b"[]".as_slice()),
            (4u32, "tokenizer_regex", br#"\p{L}+"#.as_slice()),
        ];
        let mut table = Vec::new();
        let mut payloads = Vec::new();
        write_u32(&mut table, assets.len() as u32);
        for (asset_id, kind, payload) in assets {
            write_u32(&mut table, asset_id);
            write_u32(&mut table, payloads.len() as u32);
            write_u32(&mut table, payload.len() as u32);
            write_u32(&mut table, kind.len() as u32);
            table.extend_from_slice(kind.as_bytes());
            payloads.extend_from_slice(payload);
        }
        (table, payloads)
    }

    fn build_test_runtime_directory() -> Vec<u8> {
        let (asset_table, asset_payloads) = build_asset_sections();
        let sections = [
            (1u32, build_qwen_config_section()),
            (2u32, build_tokenizer_section()),
            (3u32, build_codec_table_section()),
            (4u32, build_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
        ];
        let header_len = 8 + 2 + 2 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        write_u16(&mut out, 1);
        write_u16(&mut out, sections.len() as u16);
        for (section_id, data) in &sections {
            write_u32(&mut out, *section_id);
            write_u32(&mut out, offset);
            write_u32(&mut out, data.len() as u32);
            offset += data.len() as u32;
        }
        for (_, data) in sections {
            out.extend_from_slice(&data);
        }
        out
    }

    #[test]
    fn parses_runtime_directory_with_qwen_config_and_assets() {
        let runtime = build_test_runtime_directory();
        let parsed = FlmRuntimeDirectory::parse(&runtime).expect("parse runtime");

        assert_eq!(parsed.architecture_id, ARCH_QWEN3_6_DENSE);
        assert_eq!(parsed.qwen36_config().unwrap().hidden_size, 5120);
        assert_eq!(parsed.qwen36_config().unwrap().full_attention_layers[0], 3);
        assert_eq!(parsed.tokenizer().unwrap().vocab_asset_id, 1);
        assert_eq!(parsed.asset(4).unwrap().kind, "tokenizer_regex");
    }
}
