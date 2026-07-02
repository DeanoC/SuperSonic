use std::collections::{HashMap, HashSet};

use crate::Error;

pub const ARCH_QWEN3_6_DENSE: u32 = 1;
pub const ARCH_QWEN3_6_MOE: u32 = 2;
pub const TOKENIZER_QWEN_BPE_V1: u32 = 1;
pub const TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1: u32 = 1;
pub const TENSOR_ABI_QWEN3_6_MOE_MIXED_LOWBIT_V1: u32 = 2;
pub const CODEC_RAW_BF16: u16 = 1;
pub const CODEC_SYM_INT4_G128_BF16: u16 = 2;
pub const CODEC_RAW_I64: u16 = 3;
pub const CODEC_NVFP4_E2M1_B16_E4M3_F32: u16 = 4;
pub const CODEC_MXFP4_E2M1_B32_E8M0: u16 = 5;
pub const CODEC_MXFP8_E4M3_B32_E8M0: u16 = 6;
pub const CODEC_FP8_E4M3_F32: u16 = 7;
pub const CODEC_FP8_E4M3_B128_BF16_INV: u16 = 8;
pub const MODEL_QWEN3_6_DENSE_V1: u16 = 1;
pub const MODEL_QWEN3_6_MOE_V1: u16 = 2;
pub const QUANT_PROFILE_QWEN3_6_DENSE_CT_INT4_G128_BF16_V1: u32 = 1;
pub const QUANT_PROFILE_QWEN3_6_MOE_MIXED_LOWBIT_V1: u32 = 2;
pub const ASSET_TOKENIZER_VOCAB: u16 = 1;
pub const ASSET_TOKENIZER_MERGES: u16 = 2;
pub const ASSET_TOKENIZER_ADDED_TOKENS: u16 = 3;
pub const ASSET_TOKENIZER_REGEX: u16 = 4;
pub const ASSET_HF_CONFIG_JSON: u16 = 101;
pub const ASSET_HF_TOKENIZER_JSON: u16 = 102;
pub const ASSET_FLAG_REQUIRED_FOR_RUNTIME: u16 = 1 << 0;
pub const ASSET_FLAG_COMPATIBILITY_ONLY: u16 = 1 << 1;
pub const MANIFEST_COMPANION_NONE: u8 = 0;
pub const MANIFEST_COMPANION_PACKED: u8 = 1;
pub const MANIFEST_COMPANION_SCALE: u8 = 2;
pub const MANIFEST_COMPANION_SHAPE: u8 = 3;
pub const MANIFEST_COMPANION_SYNTHETIC_ZERO: u8 = 4;
pub const MANIFEST_FLAG_REQUIRED: u8 = 1 << 0;
pub const MANIFEST_FLAG_OPTIONAL: u8 = 1 << 1;
pub const MANIFEST_FLAG_DERIVED_ALIAS: u8 = 1 << 3;
pub const STORAGE_ROLE_VALUE: u16 = 0;
pub const STORAGE_ROLE_PACKED: u16 = 1;
pub const STORAGE_ROLE_SCALE: u16 = 2;
pub const STORAGE_ROLE_ZERO: u16 = 3;
pub const STORAGE_ROLE_SHAPE: u16 = 4;
pub const STORAGE_ROLE_GLOBAL_SCALE: u16 = 6;
pub const STORAGE_ROLE_INPUT_SCALE: u16 = 7;
pub const STORAGE_ABI_KIND_GROUP_QUANT: u16 = 1;
pub const STORAGE_ABI_KIND_SCALED_FLOAT: u16 = 2;
pub const STORAGE_ABI_ID_NONE: u16 = 0;
pub const LAYOUT_ID_DEFAULT: u16 = 0;
pub const QUANT_FLAG_SYMMETRIC: u16 = 1 << 0;
pub const LOGICAL_TENSOR_ROLE_WEIGHT: u16 = 1;
pub const LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT: u16 = 2;
pub const LOGICAL_TENSOR_FLAG_REQUIRED: u16 = 1 << 0;
pub const STORAGE_BINDING_FLAG_REQUIRED: u16 = 1 << 0;
pub const VALUE_FORMAT_RAW_DENSE: u16 = 1;
pub const VALUE_FORMAT_SYM_INT4: u16 = 2;
pub const VALUE_FORMAT_NVFP4_E2M1: u16 = 3;
pub const VALUE_FORMAT_MXFP4_E2M1: u16 = 4;
pub const VALUE_FORMAT_MXFP8_E4M3: u16 = 5;
pub const VALUE_FORMAT_FP8_E4M3_F32: u16 = 6;
pub const VALUE_FORMAT_FP8_E4M3_B128_BF16_INV: u16 = 7;
pub const CONSUME_STRATEGY_DIRECT: u16 = 1;
pub const PLAN_STREAM_DEFAULT: u16 = 0;
pub const PLAN_PRIORITY_DEFAULT: u16 = 0;
pub const PLAN_STEP_FLAG_NONE: u32 = 0;
pub const FLM_DTYPE_FP32: u16 = 0;
pub const FLM_DTYPE_BF16: u16 = 2;
pub const FLM_DTYPE_FP8_E4M3: u16 = 3;
pub const FLM_DTYPE_UINT8: u16 = 4;
pub const FLM_DTYPE_INT32: u16 = 5;
pub const FLM_DTYPE_INT64: u16 = 6;

const RUNTIME_MAGIC: &[u8; 8] = b"FLMRUN1\0";
const RUNTIME_VERSION: u16 = 4;
const SECTION_CONFIG_QWEN36_DENSE: u32 = 1;
const SECTION_TOKENIZER: u32 = 2;
const SECTION_CODEC_TABLE: u32 = 3;
const SECTION_TENSOR_ABI: u32 = 4;
const SECTION_ASSET_TABLE: u32 = 5;
const SECTION_ASSET_PAYLOADS: u32 = 6;
const SECTION_MODEL_DESCRIPTOR: u32 = 7;
const SECTION_TENSOR_MANIFEST: u32 = 8;
const SECTION_STORAGE_ABI_TABLE: u32 = 9;
const SECTION_LOGICAL_TENSOR_TABLE: u32 = 10;
const SECTION_STORAGE_BINDING_TABLE: u32 = 11;
const SECTION_PLAN_STEP_TABLE: u32 = 12;
const SECTION_CONFIG_QWEN36_MOE: u32 = 13;
const SECTION_RECORD_SIZE: usize = 12;
const HEADER_PREFIX_SIZE: usize = 16;
const CONFIG_FIXED_SIZE: usize = 13 * 4 + 2 * 8 + 2 + 4;
const MOE_CONFIG_FIXED_SIZE: usize = 17 * 4 + 2 * 8 + 5 + 4;
const TOKENIZER_SIZE: usize = 8 * 4;
const CODEC_RECORD_SIZE: usize = 10;
const MODEL_DESCRIPTOR_SIZE: usize = 24;
const TENSOR_MANIFEST_HEADER_SIZE: usize = 12;
const TENSOR_MANIFEST_ROW_SIZE: usize = 40;
const STORAGE_ABI_HEADER_SIZE: usize = 12;
const STORAGE_ABI_ROW_SIZE: usize = 21;
const LOGICAL_TENSOR_HEADER_SIZE: usize = 12;
const LOGICAL_TENSOR_ROW_SIZE: usize = 44;
const STORAGE_BINDING_HEADER_SIZE: usize = 12;
const STORAGE_BINDING_ROW_SIZE: usize = 20;
const PLAN_STEP_HEADER_SIZE: usize = 8;
const PLAN_STEP_ROW_SIZE: usize = 38;

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
pub struct FlmQwen36MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
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
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub mtp_num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub activation_id: u8,
    pub tie_word_embeddings: bool,
    pub attn_output_gate: bool,
    pub mtp_use_dedicated_embeddings: bool,
    pub mrope_interleaved: bool,
    pub eos_token_ids: Vec<u32>,
    pub full_attention_layers: Vec<usize>,
    pub mrope_section: [u32; 3],
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
    pub kind_id: u16,
    pub flags: u16,
    pub name: String,
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
    pub weight_prefix: String,
    pub int4_packed_suffix: String,
    pub int4_scale_suffix: String,
    pub int4_shape_suffix: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmModelDescriptor {
    pub descriptor_version: u16,
    pub model_id: u16,
    pub config_section_id: u32,
    pub tokenizer_id: u32,
    pub tensor_abi_id: u32,
    pub quant_profile_id: u32,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmTensorManifestRow {
    pub role_id: u32,
    pub group_id: u32,
    pub companion_kind: u8,
    pub rank: u8,
    pub dtype: u16,
    pub logical_dtype: u16,
    pub codec_id: u8,
    pub flags: u8,
    pub shape: [u32; 4],
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmTensorManifest {
    pub rows: Vec<FlmTensorManifestRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmStorageAbi {
    pub storage_abi_id: u16,
    pub abi_kind_id: u16,
    pub codec_semantic_id: u16,
    pub layout_id: u16,
    pub bits: u8,
    pub group_size: u16,
    pub quant_flags: u16,
    pub params: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmLogicalTensor {
    pub tensor_id: u32,
    pub name: String,
    pub role_id: u16,
    pub rank: u8,
    pub shape: [u32; 4],
    pub value_format_id: u16,
    pub reconstruction_dtype: u16,
    pub storage_binding_start: u32,
    pub storage_binding_count: u16,
    pub flags: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmStorageBinding {
    pub logical_tensor_id: u32,
    pub storage_role: u16,
    pub tensor_name: String,
    pub storage_dtype: u16,
    pub storage_abi_id: u16,
    pub flags: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlmPlanStep {
    pub logical_tensor_id: u32,
    pub storage_role: u16,
    pub consume_strategy: u16,
    pub target_layout_id: u16,
    pub target_dtype: u16,
    pub target_rank: u8,
    pub target_shape: [u32; 4],
    pub stream_id: u16,
    pub priority: u16,
    pub flags: u32,
}

#[derive(Debug, Clone)]
pub struct FlmRuntimeDirectory {
    pub architecture_id: u32,
    pub config: FlmQwen36DenseConfig,
    pub moe_config: Option<FlmQwen36MoeConfig>,
    pub tokenizer: FlmTokenizerDescriptor,
    pub assets: HashMap<u32, FlmAsset>,
    codecs: Vec<FlmCodecDescriptor>,
    tensor_abi: FlmTensorAbiDescriptor,
    model_descriptor: FlmModelDescriptor,
    tensor_manifest: FlmTensorManifest,
    storage_abis: Vec<FlmStorageAbi>,
    logical_tensors: Vec<FlmLogicalTensor>,
    storage_bindings: Vec<FlmStorageBinding>,
    plan_steps: Vec<FlmPlanStep>,
}

#[derive(Debug, Clone, Copy)]
struct SectionRange {
    offset: usize,
    len: usize,
}

impl FlmRuntimeDirectory {
    pub fn parse(buf: &[u8]) -> Result<Self, Error> {
        let (architecture_id, sections) = parse_section_table(buf)?;
        let (config, moe_config) = if architecture_id == ARCH_QWEN3_6_MOE {
            let moe = parse_qwen36_moe_config(section(buf, &sections, SECTION_CONFIG_QWEN36_MOE)?)?;
            (dense_config_from_moe(&moe), Some(moe))
        } else {
            (
                parse_qwen36_config(section(buf, &sections, SECTION_CONFIG_QWEN36_DENSE)?)?,
                None,
            )
        };
        let tokenizer = parse_tokenizer(section(buf, &sections, SECTION_TOKENIZER)?)?;
        let codecs = parse_codec_table(section(buf, &sections, SECTION_CODEC_TABLE)?)?;
        let tensor_abi = parse_tensor_abi(section(buf, &sections, SECTION_TENSOR_ABI)?)?;
        let model_descriptor =
            parse_model_descriptor(section(buf, &sections, SECTION_MODEL_DESCRIPTOR)?)?;
        validate_model_descriptor(&model_descriptor, architecture_id, &tokenizer, &tensor_abi)?;
        let assets = parse_assets(
            section(buf, &sections, SECTION_ASSET_TABLE)?,
            section(buf, &sections, SECTION_ASSET_PAYLOADS)?,
        )?;
        let tensor_manifest =
            parse_tensor_manifest(section(buf, &sections, SECTION_TENSOR_MANIFEST)?)?;
        let stage3_ids = [
            SECTION_STORAGE_ABI_TABLE,
            SECTION_LOGICAL_TENSOR_TABLE,
            SECTION_STORAGE_BINDING_TABLE,
            SECTION_PLAN_STEP_TABLE,
        ];
        let has_stage3 = stage3_ids
            .iter()
            .any(|section_id| sections.contains_key(section_id));
        if has_stage3
            && !stage3_ids
                .iter()
                .all(|section_id| sections.contains_key(section_id))
        {
            return Err(Error::Other(
                "FLM runtime has incomplete Stage 3 storage tables".to_string(),
            ));
        }
        let (storage_abis, logical_tensors, storage_bindings, plan_steps) = if has_stage3 {
            (
                parse_storage_abis(section(buf, &sections, SECTION_STORAGE_ABI_TABLE)?)?,
                parse_logical_tensors(section(buf, &sections, SECTION_LOGICAL_TENSOR_TABLE)?)?,
                parse_storage_bindings(section(buf, &sections, SECTION_STORAGE_BINDING_TABLE)?)?,
                parse_plan_steps(section(buf, &sections, SECTION_PLAN_STEP_TABLE)?)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        Ok(Self {
            architecture_id,
            config,
            moe_config,
            tokenizer,
            assets,
            codecs,
            tensor_abi,
            model_descriptor,
            tensor_manifest,
            storage_abis,
            logical_tensors,
            storage_bindings,
            plan_steps,
        })
    }

    pub fn qwen36_config(&self) -> Option<&FlmQwen36DenseConfig> {
        (self.architecture_id == ARCH_QWEN3_6_DENSE).then_some(&self.config)
    }

    pub fn qwen36_moe_config(&self) -> Option<&FlmQwen36MoeConfig> {
        (self.architecture_id == ARCH_QWEN3_6_MOE)
            .then_some(self.moe_config.as_ref())
            .flatten()
    }

    pub fn tokenizer(&self) -> Option<&FlmTokenizerDescriptor> {
        Some(&self.tokenizer)
    }

    pub fn asset(&self, id: u32) -> Option<&FlmAsset> {
        self.assets.get(&id)
    }

    pub fn asset_by_kind(&self, kind: &str) -> Option<&FlmAsset> {
        self.assets.values().find(|asset| asset.name == kind)
    }

    pub fn codecs(&self) -> &[FlmCodecDescriptor] {
        &self.codecs
    }

    pub fn codec_by_id(&self, id: u16) -> Option<&FlmCodecDescriptor> {
        self.codecs.iter().find(|codec| codec.codec_id == id)
    }

    pub fn codec_by_semantic_id(&self, semantic_id: u16) -> Option<&FlmCodecDescriptor> {
        self.codecs
            .iter()
            .find(|codec| codec.semantic_id as u16 == semantic_id)
    }

    pub fn tensor_abi(&self) -> &FlmTensorAbiDescriptor {
        &self.tensor_abi
    }

    pub fn model_descriptor(&self) -> &FlmModelDescriptor {
        &self.model_descriptor
    }

    pub fn tensor_manifest(&self) -> &FlmTensorManifest {
        &self.tensor_manifest
    }

    pub fn storage_abis(&self) -> &[FlmStorageAbi] {
        &self.storage_abis
    }

    pub fn logical_tensors(&self) -> &[FlmLogicalTensor] {
        &self.logical_tensors
    }

    pub fn storage_bindings(&self) -> &[FlmStorageBinding] {
        &self.storage_bindings
    }

    pub fn plan_steps(&self) -> &[FlmPlanStep] {
        &self.plan_steps
    }
}

fn config_section_id_for_architecture(architecture_id: u32) -> Result<u32, Error> {
    match architecture_id {
        ARCH_QWEN3_6_DENSE => Ok(SECTION_CONFIG_QWEN36_DENSE),
        ARCH_QWEN3_6_MOE => Ok(SECTION_CONFIG_QWEN36_MOE),
        other => Err(Error::Other(format!(
            "unsupported FLM runtime architecture {other}"
        ))),
    }
}

fn parse_section_table(buf: &[u8]) -> Result<(u32, HashMap<u32, SectionRange>), Error> {
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
    let architecture_id = read_u32(buf, 12, "FLM runtime architecture_id")?;
    if architecture_id != ARCH_QWEN3_6_DENSE && architecture_id != ARCH_QWEN3_6_MOE {
        return Err(Error::Other(format!(
            "unsupported FLM runtime architecture {architecture_id}"
        )));
    }
    let config_section_id = config_section_id_for_architecture(architecture_id)?;
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
        if !(SECTION_CONFIG_QWEN36_DENSE..=SECTION_CONFIG_QWEN36_MOE).contains(&section_id) {
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

    for required in [
        config_section_id,
        SECTION_TOKENIZER,
        SECTION_CODEC_TABLE,
        SECTION_TENSOR_ABI,
        SECTION_ASSET_TABLE,
        SECTION_ASSET_PAYLOADS,
        SECTION_MODEL_DESCRIPTOR,
        SECTION_TENSOR_MANIFEST,
    ] {
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
        if range.offset != previous_end {
            return Err(Error::Other(format!(
                "FLM runtime section {section_id} is not contiguous (offset={}, expected={previous_end})",
                range.offset
            )));
        }
        previous_end = range
            .offset
            .checked_add(range.len)
            .ok_or_else(|| Error::Other("FLM runtime section range overflows".to_string()))?;
    }
    if previous_end != buf.len() {
        return Err(Error::Other(format!(
            "FLM runtime has trailing bytes after sections (final_end={previous_end}, len={})",
            buf.len()
        )));
    }
    Ok((architecture_id, sections))
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

fn parse_qwen36_moe_config(buf: &[u8]) -> Result<FlmQwen36MoeConfig, Error> {
    read_exact_range(
        buf,
        0,
        MOE_CONFIG_FIXED_SIZE + 8 + 12,
        "FLM qwen moe config",
    )?;
    let mut offset = 0usize;
    let vocab_size = read_usize(buf, &mut offset, "FLM qwen moe vocab_size")?;
    let hidden_size = read_usize(buf, &mut offset, "FLM qwen moe hidden_size")?;
    let moe_intermediate_size = read_usize(buf, &mut offset, "FLM qwen moe moe_intermediate_size")?;
    let shared_expert_intermediate_size = read_usize(
        buf,
        &mut offset,
        "FLM qwen moe shared_expert_intermediate_size",
    )?;
    let num_hidden_layers = read_usize(buf, &mut offset, "FLM qwen moe num_hidden_layers")?;
    let num_attention_heads = read_usize(buf, &mut offset, "FLM qwen moe num_attention_heads")?;
    let num_key_value_heads = read_usize(buf, &mut offset, "FLM qwen moe num_key_value_heads")?;
    let head_dim = read_usize(buf, &mut offset, "FLM qwen moe head_dim")?;
    let max_position_embeddings =
        read_usize(buf, &mut offset, "FLM qwen moe max_position_embeddings")?;
    let linear_conv_kernel_dim =
        read_usize(buf, &mut offset, "FLM qwen moe linear_conv_kernel_dim")?;
    let linear_key_head_dim = read_usize(buf, &mut offset, "FLM qwen moe linear_key_head_dim")?;
    let linear_value_head_dim = read_usize(buf, &mut offset, "FLM qwen moe linear_value_head_dim")?;
    let linear_num_key_heads = read_usize(buf, &mut offset, "FLM qwen moe linear_num_key_heads")?;
    let linear_num_value_heads =
        read_usize(buf, &mut offset, "FLM qwen moe linear_num_value_heads")?;
    let num_experts = read_usize(buf, &mut offset, "FLM qwen moe num_experts")?;
    let num_experts_per_tok = read_usize(buf, &mut offset, "FLM qwen moe num_experts_per_tok")?;
    let mtp_num_hidden_layers = read_usize(buf, &mut offset, "FLM qwen moe mtp_num_hidden_layers")?;
    let rms_norm_eps = read_f64_advance(buf, &mut offset, "FLM qwen moe rms_norm_eps")?;
    let rope_theta = read_f64_advance(buf, &mut offset, "FLM qwen moe rope_theta")?;
    let activation_id = read_u8_advance(buf, &mut offset, "FLM qwen moe activation_id")?;
    let tie_word_embeddings =
        read_bool_advance(buf, &mut offset, "FLM qwen moe tie_word_embeddings")?;
    let attn_output_gate = read_bool_advance(buf, &mut offset, "FLM qwen moe attn_output_gate")?;
    let mtp_use_dedicated_embeddings = read_bool_advance(
        buf,
        &mut offset,
        "FLM qwen moe mtp_use_dedicated_embeddings",
    )?;
    let mrope_interleaved = read_bool_advance(buf, &mut offset, "FLM qwen moe mrope_interleaved")?;
    let eos_count = read_count(buf, &mut offset, "FLM qwen moe eos_count")?;
    let partial_rotary_factor =
        read_f64_advance(buf, &mut offset, "FLM qwen moe partial_rotary_factor")?;

    let mut eos_token_ids = Vec::with_capacity(eos_count);
    for idx in 0..eos_count {
        eos_token_ids.push(read_u32_advance(
            buf,
            &mut offset,
            &format!("FLM qwen moe eos token id {idx}"),
        )?);
    }

    let layer_count = read_count(buf, &mut offset, "FLM qwen moe full attention layer count")?;
    let mut full_attention_layers = Vec::with_capacity(layer_count);
    for idx in 0..layer_count {
        full_attention_layers.push(read_usize(
            buf,
            &mut offset,
            &format!("FLM qwen moe full attention layer {idx}"),
        )?);
    }

    let mut mrope_section = [0u32; 3];
    for (idx, value) in mrope_section.iter_mut().enumerate() {
        *value = read_u32_advance(
            buf,
            &mut offset,
            &format!("FLM qwen moe mrope_section {idx}"),
        )?;
    }

    ensure_consumed(buf, offset, "FLM qwen moe config")?;
    Ok(FlmQwen36MoeConfig {
        vocab_size,
        hidden_size,
        moe_intermediate_size,
        shared_expert_intermediate_size,
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
        num_experts,
        num_experts_per_tok,
        mtp_num_hidden_layers,
        rms_norm_eps,
        rope_theta,
        partial_rotary_factor,
        activation_id,
        tie_word_embeddings,
        attn_output_gate,
        mtp_use_dedicated_embeddings,
        mrope_interleaved,
        eos_token_ids,
        full_attention_layers,
        mrope_section,
    })
}

fn dense_config_from_moe(moe: &FlmQwen36MoeConfig) -> FlmQwen36DenseConfig {
    FlmQwen36DenseConfig {
        vocab_size: moe.vocab_size,
        hidden_size: moe.hidden_size,
        intermediate_size: moe.moe_intermediate_size,
        num_hidden_layers: moe.num_hidden_layers,
        num_attention_heads: moe.num_attention_heads,
        num_key_value_heads: moe.num_key_value_heads,
        head_dim: moe.head_dim,
        max_position_embeddings: moe.max_position_embeddings,
        linear_conv_kernel_dim: moe.linear_conv_kernel_dim,
        linear_key_head_dim: moe.linear_key_head_dim,
        linear_value_head_dim: moe.linear_value_head_dim,
        linear_num_key_heads: moe.linear_num_key_heads,
        linear_num_value_heads: moe.linear_num_value_heads,
        rms_norm_eps: moe.rms_norm_eps,
        rope_theta: moe.rope_theta,
        partial_rotary_factor: moe.partial_rotary_factor,
        activation_id: moe.activation_id,
        tie_word_embeddings: moe.tie_word_embeddings,
        eos_token_ids: moe.eos_token_ids.clone(),
        full_attention_layers: moe.full_attention_layers.clone(),
    }
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
    let weight_prefix = read_string_advance(buf, &mut offset, "FLM tensor ABI weight_prefix")?;
    let int4_packed_suffix =
        read_string_advance(buf, &mut offset, "FLM tensor ABI int4_packed_suffix")?;
    let int4_scale_suffix =
        read_string_advance(buf, &mut offset, "FLM tensor ABI int4_scale_suffix")?;
    let int4_shape_suffix =
        read_string_advance(buf, &mut offset, "FLM tensor ABI int4_shape_suffix")?;
    ensure_consumed(buf, offset, "FLM tensor ABI")?;
    Ok(FlmTensorAbiDescriptor {
        abi_id,
        weight_prefix,
        int4_packed_suffix,
        int4_scale_suffix,
        int4_shape_suffix,
    })
}

fn parse_model_descriptor(buf: &[u8]) -> Result<FlmModelDescriptor, Error> {
    if buf.len() != MODEL_DESCRIPTOR_SIZE {
        return Err(Error::Other(format!(
            "FLM model descriptor section has len {}; expected {MODEL_DESCRIPTOR_SIZE}",
            buf.len()
        )));
    }
    let mut offset = 0usize;
    let descriptor = FlmModelDescriptor {
        descriptor_version: read_u16_advance(buf, &mut offset, "FLM model descriptor version")?,
        model_id: read_u16_advance(buf, &mut offset, "FLM model descriptor model_id")?,
        config_section_id: read_u32_advance(
            buf,
            &mut offset,
            "FLM model descriptor config_section_id",
        )?,
        tokenizer_id: read_u32_advance(buf, &mut offset, "FLM model descriptor tokenizer_id")?,
        tensor_abi_id: read_u32_advance(buf, &mut offset, "FLM model descriptor tensor_abi_id")?,
        quant_profile_id: read_u32_advance(
            buf,
            &mut offset,
            "FLM model descriptor quant_profile_id",
        )?,
        flags: read_u32_advance(buf, &mut offset, "FLM model descriptor flags")?,
    };
    if descriptor.descriptor_version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM model descriptor version {}",
            descriptor.descriptor_version
        )));
    }
    Ok(descriptor)
}

fn validate_model_descriptor(
    descriptor: &FlmModelDescriptor,
    architecture_id: u32,
    tokenizer: &FlmTokenizerDescriptor,
    tensor_abi: &FlmTensorAbiDescriptor,
) -> Result<(), Error> {
    let (
        expected_model_id,
        expected_config_section_id,
        expected_tensor_abi_id,
        expected_quant_profile_id,
    ) = match architecture_id {
        ARCH_QWEN3_6_DENSE => (
            MODEL_QWEN3_6_DENSE_V1,
            SECTION_CONFIG_QWEN36_DENSE,
            TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1,
            QUANT_PROFILE_QWEN3_6_DENSE_CT_INT4_G128_BF16_V1,
        ),
        ARCH_QWEN3_6_MOE => (
            MODEL_QWEN3_6_MOE_V1,
            SECTION_CONFIG_QWEN36_MOE,
            TENSOR_ABI_QWEN3_6_MOE_MIXED_LOWBIT_V1,
            QUANT_PROFILE_QWEN3_6_MOE_MIXED_LOWBIT_V1,
        ),
        other => {
            return Err(Error::Other(format!(
                "unsupported FLM runtime architecture {other}"
            )));
        }
    };
    if descriptor.config_section_id != expected_config_section_id {
        return Err(Error::Other(format!(
            "FLM model descriptor references unsupported config section {}",
            descriptor.config_section_id
        )));
    }
    if descriptor.tokenizer_id != tokenizer.tokenizer_id {
        return Err(Error::Other(format!(
            "FLM model descriptor tokenizer_id {} does not match tokenizer {}",
            descriptor.tokenizer_id, tokenizer.tokenizer_id
        )));
    }
    if descriptor.tensor_abi_id != tensor_abi.abi_id {
        return Err(Error::Other(format!(
            "FLM model descriptor tensor_abi_id {} does not match tensor ABI {}",
            descriptor.tensor_abi_id, tensor_abi.abi_id
        )));
    }
    if descriptor.tensor_abi_id != expected_tensor_abi_id {
        return Err(Error::Other(format!(
            "FLM model descriptor tensor_abi_id {} does not match architecture {}",
            descriptor.tensor_abi_id, architecture_id
        )));
    }
    if descriptor.model_id != expected_model_id {
        return Err(Error::Other(format!(
            "unsupported FLM model id {}",
            descriptor.model_id
        )));
    }
    if descriptor.quant_profile_id != expected_quant_profile_id {
        return Err(Error::Other(format!(
            "unsupported FLM quant profile id {}",
            descriptor.quant_profile_id
        )));
    }
    Ok(())
}

fn parse_assets(table: &[u8], payloads: &[u8]) -> Result<HashMap<u32, FlmAsset>, Error> {
    let mut offset = 0usize;
    let count = read_count(table, &mut offset, "FLM asset count")?;
    let mut assets = HashMap::with_capacity(count);
    let mut payload_ranges = Vec::with_capacity(count);
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
        let kind_id = read_u16_advance(table, &mut offset, "FLM asset kind_id")?;
        let flags = read_u16_advance(table, &mut offset, "FLM asset flags")?;
        let name_len = u32_to_usize(read_u32_advance(table, &mut offset, "FLM asset name_len")?)?;
        let name_bytes = read_exact_range(table, offset, name_len, "FLM asset name")?;
        offset += name_len;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|e| Error::Other(format!("FLM asset {asset_id} name is not UTF-8: {e}")))?
            .to_string();
        let payload_end = payload_offset.checked_add(payload_len).ok_or_else(|| {
            Error::Other(format!(
                "FLM asset {asset_id} payload range overflows (offset={payload_offset}, len={payload_len})"
            ))
        })?;
        let payload = read_exact_range(
            payloads,
            payload_offset,
            payload_len,
            &format!("FLM asset {asset_id} payload"),
        )?
        .to_vec();
        payload_ranges.push((payload_offset, payload_end, asset_id));
        if assets
            .insert(
                asset_id,
                FlmAsset {
                    asset_id,
                    kind_id,
                    flags,
                    name,
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
    payload_ranges.sort_by_key(|(payload_offset, _, asset_id)| (*payload_offset, *asset_id));
    let mut previous_end = 0usize;
    for (payload_offset, payload_end, asset_id) in payload_ranges {
        if payload_offset < previous_end {
            return Err(Error::Other(format!(
                "FLM asset {asset_id} payload overlap (offset={payload_offset}, previous_end={previous_end})"
            )));
        }
        if payload_offset != previous_end {
            return Err(Error::Other(format!(
                "FLM asset {asset_id} payload gap (offset={payload_offset}, expected={previous_end})"
            )));
        }
        previous_end = payload_end;
    }
    if previous_end != payloads.len() {
        return Err(Error::Other(format!(
            "FLM assets have trailing payload bytes (final_end={previous_end}, len={})",
            payloads.len()
        )));
    }
    Ok(assets)
}

fn parse_tensor_manifest(buf: &[u8]) -> Result<FlmTensorManifest, Error> {
    read_exact_range(
        buf,
        0,
        TENSOR_MANIFEST_HEADER_SIZE,
        "FLM tensor manifest header",
    )?;
    let version = read_u16(buf, 0, "FLM tensor manifest version")?;
    if version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM tensor manifest version {version}"
        )));
    }
    let row_stride = read_u16(buf, 2, "FLM tensor manifest row_stride")? as usize;
    if row_stride != TENSOR_MANIFEST_ROW_SIZE {
        return Err(Error::Other(format!(
            "FLM tensor manifest row stride {row_stride}; expected {TENSOR_MANIFEST_ROW_SIZE}"
        )));
    }
    let row_count = u32_to_usize(read_u32(buf, 4, "FLM tensor manifest row_count")?)?;
    let string_pool_len = u32_to_usize(read_u32(buf, 8, "FLM tensor manifest string_pool_len")?)?;
    let rows_len = row_count.checked_mul(row_stride).ok_or_else(|| {
        Error::Other("FLM tensor manifest row table length overflows".to_string())
    })?;
    let string_pool_offset = TENSOR_MANIFEST_HEADER_SIZE
        .checked_add(rows_len)
        .ok_or_else(|| {
            Error::Other("FLM tensor manifest string pool offset overflows".to_string())
        })?;
    let expected_len = string_pool_offset
        .checked_add(string_pool_len)
        .ok_or_else(|| Error::Other("FLM tensor manifest length overflows".to_string()))?;
    if buf.len() != expected_len {
        return Err(Error::Other(format!(
            "FLM tensor manifest has len {}; expected {expected_len}",
            buf.len()
        )));
    }
    let string_pool = read_exact_range(
        buf,
        string_pool_offset,
        string_pool_len,
        "FLM tensor manifest string pool",
    )?;

    let mut rows = Vec::with_capacity(row_count);
    let mut required_names = HashSet::new();
    for idx in 0..row_count {
        let row_offset = TENSOR_MANIFEST_HEADER_SIZE + idx * row_stride;
        let row = read_exact_range(buf, row_offset, row_stride, "FLM tensor manifest row")?;
        let mut offset = 0usize;
        let role_id = read_u32_advance(row, &mut offset, "FLM tensor manifest role_id")?;
        let group_id = read_u32_advance(row, &mut offset, "FLM tensor manifest group_id")?;
        let companion_kind =
            read_u8_advance(row, &mut offset, "FLM tensor manifest companion_kind")?;
        let rank = read_u8_advance(row, &mut offset, "FLM tensor manifest rank")?;
        if rank > 4 {
            return Err(Error::Other(format!(
                "FLM tensor manifest row {idx} rank {rank} exceeds 4"
            )));
        }
        let dtype = read_u16_advance(row, &mut offset, "FLM tensor manifest dtype")?;
        let logical_dtype =
            read_u16_advance(row, &mut offset, "FLM tensor manifest logical_dtype")?;
        let codec_id = read_u8_advance(row, &mut offset, "FLM tensor manifest codec_id")?;
        let flags = read_u8_advance(row, &mut offset, "FLM tensor manifest flags")?;
        let mut shape = [0u32; 4];
        for (dim_idx, dim) in shape.iter_mut().enumerate() {
            *dim = read_u32_advance(
                row,
                &mut offset,
                &format!("FLM tensor manifest shape[{dim_idx}]"),
            )?;
        }
        let name_offset = u32_to_usize(read_u32_advance(
            row,
            &mut offset,
            "FLM tensor manifest name_offset",
        )?)?;
        let name_len = read_u16_advance(row, &mut offset, "FLM tensor manifest name_len")? as usize;
        let reserved = read_u16_advance(row, &mut offset, "FLM tensor manifest reserved")?;
        if reserved != 0 {
            return Err(Error::Other(format!(
                "FLM tensor manifest row {idx} reserved field is nonzero ({reserved})"
            )));
        }
        let name_end = name_offset.checked_add(name_len).ok_or_else(|| {
            Error::Other(format!(
                "FLM tensor manifest string range overflows for row {idx}"
            ))
        })?;
        let name_bytes = read_exact_range(
            string_pool,
            name_offset,
            name_len,
            "FLM tensor manifest string",
        )
        .map_err(|e| {
            Error::Other(format!(
                "FLM tensor manifest string bounds invalid for row {idx}: {e}"
            ))
        })?;
        if name_end > string_pool_len {
            return Err(Error::Other(format!(
                "FLM tensor manifest string bounds invalid for row {idx}"
            )));
        }
        let name = std::str::from_utf8(name_bytes)
            .map_err(|e| {
                Error::Other(format!(
                    "FLM tensor manifest string for row {idx} is not UTF-8: {e}"
                ))
            })?
            .to_string();
        if flags & MANIFEST_FLAG_REQUIRED != 0 && !required_names.insert(name.clone()) {
            return Err(Error::Other(format!(
                "FLM tensor manifest has duplicate required name {name}"
            )));
        }
        rows.push(FlmTensorManifestRow {
            role_id,
            group_id,
            companion_kind,
            rank,
            dtype,
            logical_dtype,
            codec_id,
            flags,
            shape,
            name,
        });
    }
    Ok(FlmTensorManifest { rows })
}

fn parse_storage_abis(buf: &[u8]) -> Result<Vec<FlmStorageAbi>, Error> {
    read_exact_range(buf, 0, STORAGE_ABI_HEADER_SIZE, "FLM storage ABI header")?;
    let version = read_u16(buf, 0, "FLM storage ABI version")?;
    if version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM storage ABI table version {version}"
        )));
    }
    let row_stride = read_u16(buf, 2, "FLM storage ABI row_stride")? as usize;
    if row_stride != STORAGE_ABI_ROW_SIZE {
        return Err(Error::Other(format!(
            "FLM storage ABI row stride {row_stride}; expected {STORAGE_ABI_ROW_SIZE}"
        )));
    }
    let row_count = u32_to_usize(read_u32(buf, 4, "FLM storage ABI row_count")?)?;
    let params_len = u32_to_usize(read_u32(buf, 8, "FLM storage ABI params_len")?)?;
    let rows_len = row_count
        .checked_mul(row_stride)
        .ok_or_else(|| Error::Other("FLM storage ABI row table length overflows".to_string()))?;
    let params_offset = STORAGE_ABI_HEADER_SIZE
        .checked_add(rows_len)
        .ok_or_else(|| Error::Other("FLM storage ABI params offset overflows".to_string()))?;
    let expected_len = params_offset
        .checked_add(params_len)
        .ok_or_else(|| Error::Other("FLM storage ABI length overflows".to_string()))?;
    if buf.len() != expected_len {
        return Err(Error::Other(format!(
            "FLM storage ABI table has len {}; expected {expected_len}",
            buf.len()
        )));
    }
    let params = read_exact_range(buf, params_offset, params_len, "FLM storage ABI params")?;

    let mut rows = Vec::with_capacity(row_count);
    let mut seen_ids = HashSet::new();
    for idx in 0..row_count {
        let row_offset = STORAGE_ABI_HEADER_SIZE + idx * row_stride;
        let row = read_exact_range(buf, row_offset, row_stride, "FLM storage ABI row")?;
        let mut offset = 0usize;
        let storage_abi_id = read_u16_advance(row, &mut offset, "FLM storage ABI id")?;
        let abi_kind_id = read_u16_advance(row, &mut offset, "FLM storage ABI kind")?;
        let codec_semantic_id =
            read_u16_advance(row, &mut offset, "FLM storage ABI codec semantic id")?;
        let layout_id = read_u16_advance(row, &mut offset, "FLM storage ABI layout id")?;
        let bits = read_u8_advance(row, &mut offset, "FLM storage ABI bits")?;
        let group_size = read_u16_advance(row, &mut offset, "FLM storage ABI group size")?;
        let quant_flags = read_u16_advance(row, &mut offset, "FLM storage ABI quant flags")?;
        let param_offset = u32_to_usize(read_u32_advance(
            row,
            &mut offset,
            "FLM storage ABI param offset",
        )?)?;
        let param_len = u32_to_usize(read_u32_advance(
            row,
            &mut offset,
            "FLM storage ABI param len",
        )?)?;
        ensure_consumed(row, offset, "FLM storage ABI row")?;
        if !seen_ids.insert(storage_abi_id) {
            return Err(Error::Other(format!(
                "FLM storage ABI table has duplicate id {storage_abi_id}"
            )));
        }
        let param_bytes = read_exact_range(
            params,
            param_offset,
            param_len,
            "FLM storage ABI params range",
        )?
        .to_vec();
        rows.push(FlmStorageAbi {
            storage_abi_id,
            abi_kind_id,
            codec_semantic_id,
            layout_id,
            bits,
            group_size,
            quant_flags,
            params: param_bytes,
        });
    }
    Ok(rows)
}

fn parse_logical_tensors(buf: &[u8]) -> Result<Vec<FlmLogicalTensor>, Error> {
    read_exact_range(
        buf,
        0,
        LOGICAL_TENSOR_HEADER_SIZE,
        "FLM logical tensor header",
    )?;
    let version = read_u16(buf, 0, "FLM logical tensor version")?;
    if version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM logical tensor table version {version}"
        )));
    }
    let row_stride = read_u16(buf, 2, "FLM logical tensor row_stride")? as usize;
    if row_stride != LOGICAL_TENSOR_ROW_SIZE {
        return Err(Error::Other(format!(
            "FLM logical tensor row stride {row_stride}; expected {LOGICAL_TENSOR_ROW_SIZE}"
        )));
    }
    let row_count = u32_to_usize(read_u32(buf, 4, "FLM logical tensor row_count")?)?;
    let string_pool_len = u32_to_usize(read_u32(buf, 8, "FLM logical tensor string_pool_len")?)?;
    let rows_len = row_count
        .checked_mul(row_stride)
        .ok_or_else(|| Error::Other("FLM logical tensor row table length overflows".to_string()))?;
    let string_pool_offset = LOGICAL_TENSOR_HEADER_SIZE
        .checked_add(rows_len)
        .ok_or_else(|| {
            Error::Other("FLM logical tensor string pool offset overflows".to_string())
        })?;
    let expected_len = string_pool_offset
        .checked_add(string_pool_len)
        .ok_or_else(|| Error::Other("FLM logical tensor length overflows".to_string()))?;
    if buf.len() != expected_len {
        return Err(Error::Other(format!(
            "FLM logical tensor table has len {}; expected {expected_len}",
            buf.len()
        )));
    }
    let string_pool = read_exact_range(
        buf,
        string_pool_offset,
        string_pool_len,
        "FLM logical tensor string pool",
    )?;

    let mut rows = Vec::with_capacity(row_count);
    let mut seen_ids = HashSet::new();
    for idx in 0..row_count {
        let row_offset = LOGICAL_TENSOR_HEADER_SIZE + idx * row_stride;
        let row = read_exact_range(buf, row_offset, row_stride, "FLM logical tensor row")?;
        let mut offset = 0usize;
        let tensor_id = read_u32_advance(row, &mut offset, "FLM logical tensor id")?;
        let name_offset = u32_to_usize(read_u32_advance(
            row,
            &mut offset,
            "FLM logical tensor name offset",
        )?)?;
        let name_len = read_u16_advance(row, &mut offset, "FLM logical tensor name len")? as usize;
        let role_id = read_u16_advance(row, &mut offset, "FLM logical tensor role id")?;
        let rank = read_u8_advance(row, &mut offset, "FLM logical tensor rank")?;
        let reserved0 = read_u8_advance(row, &mut offset, "FLM logical tensor reserved0")?;
        if rank > 4 {
            return Err(Error::Other(format!(
                "FLM logical tensor row {idx} rank {rank} exceeds 4"
            )));
        }
        if reserved0 != 0 {
            return Err(Error::Other(format!(
                "FLM logical tensor row {idx} reserved0 is nonzero ({reserved0})"
            )));
        }
        let mut shape = [0u32; 4];
        for (dim_idx, dim) in shape.iter_mut().enumerate() {
            *dim = read_u32_advance(
                row,
                &mut offset,
                &format!("FLM logical tensor shape[{dim_idx}]"),
            )?;
        }
        if shape[rank as usize..].iter().any(|dim| *dim != 0) {
            return Err(Error::Other(format!(
                "FLM logical tensor row {idx} has nonzero shape beyond rank {rank}"
            )));
        }
        let value_format_id =
            read_u16_advance(row, &mut offset, "FLM logical tensor value format")?;
        let reconstruction_dtype =
            read_u16_advance(row, &mut offset, "FLM logical tensor reconstruction dtype")?;
        let storage_binding_start =
            read_u32_advance(row, &mut offset, "FLM logical tensor storage binding start")?;
        let storage_binding_count =
            read_u16_advance(row, &mut offset, "FLM logical tensor storage binding count")?;
        let flags = read_u16_advance(row, &mut offset, "FLM logical tensor flags")?;
        let reserved1 = read_u16_advance(row, &mut offset, "FLM logical tensor reserved1")?;
        ensure_consumed(row, offset, "FLM logical tensor row")?;
        if reserved1 != 0 {
            return Err(Error::Other(format!(
                "FLM logical tensor row {idx} reserved1 is nonzero ({reserved1})"
            )));
        }
        if !seen_ids.insert(tensor_id) {
            return Err(Error::Other(format!(
                "FLM logical tensor table has duplicate id {tensor_id}"
            )));
        }
        let name_bytes = read_exact_range(
            string_pool,
            name_offset,
            name_len,
            "FLM logical tensor string",
        )?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|e| {
                Error::Other(format!(
                    "FLM logical tensor string for row {idx} is not UTF-8: {e}"
                ))
            })?
            .to_string();
        rows.push(FlmLogicalTensor {
            tensor_id,
            name,
            role_id,
            rank,
            shape,
            value_format_id,
            reconstruction_dtype,
            storage_binding_start,
            storage_binding_count,
            flags,
        });
    }
    Ok(rows)
}

fn parse_storage_bindings(buf: &[u8]) -> Result<Vec<FlmStorageBinding>, Error> {
    read_exact_range(
        buf,
        0,
        STORAGE_BINDING_HEADER_SIZE,
        "FLM storage binding header",
    )?;
    let version = read_u16(buf, 0, "FLM storage binding version")?;
    if version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM storage binding table version {version}"
        )));
    }
    let row_stride = read_u16(buf, 2, "FLM storage binding row_stride")? as usize;
    if row_stride != STORAGE_BINDING_ROW_SIZE {
        return Err(Error::Other(format!(
            "FLM storage binding row stride {row_stride}; expected {STORAGE_BINDING_ROW_SIZE}"
        )));
    }
    let row_count = u32_to_usize(read_u32(buf, 4, "FLM storage binding row_count")?)?;
    let string_pool_len = u32_to_usize(read_u32(buf, 8, "FLM storage binding string_pool_len")?)?;
    let rows_len = row_count.checked_mul(row_stride).ok_or_else(|| {
        Error::Other("FLM storage binding row table length overflows".to_string())
    })?;
    let string_pool_offset = STORAGE_BINDING_HEADER_SIZE
        .checked_add(rows_len)
        .ok_or_else(|| {
            Error::Other("FLM storage binding string pool offset overflows".to_string())
        })?;
    let expected_len = string_pool_offset
        .checked_add(string_pool_len)
        .ok_or_else(|| Error::Other("FLM storage binding length overflows".to_string()))?;
    if buf.len() != expected_len {
        return Err(Error::Other(format!(
            "FLM storage binding table has len {}; expected {expected_len}",
            buf.len()
        )));
    }
    let string_pool = read_exact_range(
        buf,
        string_pool_offset,
        string_pool_len,
        "FLM storage binding string pool",
    )?;

    let mut rows = Vec::with_capacity(row_count);
    for idx in 0..row_count {
        let row_offset = STORAGE_BINDING_HEADER_SIZE + idx * row_stride;
        let row = read_exact_range(buf, row_offset, row_stride, "FLM storage binding row")?;
        let mut offset = 0usize;
        let logical_tensor_id =
            read_u32_advance(row, &mut offset, "FLM storage binding logical tensor id")?;
        let name_offset = u32_to_usize(read_u32_advance(
            row,
            &mut offset,
            "FLM storage binding name offset",
        )?)?;
        let name_len = read_u16_advance(row, &mut offset, "FLM storage binding name len")? as usize;
        let storage_role = read_u16_advance(row, &mut offset, "FLM storage binding role")?;
        let storage_dtype = read_u16_advance(row, &mut offset, "FLM storage binding dtype")?;
        let storage_abi_id = read_u16_advance(row, &mut offset, "FLM storage binding ABI id")?;
        let flags = read_u16_advance(row, &mut offset, "FLM storage binding flags")?;
        let reserved = read_u16_advance(row, &mut offset, "FLM storage binding reserved")?;
        ensure_consumed(row, offset, "FLM storage binding row")?;
        if reserved != 0 {
            return Err(Error::Other(format!(
                "FLM storage binding row {idx} reserved field is nonzero ({reserved})"
            )));
        }
        let name_bytes = read_exact_range(
            string_pool,
            name_offset,
            name_len,
            "FLM storage binding string",
        )?;
        let tensor_name = std::str::from_utf8(name_bytes)
            .map_err(|e| {
                Error::Other(format!(
                    "FLM storage binding string for row {idx} is not UTF-8: {e}"
                ))
            })?
            .to_string();
        rows.push(FlmStorageBinding {
            logical_tensor_id,
            storage_role,
            tensor_name,
            storage_dtype,
            storage_abi_id,
            flags,
        });
    }
    Ok(rows)
}

fn parse_plan_steps(buf: &[u8]) -> Result<Vec<FlmPlanStep>, Error> {
    read_exact_range(buf, 0, PLAN_STEP_HEADER_SIZE, "FLM plan step header")?;
    let version = read_u16(buf, 0, "FLM plan step version")?;
    if version != 1 {
        return Err(Error::Other(format!(
            "unsupported FLM plan step table version {version}"
        )));
    }
    let row_stride = read_u16(buf, 2, "FLM plan step row_stride")? as usize;
    if row_stride != PLAN_STEP_ROW_SIZE {
        return Err(Error::Other(format!(
            "FLM plan step row stride {row_stride}; expected {PLAN_STEP_ROW_SIZE}"
        )));
    }
    let row_count = u32_to_usize(read_u32(buf, 4, "FLM plan step row_count")?)?;
    let rows_len = row_count
        .checked_mul(row_stride)
        .ok_or_else(|| Error::Other("FLM plan step row table length overflows".to_string()))?;
    let expected_len = PLAN_STEP_HEADER_SIZE
        .checked_add(rows_len)
        .ok_or_else(|| Error::Other("FLM plan step length overflows".to_string()))?;
    if buf.len() != expected_len {
        return Err(Error::Other(format!(
            "FLM plan step table has len {}; expected {expected_len}",
            buf.len()
        )));
    }

    let mut rows = Vec::with_capacity(row_count);
    for idx in 0..row_count {
        let row_offset = PLAN_STEP_HEADER_SIZE + idx * row_stride;
        let row = read_exact_range(buf, row_offset, row_stride, "FLM plan step row")?;
        let mut offset = 0usize;
        let logical_tensor_id = read_u32_advance(row, &mut offset, "FLM plan step logical id")?;
        let storage_role = read_u16_advance(row, &mut offset, "FLM plan step storage role")?;
        let consume_strategy =
            read_u16_advance(row, &mut offset, "FLM plan step consume strategy")?;
        let target_layout_id = read_u16_advance(row, &mut offset, "FLM plan step target layout")?;
        let target_dtype = read_u16_advance(row, &mut offset, "FLM plan step target dtype")?;
        let target_rank = read_u8_advance(row, &mut offset, "FLM plan step target rank")?;
        let reserved0 = read_u8_advance(row, &mut offset, "FLM plan step reserved0")?;
        if reserved0 != 0 {
            return Err(Error::Other(format!(
                "FLM plan step row {idx} reserved0={reserved0}; expected 0"
            )));
        }
        if target_rank > 4 {
            return Err(Error::Other(format!(
                "FLM plan step row {idx} target rank {target_rank} exceeds 4"
            )));
        }
        let mut target_shape = [0u32; 4];
        for dim in &mut target_shape {
            *dim = read_u32_advance(row, &mut offset, "FLM plan step target shape")?;
        }
        for (dim_idx, dim) in target_shape.iter().enumerate().skip(target_rank as usize) {
            if *dim != 0 {
                return Err(Error::Other(format!(
                    "FLM plan step row {idx} target_shape[{dim_idx}]={dim}; expected 0 beyond rank"
                )));
            }
        }
        rows.push(FlmPlanStep {
            logical_tensor_id,
            storage_role,
            consume_strategy,
            target_layout_id,
            target_dtype,
            target_rank,
            target_shape,
            stream_id: read_u16_advance(row, &mut offset, "FLM plan step stream id")?,
            priority: read_u16_advance(row, &mut offset, "FLM plan step priority")?,
            flags: read_u32_advance(row, &mut offset, "FLM plan step flags")?,
        });
        ensure_consumed(row, offset, "FLM plan step row")?;
        debug_assert!(idx < row_count);
    }
    Ok(rows)
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

fn read_u8_advance(buf: &[u8], offset: &mut usize, what: &str) -> Result<u8, Error> {
    let value = read_exact_range(buf, *offset, 1, what)?[0];
    *offset += 1;
    Ok(value)
}

fn read_bool_advance(buf: &[u8], offset: &mut usize, what: &str) -> Result<bool, Error> {
    match read_u8_advance(buf, offset, what)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(Error::Other(format!(
            "{what} has invalid bool value {other}"
        ))),
    }
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

    fn read_u16_at(buf: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap())
    }

    fn read_u32_at(buf: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn put_u16_at(buf: &mut [u8], offset: usize, value: u16) {
        buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32_at(buf: &mut [u8], offset: usize, value: u32) {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
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

    fn build_qwen_moe_config_section() -> Vec<u8> {
        let mut out = Vec::new();
        for value in [
            248_320u32, 2048, 512, 512, 40, 16, 2, 256, 262_144, 4, 128, 128, 16, 32, 256, 8, 1,
        ] {
            write_u32(&mut out, value);
        }
        write_f64(&mut out, 1e-6);
        write_f64(&mut out, 10_000_000.0);
        out.push(1);
        out.push(0);
        out.push(1);
        out.push(0);
        out.push(1);
        write_u32(&mut out, 1);
        write_f64(&mut out, 0.25);
        write_u32(&mut out, 248_044);
        write_u32(&mut out, 10);
        for layer in [3u32, 7, 11, 15, 19, 23, 27, 31, 35, 39] {
            write_u32(&mut out, layer);
        }
        for value in [11u32, 11, 10] {
            write_u32(&mut out, value);
        }
        out
    }

    fn build_tokenizer_section() -> Vec<u8> {
        let mut out = Vec::new();
        for value in [
            TOKENIZER_QWEN_BPE_V1,
            TOKENIZER_QWEN_BPE_V1,
            151_936,
            1,
            2,
            3,
            4,
            0,
        ] {
            write_u32(&mut out, value);
        }
        out
    }

    fn build_codec_table_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, 3);
        for (codec_id, semantic_id, layout_id, decoder_id, flags) in [
            (0u8, CODEC_RAW_BF16 as u8, 0u16, 0u16, 0u32),
            (1u8, CODEC_SYM_INT4_G128_BF16 as u8, 0u16, 1u16, 0u32),
            (2u8, CODEC_RAW_I64 as u8, 0u16, 0u16, 0u32),
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
        write_string(&mut out, "model.language_model");
        write_string(&mut out, ".weight_packed");
        write_string(&mut out, ".weight_scale");
        write_string(&mut out, ".weight_shape");
        out
    }

    fn build_moe_tensor_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u32(&mut out, TENSOR_ABI_QWEN3_6_MOE_MIXED_LOWBIT_V1);
        write_string(&mut out, "model.language_model");
        write_string(&mut out, ".weight_packed");
        write_string(&mut out, ".weight_scale");
        write_string(&mut out, ".weight_shape");
        out
    }

    fn build_asset_sections() -> (Vec<u8>, Vec<u8>) {
        let assets = [
            (
                1u32,
                ASSET_TOKENIZER_VOCAB,
                ASSET_FLAG_REQUIRED_FOR_RUNTIME,
                "tokenizer_vocab",
                br#"{"hello":0}"#.as_slice(),
            ),
            (
                2u32,
                ASSET_TOKENIZER_MERGES,
                ASSET_FLAG_REQUIRED_FOR_RUNTIME,
                "tokenizer_merges",
                b"#version: 0.2\n".as_slice(),
            ),
            (
                3u32,
                ASSET_TOKENIZER_ADDED_TOKENS,
                0,
                "tokenizer_added_tokens",
                b"[]".as_slice(),
            ),
            (
                4u32,
                ASSET_TOKENIZER_REGEX,
                ASSET_FLAG_REQUIRED_FOR_RUNTIME,
                "tokenizer_regex",
                br#"\p{L}+"#.as_slice(),
            ),
        ];
        let mut table = Vec::new();
        let mut payloads = Vec::new();
        write_u32(&mut table, assets.len() as u32);
        for (asset_id, kind_id, flags, name, payload) in assets {
            write_u32(&mut table, asset_id);
            write_u32(&mut table, payloads.len() as u32);
            write_u32(&mut table, payload.len() as u32);
            write_u16(&mut table, kind_id);
            write_u16(&mut table, flags);
            write_u32(&mut table, name.len() as u32);
            table.extend_from_slice(name.as_bytes());
            payloads.extend_from_slice(payload);
        }
        (table, payloads)
    }

    fn build_model_descriptor_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, MODEL_QWEN3_6_DENSE_V1);
        write_u32(&mut out, SECTION_CONFIG_QWEN36_DENSE);
        write_u32(&mut out, TOKENIZER_QWEN_BPE_V1);
        write_u32(&mut out, TENSOR_ABI_QWEN3_6_DENSE_CT_INT4_V1);
        write_u32(&mut out, QUANT_PROFILE_QWEN3_6_DENSE_CT_INT4_G128_BF16_V1);
        write_u32(&mut out, 0);
        out
    }

    fn build_moe_model_descriptor_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, MODEL_QWEN3_6_MOE_V1);
        write_u32(&mut out, SECTION_CONFIG_QWEN36_MOE);
        write_u32(&mut out, TOKENIZER_QWEN_BPE_V1);
        write_u32(&mut out, TENSOR_ABI_QWEN3_6_MOE_MIXED_LOWBIT_V1);
        write_u32(&mut out, QUANT_PROFILE_QWEN3_6_MOE_MIXED_LOWBIT_V1);
        write_u32(&mut out, 0);
        out
    }

    fn write_manifest_row(
        out: &mut Vec<u8>,
        role_id: u32,
        group_id: u32,
        companion_kind: u8,
        rank: u8,
        dtype: u16,
        logical_dtype: u16,
        codec_id: u8,
        flags: u8,
        shape: [u32; 4],
        name_offset: u32,
        name_len: u16,
    ) {
        write_u32(out, role_id);
        write_u32(out, group_id);
        out.push(companion_kind);
        out.push(rank);
        write_u16(out, dtype);
        write_u16(out, logical_dtype);
        out.push(codec_id);
        out.push(flags);
        for dim in shape {
            write_u32(out, dim);
        }
        write_u32(out, name_offset);
        write_u16(out, name_len);
        write_u16(out, 0);
    }

    fn build_tensor_manifest_section() -> Vec<u8> {
        let names = b"model.language_model.layers.0.mlp.gate_proj.weight_packed";
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, TENSOR_MANIFEST_ROW_SIZE as u16);
        write_u32(&mut out, 1);
        write_u32(&mut out, names.len() as u32);
        write_manifest_row(
            &mut out,
            1,
            0,
            MANIFEST_COMPANION_PACKED,
            2,
            CODEC_SYM_INT4_G128_BF16,
            CODEC_SYM_INT4_G128_BF16,
            1,
            MANIFEST_FLAG_REQUIRED,
            [5120, 27648, 0, 0],
            0,
            names.len() as u16,
        );
        out.extend_from_slice(names);
        out
    }

    fn build_stage3_storage_abi_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, STORAGE_ABI_ROW_SIZE as u16);
        write_u32(&mut out, 1);
        write_u32(&mut out, 0);
        write_u16(&mut out, 1);
        write_u16(&mut out, STORAGE_ABI_KIND_GROUP_QUANT);
        write_u16(&mut out, CODEC_SYM_INT4_G128_BF16);
        write_u16(&mut out, LAYOUT_ID_DEFAULT);
        out.push(4);
        write_u16(&mut out, 128);
        write_u16(&mut out, QUANT_FLAG_SYMMETRIC);
        write_u32(&mut out, 0);
        write_u32(&mut out, 0);
        out
    }

    fn build_stage3_logical_tensor_section() -> Vec<u8> {
        let name = b"model.language_model.layers.0.mlp.gate_proj.weight";
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, LOGICAL_TENSOR_ROW_SIZE as u16);
        write_u32(&mut out, 1);
        write_u32(&mut out, name.len() as u32);
        write_u32(&mut out, 1);
        write_u32(&mut out, 0);
        write_u16(&mut out, name.len() as u16);
        write_u16(&mut out, LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT);
        out.push(2);
        out.push(0);
        for dim in [128u32, 64, 0, 0] {
            write_u32(&mut out, dim);
        }
        write_u16(&mut out, VALUE_FORMAT_SYM_INT4);
        write_u16(&mut out, FLM_DTYPE_BF16);
        write_u32(&mut out, 0);
        write_u16(&mut out, 3);
        write_u16(&mut out, LOGICAL_TENSOR_FLAG_REQUIRED);
        write_u16(&mut out, 0);
        out.extend_from_slice(name);
        out
    }

    fn build_stage3_storage_binding_section() -> Vec<u8> {
        let names: [(&[u8], u16, u16); 3] = [
            (
                b"storage/l0_gate_packed",
                STORAGE_ROLE_PACKED,
                FLM_DTYPE_INT32,
            ),
            (b"storage/l0_gate_scale", STORAGE_ROLE_SCALE, FLM_DTYPE_BF16),
            (
                b"storage/l0_gate_shape",
                STORAGE_ROLE_SHAPE,
                FLM_DTYPE_INT64,
            ),
        ];
        let pool_len: usize = names.iter().map(|(name, _, _)| name.len()).sum();
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, STORAGE_BINDING_ROW_SIZE as u16);
        write_u32(&mut out, names.len() as u32);
        write_u32(&mut out, pool_len as u32);
        let mut pool_offset = 0u32;
        let mut pool = Vec::new();
        for (name, role, dtype) in names {
            write_u32(&mut out, 1);
            write_u32(&mut out, pool_offset);
            write_u16(&mut out, name.len() as u16);
            write_u16(&mut out, role);
            write_u16(&mut out, dtype);
            write_u16(&mut out, 1);
            write_u16(&mut out, STORAGE_BINDING_FLAG_REQUIRED);
            write_u16(&mut out, 0);
            pool.extend_from_slice(name);
            pool_offset += name.len() as u32;
        }
        out.extend_from_slice(&pool);
        out
    }

    fn build_stage3_plan_step_section() -> Vec<u8> {
        let mut out = Vec::new();
        write_u16(&mut out, 1);
        write_u16(&mut out, PLAN_STEP_ROW_SIZE as u16);
        write_u32(&mut out, 2);

        write_u32(&mut out, 1);
        write_u16(&mut out, STORAGE_ROLE_PACKED);
        write_u16(&mut out, CONSUME_STRATEGY_DIRECT);
        write_u16(&mut out, LAYOUT_ID_DEFAULT);
        write_u16(&mut out, FLM_DTYPE_UINT8);
        out.push(2);
        out.push(0);
        for dim in [128u32, 32, 0, 0] {
            write_u32(&mut out, dim);
        }
        write_u16(&mut out, PLAN_STREAM_DEFAULT);
        write_u16(&mut out, PLAN_PRIORITY_DEFAULT);
        write_u32(&mut out, PLAN_STEP_FLAG_NONE);

        write_u32(&mut out, 1);
        write_u16(&mut out, STORAGE_ROLE_SCALE);
        write_u16(&mut out, CONSUME_STRATEGY_DIRECT);
        write_u16(&mut out, LAYOUT_ID_DEFAULT);
        write_u16(&mut out, FLM_DTYPE_BF16);
        out.push(2);
        out.push(0);
        for dim in [128u32, 1, 0, 0] {
            write_u32(&mut out, dim);
        }
        write_u16(&mut out, PLAN_STREAM_DEFAULT);
        write_u16(&mut out, PLAN_PRIORITY_DEFAULT);
        write_u32(&mut out, PLAN_STEP_FLAG_NONE);
        out
    }

    fn build_test_runtime_directory_with_stage3_tables() -> Vec<u8> {
        let (asset_table, asset_payloads) = build_asset_sections();
        let sections = [
            (1u32, build_qwen_config_section()),
            (2u32, build_tokenizer_section()),
            (3u32, build_codec_table_section()),
            (4u32, build_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, build_model_descriptor_section()),
            (8u32, build_tensor_manifest_section()),
            (9u32, build_stage3_storage_abi_section()),
            (10u32, build_stage3_logical_tensor_section()),
            (11u32, build_stage3_storage_binding_section()),
            (12u32, build_stage3_plan_step_section()),
        ];
        let header_len = 8 + 2 + 2 + 4 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        write_u16(&mut out, RUNTIME_VERSION);
        write_u16(&mut out, sections.len() as u16);
        write_u32(&mut out, ARCH_QWEN3_6_DENSE);
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

    fn build_test_runtime_directory() -> Vec<u8> {
        let (asset_table, asset_payloads) = build_asset_sections();
        let sections = [
            (1u32, build_qwen_config_section()),
            (2u32, build_tokenizer_section()),
            (3u32, build_codec_table_section()),
            (4u32, build_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, build_model_descriptor_section()),
            (8u32, build_tensor_manifest_section()),
        ];
        let header_len = 8 + 2 + 2 + 4 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        write_u16(&mut out, RUNTIME_VERSION);
        write_u16(&mut out, sections.len() as u16);
        write_u32(&mut out, ARCH_QWEN3_6_DENSE);
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

    fn build_test_moe_runtime_directory() -> Vec<u8> {
        let (asset_table, asset_payloads) = build_asset_sections();
        let sections = [
            (SECTION_CONFIG_QWEN36_MOE, build_qwen_moe_config_section()),
            (2u32, build_tokenizer_section()),
            (3u32, build_codec_table_section()),
            (4u32, build_moe_tensor_abi_section()),
            (5u32, asset_table),
            (6u32, asset_payloads),
            (7u32, build_moe_model_descriptor_section()),
            (8u32, build_tensor_manifest_section()),
        ];
        let header_len = 8 + 2 + 2 + 4 + sections.len() * 12;
        let mut offset = header_len as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"FLMRUN1\0");
        write_u16(&mut out, RUNTIME_VERSION);
        write_u16(&mut out, sections.len() as u16);
        write_u32(&mut out, ARCH_QWEN3_6_MOE);
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

    fn section_record_offset(runtime: &[u8], section_id: u32) -> usize {
        let count = read_u16_at(runtime, 10) as usize;
        for idx in 0..count {
            let offset = 16 + idx * 12;
            if read_u32_at(runtime, offset) == section_id {
                return offset;
            }
        }
        panic!("missing section {section_id}");
    }

    fn section_range(runtime: &[u8], section_id: u32) -> (usize, usize) {
        let record = section_record_offset(runtime, section_id);
        (
            read_u32_at(runtime, record + 4) as usize,
            read_u32_at(runtime, record + 8) as usize,
        )
    }

    fn insert_gap_before_section(runtime: &mut Vec<u8>, section_id: u32) {
        let (gap_offset, _) = section_range(runtime, section_id);
        runtime.insert(gap_offset, 0);
        let count = read_u16_at(runtime, 10) as usize;
        for idx in 0..count {
            let record = 16 + idx * 12;
            let offset = read_u32_at(runtime, record + 4) as usize;
            if offset >= gap_offset {
                put_u32_at(runtime, record + 4, (offset + 1) as u32);
            }
        }
    }

    fn overlap_section_with_previous(runtime: &mut [u8], section_id: u32) {
        let record = section_record_offset(runtime, section_id);
        let offset = read_u32_at(runtime, record + 4);
        put_u32_at(runtime, record + 4, offset - 1);
    }

    fn shift_sections_after(runtime: &mut [u8], insert_offset: usize, amount: usize) {
        let count = read_u16_at(runtime, 10) as usize;
        for idx in 0..count {
            let record = 16 + idx * 12;
            let offset = read_u32_at(runtime, record + 4) as usize;
            if offset >= insert_offset {
                put_u32_at(runtime, record + 4, (offset + amount) as u32);
            }
        }
    }

    fn asset_payload_record(runtime: &[u8], asset_id: u32) -> (usize, usize, usize) {
        let (table_offset, _) = section_range(runtime, SECTION_ASSET_TABLE);
        let count = read_u32_at(runtime, table_offset) as usize;
        let mut offset = table_offset + 4;
        for _ in 0..count {
            let current_id = read_u32_at(runtime, offset);
            let payload_offset = read_u32_at(runtime, offset + 4) as usize;
            let payload_len = read_u32_at(runtime, offset + 8) as usize;
            let name_len = read_u32_at(runtime, offset + 16) as usize;
            if current_id == asset_id {
                return (offset + 4, payload_offset, payload_len);
            }
            offset += 20 + name_len;
        }
        panic!("missing asset {asset_id}");
    }

    fn insert_asset_payload_gap_before_asset(runtime: &mut Vec<u8>, asset_id: u32) {
        let (payload_section_offset, payload_section_len) =
            section_range(runtime, SECTION_ASSET_PAYLOADS);
        let (_, gap_offset, _) = asset_payload_record(runtime, asset_id);
        let insert_offset = payload_section_offset + gap_offset;
        runtime.insert(insert_offset, 0);
        shift_sections_after(runtime, insert_offset, 1);

        let payload_record = section_record_offset(runtime, SECTION_ASSET_PAYLOADS);
        put_u32_at(
            runtime,
            payload_record + 8,
            (payload_section_len + 1) as u32,
        );

        let (table_offset, _) = section_range(runtime, SECTION_ASSET_TABLE);
        let count = read_u32_at(runtime, table_offset) as usize;
        let mut offset = table_offset + 4;
        for _ in 0..count {
            let payload_offset = read_u32_at(runtime, offset + 4) as usize;
            if payload_offset >= gap_offset {
                put_u32_at(runtime, offset + 4, (payload_offset + 1) as u32);
            }
            let name_len = read_u32_at(runtime, offset + 16) as usize;
            offset += 20 + name_len;
        }
    }

    fn append_asset_payload_trailing_byte(runtime: &mut Vec<u8>) {
        let (payload_section_offset, payload_section_len) =
            section_range(runtime, SECTION_ASSET_PAYLOADS);
        let insert_offset = payload_section_offset + payload_section_len;
        runtime.insert(insert_offset, 0);
        shift_sections_after(runtime, insert_offset, 1);
        let payload_record = section_record_offset(runtime, SECTION_ASSET_PAYLOADS);
        put_u32_at(
            runtime,
            payload_record + 8,
            (payload_section_len + 1) as u32,
        );
    }

    fn expect_parse_error_contains(runtime: &[u8], needle: &str) {
        let err = FlmRuntimeDirectory::parse(runtime).expect_err("runtime should be rejected");
        let text = err.to_string();
        assert!(
            text.contains(needle),
            "expected error to contain {needle:?}, got {text:?}"
        );
    }

    #[test]
    fn parses_runtime_directory_with_qwen_config_and_assets() {
        let runtime = build_test_runtime_directory();
        let parsed = FlmRuntimeDirectory::parse(&runtime).expect("parse runtime");

        assert_eq!(parsed.architecture_id, ARCH_QWEN3_6_DENSE);
        assert_eq!(parsed.qwen36_config().unwrap().hidden_size, 5120);
        assert_eq!(parsed.qwen36_config().unwrap().full_attention_layers[0], 3);
        assert_eq!(
            parsed.tokenizer().unwrap().tokenizer_id,
            TOKENIZER_QWEN_BPE_V1
        );
        assert_eq!(parsed.tokenizer().unwrap().vocab_asset_id, 1);
        assert_eq!(parsed.asset(4).unwrap().name, "tokenizer_regex");
        assert_eq!(parsed.tensor_abi().weight_prefix, "model.language_model");
        assert_eq!(parsed.tensor_abi().int4_packed_suffix, ".weight_packed");
        assert_eq!(parsed.tensor_abi().int4_scale_suffix, ".weight_scale");
        assert_eq!(parsed.tensor_abi().int4_shape_suffix, ".weight_shape");
        assert_eq!(
            parsed.codec_by_id(0).unwrap().semantic_id as u16,
            CODEC_RAW_BF16
        );
        assert_eq!(parsed.codec_by_id(0).unwrap().layout_id, 0);
        assert_eq!(parsed.codec_by_id(0).unwrap().decoder_id, 0);
        assert_eq!(
            parsed.codec_by_id(1).unwrap().semantic_id as u16,
            CODEC_SYM_INT4_G128_BF16
        );
        assert_eq!(parsed.codec_by_id(1).unwrap().layout_id, 0);
        assert_eq!(parsed.codec_by_id(1).unwrap().decoder_id, 1);
        assert_eq!(parsed.codec_by_id(2).unwrap().layout_id, 0);
        assert_eq!(parsed.codec_by_id(2).unwrap().decoder_id, 0);
        assert_eq!(
            parsed.codec_by_semantic_id(CODEC_RAW_I64).unwrap().codec_id,
            2
        );
    }

    #[test]
    fn parses_runtime_directory_with_qwen_moe_config() {
        let runtime = build_test_moe_runtime_directory();
        let parsed = FlmRuntimeDirectory::parse(&runtime).expect("parse moe runtime");

        assert_eq!(parsed.architecture_id, ARCH_QWEN3_6_MOE);
        assert!(parsed.qwen36_config().is_none());
        assert_eq!(parsed.qwen36_moe_config().unwrap().hidden_size, 2048);
        assert_eq!(parsed.qwen36_moe_config().unwrap().num_experts, 256);
        assert_eq!(parsed.qwen36_moe_config().unwrap().num_experts_per_tok, 8);
        assert_eq!(
            parsed.qwen36_moe_config().unwrap().mrope_section,
            [11, 11, 10]
        );
        assert_eq!(parsed.model_descriptor().model_id, MODEL_QWEN3_6_MOE_V1);
    }

    #[test]
    fn parses_stage3_storage_tables() {
        let runtime = build_test_runtime_directory_with_stage3_tables();
        let parsed = FlmRuntimeDirectory::parse(&runtime).expect("parse runtime v4");

        assert_eq!(parsed.storage_abis()[0].storage_abi_id, 1);
        assert_eq!(
            parsed.logical_tensors()[0].name,
            "model.language_model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            parsed.storage_bindings()[0].tensor_name,
            "storage/l0_gate_packed"
        );
        assert_eq!(
            parsed.plan_steps()[0].consume_strategy,
            CONSUME_STRATEGY_DIRECT
        );
        assert_eq!(parsed.plan_steps()[0].storage_role, STORAGE_ROLE_PACKED);
        assert_eq!(parsed.plan_steps()[0].target_dtype, FLM_DTYPE_UINT8);
        assert_eq!(parsed.plan_steps()[0].target_rank, 2);
        assert_eq!(parsed.plan_steps()[0].target_shape, [128, 32, 0, 0]);
        assert_eq!(parsed.plan_steps()[1].storage_role, STORAGE_ROLE_SCALE);
        assert_eq!(parsed.plan_steps()[1].target_dtype, FLM_DTYPE_BF16);
        assert_eq!(parsed.plan_steps()[1].target_shape, [128, 1, 0, 0]);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut runtime = build_test_runtime_directory();
        runtime[0] = b'X';
        expect_parse_error_contains(&runtime, "bad FLM runtime magic");

        let mut runtime = build_test_runtime_directory();
        put_u16_at(&mut runtime, 8, RUNTIME_VERSION + 1);
        expect_parse_error_contains(&runtime, "unsupported FLM runtime version");
    }

    #[test]
    fn parses_runtime_v4_model_descriptor_asset_kinds_and_tensor_manifest() {
        let runtime = build_test_runtime_directory();
        let parsed = FlmRuntimeDirectory::parse(&runtime).expect("parse runtime v4");

        assert_eq!(parsed.model_descriptor().model_id, MODEL_QWEN3_6_DENSE_V1);
        assert_eq!(
            parsed.model_descriptor().quant_profile_id,
            QUANT_PROFILE_QWEN3_6_DENSE_CT_INT4_G128_BF16_V1
        );
        assert_eq!(parsed.asset(1).unwrap().kind_id, ASSET_TOKENIZER_VOCAB);
        assert!(parsed.asset(1).unwrap().flags & ASSET_FLAG_REQUIRED_FOR_RUNTIME != 0);
        assert!(parsed.tensor_manifest().rows.iter().any(|row| {
            row.name == "model.language_model.layers.0.mlp.gate_proj.weight_packed"
                && row.companion_kind == MANIFEST_COMPANION_PACKED
        }));
    }

    #[test]
    fn rejects_runtime_v2_on_normal_parser_path() {
        let mut runtime = build_test_runtime_directory();
        put_u16_at(&mut runtime, 8, 2);
        expect_parse_error_contains(&runtime, "unsupported FLM runtime version");
    }

    #[test]
    fn rejects_manifest_string_bounds() {
        let mut runtime = build_test_runtime_directory();
        let (manifest_offset, _) = section_range(&runtime, SECTION_TENSOR_MANIFEST);
        let row_name_len = manifest_offset + 12 + 4 + 4 + 1 + 1 + 2 + 2 + 1 + 1 + 16 + 4;
        put_u16_at(&mut runtime, row_name_len, u16::MAX);
        expect_parse_error_contains(&runtime, "manifest string");
    }

    #[test]
    fn uses_geo_manifest_row_stride_40_and_rejects_reserved_u16() {
        let runtime = build_test_runtime_directory();
        let (manifest_offset, manifest_len) = section_range(&runtime, SECTION_TENSOR_MANIFEST);
        let string_pool_len = read_u32_at(&runtime, manifest_offset + 8) as usize;
        assert_eq!(read_u16_at(&runtime, manifest_offset + 2), 40);
        assert_eq!(manifest_len, 12 + 40 + string_pool_len);

        let mut runtime = runtime;
        let row_reserved = manifest_offset + 12 + 4 + 4 + 1 + 1 + 2 + 2 + 1 + 1 + 16 + 4 + 2;
        put_u16_at(&mut runtime, row_reserved, 1);
        expect_parse_error_contains(&runtime, "reserved");
    }

    #[test]
    fn rejects_unknown_runtime_architecture_id() {
        let mut runtime = build_test_runtime_directory();
        put_u32_at(&mut runtime, 12, 999);
        expect_parse_error_contains(&runtime, "architecture");
    }

    #[test]
    fn rejects_duplicate_section_ids() {
        let mut runtime = build_test_runtime_directory();
        let record = section_record_offset(&runtime, SECTION_ASSET_PAYLOADS);
        put_u32_at(&mut runtime, record, SECTION_ASSET_TABLE);
        expect_parse_error_contains(&runtime, "duplicate section");
    }

    #[test]
    fn rejects_runtime_section_gap_and_trailing_bytes() {
        let mut runtime = build_test_runtime_directory();
        insert_gap_before_section(&mut runtime, SECTION_TOKENIZER);
        expect_parse_error_contains(&runtime, "not contiguous");

        let mut runtime = build_test_runtime_directory();
        overlap_section_with_previous(&mut runtime, SECTION_TOKENIZER);
        expect_parse_error_contains(&runtime, "overlap");

        let mut runtime = build_test_runtime_directory();
        runtime.push(0);
        expect_parse_error_contains(&runtime, "trailing bytes");
    }

    #[test]
    fn rejects_duplicate_asset_ids_and_invalid_asset_utf8() {
        let mut runtime = build_test_runtime_directory();
        let (asset2_field, _, _) = asset_payload_record(&runtime, 2);
        put_u32_at(&mut runtime, asset2_field - 4, 1);
        expect_parse_error_contains(&runtime, "duplicate asset");

        let mut runtime = build_test_runtime_directory();
        let (table_offset, _) = section_range(&runtime, SECTION_ASSET_TABLE);
        runtime[table_offset + 4 + 20] = 0xff;
        expect_parse_error_contains(&runtime, "not UTF-8");
    }

    #[test]
    fn rejects_asset_payload_gap_overlap_and_trailing_bytes() {
        let mut runtime = build_test_runtime_directory();
        insert_asset_payload_gap_before_asset(&mut runtime, 2);
        expect_parse_error_contains(&runtime, "payload gap");

        let mut runtime = build_test_runtime_directory();
        let (_, asset1_offset, _) = asset_payload_record(&runtime, 1);
        let (asset2_offset_field, _, _) = asset_payload_record(&runtime, 2);
        put_u32_at(&mut runtime, asset2_offset_field, asset1_offset as u32);
        expect_parse_error_contains(&runtime, "payload overlap");

        let mut runtime = build_test_runtime_directory();
        append_asset_payload_trailing_byte(&mut runtime);
        expect_parse_error_contains(&runtime, "trailing payload");
    }

    #[test]
    fn rejects_malformed_tensor_abi() {
        let mut runtime = build_test_runtime_directory();
        let (abi_offset, _) = section_range(&runtime, SECTION_TENSOR_ABI);
        put_u32_at(&mut runtime, abi_offset + 4, 999_999);
        expect_parse_error_contains(&runtime, "FLM tensor ABI weight_prefix");
    }
}
