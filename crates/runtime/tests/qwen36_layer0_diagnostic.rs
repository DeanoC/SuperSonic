use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use base64::Engine;
use gpu_hal::{memset_zeros, set_backend, Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::qwen36_moe::{
    linear_step_launch, Qwen36MoeInt4WeightDesc, Qwen36MoeLinearStepInt4,
    Qwen36MoeLinearStepParams, Qwen36MoeLinearStepWeights,
};
use model_store::store::{BakedStore, Int4StorageKind, Int4StorageView};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use supersonic_runtime::flm_model_source::{FlmModelSource, FlmModelSourceOptions};
use supersonic_runtime::qwen36_moe::layer_loader::{
    load_to_gpu, QWEN36_MOE_INT4_GROUP_SIZE, QWEN36_MOE_LOWBIT_NATIVE_INT4,
};
use supersonic_runtime::qwen36_moe::persistent_decode::build_int4_weight_desc;
use supersonic_runtime::qwen36_moe::types::LoadedInt4Sidecar;
use supersonic_runtime::qwen36_moe::weights::dequant_int4_to_bf16_bytes;

const LAYER_PREFIX: &str = "model.language_model.layers.0";
const LINEAR_PREFIX: &str = "model.language_model.layers.0.linear_attn";
const EMBEDDING_NAME: &str = "model.language_model.embed_tokens.weight";
const KNOWN_BYTES_FIXTURE: &str =
    include_str!("../../../oracle/fixtures/qwen36_native_int4_v1_known_bytes.json");
const EXACT_PROMPT_SHA256: &str =
    "540f92c1fe4446d0f9764de537a1a59603515b94de27b8ea0562420c5f8ffb8b";
const BOUNDARY_ORDER: [&str; 19] = [
    "embedding",
    "layer_input",
    "input_rmsnorm",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "conv_output",
    "conv_silu",
    "q",
    "k",
    "v",
    "beta",
    "decay",
    "recurrent_state_update",
    "core_output",
    "gated_rmsnorm",
    "out_proj",
    "post_attn_residual",
];

#[derive(Clone, Copy)]
struct Geometry {
    hidden: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel_dim: usize,
    rms_norm_eps: f32,
}

impl Geometry {
    fn from_report(report: &Value) -> Result<Self> {
        let geometry = &report["geometry"];
        Ok(Self {
            hidden: usize_field(geometry, "hidden")?,
            num_k_heads: usize_field(geometry, "num_k_heads")?,
            num_v_heads: usize_field(geometry, "num_v_heads")?,
            head_k_dim: usize_field(geometry, "head_k_dim")?,
            head_v_dim: usize_field(geometry, "head_v_dim")?,
            conv_kernel_dim: usize_field(geometry, "conv_kernel_dim")?,
            rms_norm_eps: geometry["rms_norm_eps"]
                .as_f64()
                .context("geometry.rms_norm_eps")? as f32,
        })
    }

    fn key_dim(self) -> usize {
        self.num_k_heads * self.head_k_dim
    }

    fn value_dim(self) -> usize {
        self.num_v_heads * self.head_v_dim
    }

    fn qkv_dim(self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    fn recurrent_state_elems(self) -> usize {
        self.num_v_heads * self.head_k_dim * self.head_v_dim
    }

    fn workspace_floats(self) -> usize {
        self.qkv_dim()
            + self.value_dim()
            + 2 * self.num_v_heads
            + 2 * self.key_dim()
            + 2 * self.num_v_heads * self.head_k_dim
            + 2 * self.num_v_heads
            + self.value_dim()
    }

    fn output_elems(self) -> usize {
        2 * self.num_v_heads * self.head_k_dim + self.value_dim()
    }

    fn params(self, stage: i32) -> Qwen36MoeLinearStepParams {
        Qwen36MoeLinearStepParams {
            stage,
            hidden: self.hidden as i32,
            num_k_heads: self.num_k_heads as i32,
            num_v_heads: self.num_v_heads as i32,
            head_k_dim: self.head_k_dim as i32,
            head_v_dim: self.head_v_dim as i32,
            conv_kernel_dim: self.conv_kernel_dim as i32,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

struct LayerWeights {
    input_norm: GpuBuffer,
    in_proj_qkv: GpuBuffer,
    in_proj_z: GpuBuffer,
    in_proj_a: GpuBuffer,
    in_proj_b: GpuBuffer,
    conv1d: GpuBuffer,
    dt_bias: GpuBuffer,
    a_log: GpuBuffer,
    norm: GpuBuffer,
    out_proj: GpuBuffer,
    qkv_int4: LoadedInt4Sidecar,
    z_int4: LoadedInt4Sidecar,
    out_int4: LoadedInt4Sidecar,
}

struct StageResult {
    output: Vec<u8>,
    workspace: Vec<f32>,
    conv_state: Vec<u8>,
    recurrent_state: Vec<u8>,
}

struct RecurrenceResult {
    initial_conv_state: Vec<u8>,
    initial_recurrent_state: Vec<u8>,
    conv_state_before_final: Vec<u8>,
    recurrent_state_before_final: Vec<u8>,
    final_stage: StageResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DescriptorContract {
    encoding: i32,
    input_group_size: i32,
    output_group_size: i32,
    implicit_zero_code: i32,
    requires_zero: bool,
    packed_row_stride_bytes: u64,
    packed_expert_stride_bytes: u64,
    scale_row_stride_elements: u64,
    scale_expert_stride_elements: u64,
}

fn descriptor_contract(view: &Int4StorageView) -> Result<DescriptorContract> {
    if !matches!(view.logical_shape.as_slice(), [_, _] | [_, _, _])
        || view.logical_shape.contains(&0)
    {
        bail!(
            "INT4 diagnostic view must have nonzero rank-2/rank-3 shape, got {:?}",
            view.logical_shape
        );
    }
    let rows = view.logical_shape[view.logical_shape.len() - 2];
    let cols = view.logical_shape[view.logical_shape.len() - 1];
    let (encoding, requires_zero, implicit_zero_code) = match view.kind {
        Int4StorageKind::RowGroupSymmetric => {
            if view.group_size != 32
                || view.output_group_size != 1
                || view.zero_tensor.is_some()
                || view.implicit_zero_code != Some(8)
            {
                bail!("encoding 2 requires null zero, G32/output-G1, and implicit code 8");
            }
            (2, false, 8)
        }
        Int4StorageKind::TileV1 => {
            if view.group_size != QWEN36_MOE_INT4_GROUP_SIZE as usize
                || view.output_group_size != QWEN36_MOE_INT4_GROUP_SIZE as usize
                || view.zero_tensor.is_none()
                || view.implicit_zero_code.is_some()
            {
                bail!("encoding 1 requires an explicit zero plane and G128 tiles");
            }
            (1, true, -1)
        }
        Int4StorageKind::CtSymmetric => {
            if view.group_size != 128
                || view.output_group_size != 1
                || view.implicit_zero_code != Some(8)
            {
                bail!("CT symmetric INT4 requires G128/output-G1 and implicit code 8");
            }
            (1, true, 8)
        }
    };
    if cols % 2 != 0 || cols % view.group_size != 0 || rows % view.output_group_size != 0 {
        bail!("INT4 diagnostic view shape is not group aligned");
    }
    let packed_row = cols / 2;
    let scale_row = cols / view.group_size;
    let packed_expert = rows
        .checked_mul(packed_row)
        .context("packed expert stride overflow")?;
    let scale_expert = (rows / view.output_group_size)
        .checked_mul(scale_row)
        .context("scale expert stride overflow")?;
    let rank3 = view.logical_shape.len() == 3;
    let expected_packed_expert = rank3.then_some(packed_expert).unwrap_or(0);
    let expected_scale_expert = rank3.then_some(scale_expert).unwrap_or(0);
    if view.packed_row_stride_bytes != packed_row
        || view.packed_expert_stride_bytes != expected_packed_expert
        || view.scale_row_stride_elements != scale_row
        || view.scale_expert_stride_elements != expected_scale_expert
    {
        bail!(
            "INT4 diagnostic view has noncanonical row/expert strides: {:?}",
            view
        );
    }
    Ok(DescriptorContract {
        encoding,
        input_group_size: i32::try_from(view.group_size)?,
        output_group_size: i32::try_from(view.output_group_size)?,
        implicit_zero_code,
        requires_zero,
        packed_row_stride_bytes: u64::try_from(packed_row)?,
        packed_expert_stride_bytes: u64::try_from(expected_packed_expert)?,
        scale_row_stride_elements: u64::try_from(scale_row)?,
        scale_expert_stride_elements: u64::try_from(expected_scale_expert)?,
    })
}

fn build_diagnostic_descriptor(sidecar: &LoadedInt4Sidecar) -> Result<Qwen36MoeInt4WeightDesc> {
    let contract = descriptor_contract(&sidecar.view)?;
    let desc = build_int4_weight_desc(sidecar)?;
    if desc.scale != sidecar.scale.as_ptr()
        || desc.zero != sidecar.zero_ptr()
        || desc.encoding != contract.encoding
        || desc.input_group_size != contract.input_group_size
        || desc.output_group_size != contract.output_group_size
        || desc.implicit_zero_code != contract.implicit_zero_code
        || desc.packed_row_stride_bytes != contract.packed_row_stride_bytes
        || desc.packed_expert_stride_bytes != contract.packed_expert_stride_bytes
        || desc.scale_row_stride_elements != contract.scale_row_stride_elements
        || desc.scale_expert_stride_elements != contract.scale_expert_stride_elements
        || contract.requires_zero != !desc.zero.is_null()
    {
        bail!("production INT4 descriptor contradicts its typed storage view");
    }
    Ok(desc)
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    Ok(value[key]
        .as_u64()
        .with_context(|| format!("missing integer {key}"))? as usize)
}

fn payload(dtype: &str, shape: &[usize], bytes: &[u8]) -> Value {
    json!({
        "dtype": dtype,
        "shape": shape,
        "base64": base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_prompt_contract(report: &Value) -> Result<()> {
    let prompt = &report["prompt"];
    if prompt["sha256"].as_str() != Some(EXACT_PROMPT_SHA256)
        || prompt["token_count"].as_u64() != Some(322)
        || prompt["final_position"].as_u64() != Some(321)
    {
        bail!("D requires the pinned 322-token prompt fixture identity");
    }
    let token_ids = prompt["token_ids"].as_array().context("prompt.token_ids")?;
    let transformers_ids = prompt["transformers_token_ids"]
        .as_array()
        .context("prompt.transformers_token_ids")?;
    if token_ids.len() != 322 || transformers_ids.len() != 322 || token_ids != transformers_ids {
        bail!("D requires identical 322-ID tokenizer sequences");
    }
    Ok(())
}

fn load_projection(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(GpuBuffer, LoadedInt4Sidecar)> {
    let view = store
        .int4_storage_view(name)
        .with_context(|| format!("missing typed INT4 storage view for {name}"))?
        .clone();
    if view.logical_shape.as_slice() != [expected_rows, expected_cols] {
        bail!(
            "{name} logical shape {:?} != [{expected_rows}, {expected_cols}]",
            view.logical_shape
        );
    }
    descriptor_contract(&view).with_context(|| format!("validate {name} storage view"))?;
    let packed = load_to_gpu(store, ordinal, &view.packed_tensor)?;
    let scale = load_to_gpu(store, ordinal, &view.scale_tensor)?;
    let zero = view
        .zero_tensor
        .as_deref()
        .map(|zero_name| load_to_gpu(store, ordinal, zero_name))
        .transpose()?;
    let sidecar = LoadedInt4Sidecar { scale, zero, view };
    build_diagnostic_descriptor(&sidecar)
        .with_context(|| format!("build {name} production descriptor"))?;
    Ok((packed, sidecar))
}

fn load_weights(
    source: &FlmModelSource,
    ordinal: usize,
    geometry: Geometry,
) -> Result<LayerWeights> {
    let store = &source.store;
    let load = |name: &str| load_to_gpu(store, ordinal, name);
    let qkv_name = format!("{LINEAR_PREFIX}.in_proj_qkv.weight");
    let z_name = format!("{LINEAR_PREFIX}.in_proj_z.weight");
    let out_name = format!("{LINEAR_PREFIX}.out_proj.weight");
    let (in_proj_qkv, qkv_int4) = load_projection(
        store,
        ordinal,
        &qkv_name,
        geometry.qkv_dim(),
        geometry.hidden,
    )?;
    let (in_proj_z, z_int4) = load_projection(
        store,
        ordinal,
        &z_name,
        geometry.value_dim(),
        geometry.hidden,
    )?;
    let (out_proj, out_int4) = load_projection(
        store,
        ordinal,
        &out_name,
        geometry.hidden,
        geometry.value_dim(),
    )?;
    Ok(LayerWeights {
        input_norm: load(&format!("{LAYER_PREFIX}.input_layernorm.weight"))?,
        in_proj_qkv,
        in_proj_z,
        in_proj_a: load(&format!("{LINEAR_PREFIX}.in_proj_a.weight"))?,
        in_proj_b: load(&format!("{LINEAR_PREFIX}.in_proj_b.weight"))?,
        conv1d: load(&format!("{LINEAR_PREFIX}.conv1d.weight"))?,
        dt_bias: load(&format!("{LINEAR_PREFIX}.dt_bias"))?,
        a_log: load(&format!("{LINEAR_PREFIX}.A_log"))?,
        norm: load(&format!("{LINEAR_PREFIX}.norm.weight"))?,
        out_proj,
        qkv_int4,
        z_int4,
        out_int4,
    })
}

fn linear_int4(weights: &LayerWeights) -> Result<Qwen36MoeLinearStepInt4> {
    let qkv = build_diagnostic_descriptor(&weights.qkv_int4)?;
    let z = build_diagnostic_descriptor(&weights.z_int4)?;
    let out = build_diagnostic_descriptor(&weights.out_int4)?;
    let group_size = weights.qkv_int4.view.group_size;
    if weights.z_int4.view.group_size != group_size
        || weights.out_int4.view.group_size != group_size
    {
        bail!("linear-attention INT4 projections use mixed input group sizes");
    }
    Ok(Qwen36MoeLinearStepInt4 {
        group_size: i32::try_from(group_size)?,
        in_proj_qkv_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
        in_proj_qkv: qkv,
        in_proj_qkv_scale: weights.qkv_int4.scale.as_ptr(),
        in_proj_qkv_zero: weights.qkv_int4.zero_ptr(),
        in_proj_z_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
        in_proj_z: z,
        in_proj_z_scale: weights.z_int4.scale.as_ptr(),
        in_proj_z_zero: weights.z_int4.zero_ptr(),
        out_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
        out_proj: out,
        out_proj_scale: weights.out_int4.scale.as_ptr(),
        out_proj_zero: weights.out_int4.zero_ptr(),
    })
}

fn descriptor_evidence(sidecar: &LoadedInt4Sidecar) -> Result<Value> {
    let desc = build_diagnostic_descriptor(sidecar)?;
    Ok(json!({
        "storage_kind": match sidecar.view.kind {
            Int4StorageKind::TileV1 => "tile-v1",
            Int4StorageKind::RowGroupSymmetric => "row-group-symmetric",
            Int4StorageKind::CtSymmetric => "ct-symmetric",
        },
        "logical_shape": sidecar.view.logical_shape,
        "scale_tensor": sidecar.view.scale_tensor,
        "zero_tensor": sidecar.view.zero_tensor,
        "encoding": desc.encoding,
        "zero_is_null": desc.zero.is_null(),
        "packed_row_stride_bytes": desc.packed_row_stride_bytes,
        "packed_expert_stride_bytes": desc.packed_expert_stride_bytes,
        "scale_row_stride_elements": desc.scale_row_stride_elements,
        "scale_expert_stride_elements": desc.scale_expert_stride_elements,
        "input_group_size": desc.input_group_size,
        "output_group_size": desc.output_group_size,
        "implicit_zero_code": desc.implicit_zero_code,
    }))
}

fn hex_bytes(value: &str) -> Result<Vec<u8>> {
    if value.len() % 2 != 0 {
        bail!("fixture hex must contain byte pairs");
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).context("fixture hex is not ASCII")?;
            u8::from_str_radix(pair, 16).context("invalid fixture hex byte")
        })
        .collect()
}

fn abi_validation_evidence() -> Result<Value> {
    let fixture: Value =
        serde_json::from_str(KNOWN_BYTES_FIXTURE).context("parse known-byte ABI fixture")?;
    if fixture["schema"].as_str() != Some("qwen36-native-int4-known-bytes/v1") {
        bail!("unexpected known-byte ABI fixture schema");
    }
    if !fixture["provenance"]
        .as_str()
        .unwrap_or_default()
        .contains("no producer packer")
    {
        bail!("known-byte ABI fixture is not producer-independent");
    }
    let rows = fixture["logical_shape"][0]
        .as_u64()
        .context("fixture rows")? as usize;
    let cols = fixture["logical_shape"][1]
        .as_u64()
        .context("fixture cols")? as usize;
    let group_size = fixture["group_size"].as_u64().context("group size")? as usize;
    let row_pattern = hex_bytes(
        fixture["packed"]["row_pattern_hex"]
            .as_str()
            .context("row pattern")?,
    )?;
    let packed_row = row_pattern.repeat(
        fixture["packed"]["pattern_repeats_per_row"]
            .as_u64()
            .context("pattern repeats")? as usize,
    );
    let packed = packed_row.repeat(
        fixture["packed"]["row_repeats"]
            .as_u64()
            .context("row repeats")? as usize,
    );
    let scale = hex_bytes(
        fixture["scale_bf16_le_hex"]
            .as_str()
            .context("scale bytes")?,
    )?;
    let zero = hex_bytes(fixture["zero_bf16_le_hex"].as_str().context("zero bytes")?)?;
    let decoded = dequant_int4_to_bf16_bytes(&packed, &scale, &zero, rows, cols, group_size);
    let expected_tiles = fixture["expected_bf16_bits_by_tile"]
        .as_array()
        .context("expected tile tables")?;
    let mut production_decoder_match = true;
    'rows: for row in 0..rows {
        for col in 0..cols {
            let tile = (row / group_size) * 2 + col / group_size;
            let nibble = col % 16;
            let expected = u16::from_str_radix(
                expected_tiles[tile][nibble]
                    .as_str()
                    .context("expected BF16 bits")?,
                16,
            )
            .context("invalid expected BF16 bits")?;
            let offset = (row * cols + col) * 2;
            let actual = u16::from_le_bytes([decoded[offset], decoded[offset + 1]]);
            if actual != expected {
                production_decoder_match = false;
                break 'rows;
            }
        }
    }
    Ok(json!({
        "fixture_schema": fixture["schema"],
        "fixture_sha256": format!("{:x}", Sha256::digest(KNOWN_BYTES_FIXTURE.as_bytes())),
        "production_decoder_match": production_decoder_match,
    }))
}

fn artifact_input_sequence(
    source: &FlmModelSource,
    report: &Value,
    geometry: Geometry,
) -> Result<Vec<Vec<u8>>> {
    let token_ids = report["prompt"]["transformers_token_ids"]
        .as_array()
        .context("prompt.transformers_token_ids")?;
    if token_ids.len() != 322 {
        bail!("D requires exactly 322 transformer token IDs");
    }
    let shape = source
        .store
        .shape(EMBEDDING_NAME)
        .context("artifact embedding shape")?;
    if shape.len() != 2 || shape[1] != geometry.hidden {
        bail!(
            "artifact embedding shape {:?} does not end in hidden={}",
            shape,
            geometry.hidden
        );
    }
    let embedding = source
        .store
        .raw_bytes(EMBEDDING_NAME)
        .context("artifact embedding raw BF16 bytes")?;
    let row_bytes = geometry.hidden * 2;
    if embedding.len() != shape[0] * row_bytes {
        bail!("artifact embedding byte length contradicts its shape");
    }
    token_ids
        .iter()
        .enumerate()
        .map(|(position, token)| {
            let token = token
                .as_u64()
                .with_context(|| format!("token ID at position {position}"))?
                as usize;
            if token >= shape[0] {
                bail!("token ID {token} at position {position} exceeds artifact vocab");
            }
            let start = token * row_bytes;
            Ok(embedding[start..start + row_bytes].to_vec())
        })
        .collect()
}

fn run_production_recurrence(
    geometry: Geometry,
    inputs: &[Vec<u8>],
    weights: &LayerWeights,
) -> Result<RecurrenceResult> {
    if inputs.len() != 322 {
        bail!("production recurrence requires 322 inputs");
    }
    let ordinal = 0;
    let mut conv_state = GpuBuffer::zeros(
        ordinal,
        ScalarType::BF16,
        &[geometry.qkv_dim(), geometry.conv_kernel_dim - 1],
    )?;
    let mut recurrent_state = GpuBuffer::zeros(
        ordinal,
        ScalarType::F32,
        &[geometry.recurrent_state_elems()],
    )?;
    let initial_conv_state = conv_state.to_host_bytes()?;
    let initial_recurrent_state = recurrent_state.to_host_bytes()?;
    if initial_conv_state.iter().any(|byte| *byte != 0)
        || initial_recurrent_state.iter().any(|byte| *byte != 0)
    {
        bail!("GPU zero-state allocation published nonzero bytes");
    }
    let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geometry.output_elems()])?;
    let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[geometry.workspace_floats()])?;
    let mut sync = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])?;
    let int4 = linear_int4(weights)?;
    let mut state_before_final = None;
    for (position, input_bytes) in inputs.iter().enumerate() {
        if input_bytes.len() != geometry.hidden * 2 {
            bail!("input {position} has the wrong BF16 byte length");
        }
        if position == inputs.len() - 1 {
            state_before_final = Some((
                conv_state.to_host_bytes()?,
                recurrent_state.to_host_bytes()?,
            ));
        }
        let input =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[geometry.hidden], input_bytes)?;
        let pointers = Qwen36MoeLinearStepWeights {
            input_hidden: input.as_ptr(),
            input_norm_w: weights.input_norm.as_ptr(),
            in_proj_qkv_w: weights.in_proj_qkv.as_ptr(),
            in_proj_z_w: weights.in_proj_z.as_ptr(),
            in_proj_a_w: weights.in_proj_a.as_ptr(),
            in_proj_b_w: weights.in_proj_b.as_ptr(),
            conv1d_w: weights.conv1d.as_ptr(),
            conv1d_bias: std::ptr::null(),
            dt_bias: weights.dt_bias.as_ptr(),
            a_log: weights.a_log.as_ptr(),
            norm_w: weights.norm.as_ptr(),
            out_proj_w: weights.out_proj.as_ptr(),
            conv_state: conv_state.as_mut_ptr(),
            recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
        };
        memset_zeros(ordinal, sync.as_mut_ptr(), sync.len_bytes())
            .with_context(|| format!("reset recurrence sync at position {position}"))?;
        linear_step_launch(
            ordinal,
            ScalarType::BF16,
            geometry.params(5),
            &pointers,
            &int4,
            &mut output,
            &mut workspace,
            &mut sync,
        )
        .with_context(|| format!("production linear recurrence at position {position}"))?;
    }
    let (conv_state_before_final, recurrent_state_before_final) =
        state_before_final.context("missing state before final position")?;
    Ok(RecurrenceResult {
        initial_conv_state,
        initial_recurrent_state,
        conv_state_before_final,
        recurrent_state_before_final,
        final_stage: StageResult {
            output: output.to_host_bytes()?,
            workspace: f32_values(&workspace.to_host_bytes()?),
            conv_state: conv_state.to_host_bytes()?,
            recurrent_state: recurrent_state.to_host_bytes()?,
        },
    })
}

fn run_stage(
    geometry: Geometry,
    stage: i32,
    input: &GpuBuffer,
    weights: &LayerWeights,
    conv_state_before: &[u8],
    recurrent_state_before: &[u8],
) -> Result<StageResult> {
    let ordinal = 0;
    let mut conv_state = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[geometry.qkv_dim(), geometry.conv_kernel_dim - 1],
        conv_state_before,
    )?;
    let mut recurrent_state = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::F32,
        &[geometry.recurrent_state_elems()],
        recurrent_state_before,
    )?;
    let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[geometry.output_elems()])?;
    let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[geometry.workspace_floats()])?;
    let mut sync = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])?;
    let pointers = Qwen36MoeLinearStepWeights {
        input_hidden: input.as_ptr(),
        input_norm_w: weights.input_norm.as_ptr(),
        in_proj_qkv_w: weights.in_proj_qkv.as_ptr(),
        in_proj_z_w: weights.in_proj_z.as_ptr(),
        in_proj_a_w: weights.in_proj_a.as_ptr(),
        in_proj_b_w: weights.in_proj_b.as_ptr(),
        conv1d_w: weights.conv1d.as_ptr(),
        conv1d_bias: std::ptr::null(),
        dt_bias: weights.dt_bias.as_ptr(),
        a_log: weights.a_log.as_ptr(),
        norm_w: weights.norm.as_ptr(),
        out_proj_w: weights.out_proj.as_ptr(),
        conv_state: conv_state.as_mut_ptr(),
        recurrent_state: recurrent_state.as_mut_ptr() as *mut f32,
    };
    let int4 = linear_int4(weights)?;
    linear_step_launch(
        ordinal,
        ScalarType::BF16,
        geometry.params(stage),
        &pointers,
        &int4,
        &mut output,
        &mut workspace,
        &mut sync,
    )
    .with_context(|| format!("linear stage {stage}"))?;
    Ok(StageResult {
        output: output.to_host_bytes()?,
        workspace: f32_values(&workspace.to_host_bytes()?),
        conv_state: conv_state.to_host_bytes()?,
        recurrent_state: recurrent_state.to_host_bytes()?,
    })
}

fn input_rmsnorm(input: &[u8], norm_weight: &[u8], epsilon: f32) -> Vec<u8> {
    let input = bf16_values(input);
    let weight = bf16_values(norm_weight);
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_square + epsilon).sqrt().recip();
    bf16_bytes(
        input
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inv_rms * (1.0 + weight)),
    )
}

fn conv_output(qkv: &[u8], conv_state: &[u8], conv_weight: &[u8], geometry: Geometry) -> Vec<u8> {
    let qkv = bf16_values(qkv);
    let state = bf16_values(conv_state);
    let weight = bf16_values(conv_weight);
    let state_width = geometry.conv_kernel_dim - 1;
    bf16_bytes((0..geometry.qkv_dim()).map(|channel| {
        let mut sum = 0.0f32;
        for position in 0..state_width {
            sum += state[channel * state_width + position]
                * weight[channel * geometry.conv_kernel_dim + position];
        }
        sum += qkv[channel] * weight[channel * geometry.conv_kernel_dim + state_width];
        sum
    }))
}

fn bf16_workspace(workspace: &[f32], offset: usize, count: usize) -> Vec<u8> {
    bf16_bytes(workspace[offset..offset + count].iter().copied())
}

fn row_group_test_view() -> Int4StorageView {
    Int4StorageView {
        kind: Int4StorageKind::RowGroupSymmetric,
        group_size: 32,
        packed_tensor: "weight".into(),
        scale_tensor: "weight_int4_scale".into(),
        zero_tensor: None,
        logical_shape: vec![2, 4, 64],
        packed_row_stride_bytes: 32,
        packed_expert_stride_bytes: 128,
        scale_row_stride_elements: 2,
        scale_expert_stride_elements: 8,
        output_group_size: 1,
        implicit_zero_code: Some(8),
    }
}

#[test]
fn diagnostic_descriptor_contract_supports_row_group_encoding2() {
    let contract = descriptor_contract(&row_group_test_view()).expect("row-group contract");

    assert_eq!(contract.encoding, 2);
    assert_eq!(contract.input_group_size, 32);
    assert_eq!(contract.output_group_size, 1);
    assert_eq!(contract.implicit_zero_code, 8);
    assert!(!contract.requires_zero);
    assert_eq!(contract.packed_row_stride_bytes, 32);
    assert_eq!(contract.packed_expert_stride_bytes, 128);
    assert_eq!(contract.scale_row_stride_elements, 2);
    assert_eq!(contract.scale_expert_stride_elements, 8);
}

#[test]
fn diagnostic_descriptor_contract_retains_tile_v1_encoding1() {
    let view = Int4StorageView {
        kind: Int4StorageKind::TileV1,
        group_size: 128,
        packed_tensor: "weight".into(),
        scale_tensor: "weight_int4_scale".into(),
        zero_tensor: Some("weight_int4_zero".into()),
        logical_shape: vec![256, 256],
        packed_row_stride_bytes: 128,
        packed_expert_stride_bytes: 0,
        scale_row_stride_elements: 2,
        scale_expert_stride_elements: 0,
        output_group_size: 128,
        implicit_zero_code: None,
    };

    let contract = descriptor_contract(&view).expect("tile-v1 contract");

    assert_eq!(contract.encoding, 1);
    assert_eq!(contract.input_group_size, 128);
    assert_eq!(contract.output_group_size, 128);
    assert_eq!(contract.implicit_zero_code, -1);
    assert!(contract.requires_zero);
}

#[test]
fn diagnostic_descriptor_contract_rejects_malformed_encoding2_views() {
    let mut cases = Vec::new();
    let mut wrong_group = row_group_test_view();
    wrong_group.group_size = 64;
    cases.push(wrong_group);
    let mut wrong_output_group = row_group_test_view();
    wrong_output_group.output_group_size = 32;
    cases.push(wrong_output_group);
    let mut wrong_zero = row_group_test_view();
    wrong_zero.zero_tensor = Some("weight_int4_zero".into());
    cases.push(wrong_zero);
    let mut wrong_implicit = row_group_test_view();
    wrong_implicit.implicit_zero_code = Some(7);
    cases.push(wrong_implicit);
    let mut wrong_packed_stride = row_group_test_view();
    wrong_packed_stride.packed_expert_stride_bytes -= 1;
    cases.push(wrong_packed_stride);
    let mut wrong_scale_stride = row_group_test_view();
    wrong_scale_stride.scale_expert_stride_elements -= 1;
    cases.push(wrong_scale_stride);

    for view in cases {
        assert!(descriptor_contract(&view).is_err(), "accepted {view:?}");
    }
}

#[test]
fn diagnostic_prompt_contract_pins_fixture_hash_and_322_ids() {
    let report = json!({
        "prompt": {
            "sha256": EXACT_PROMPT_SHA256,
            "token_count": 322,
            "final_position": 321,
            "token_ids": vec![1; 322],
            "transformers_token_ids": vec![1; 322],
        }
    });

    validate_prompt_contract(&report).expect("exact prompt contract");

    let mut wrong = report;
    wrong["prompt"]["sha256"] = Value::String("wrong".into());
    assert!(validate_prompt_contract(&wrong).is_err());
}

#[test]
fn captures_real_flm_layer0_stages_from_mode_aligned_reference() -> Result<()> {
    let Some(abc_path) = std::env::var_os("SUPERSONIC_QWEN36_LAYER0_ABC_JSON") else {
        eprintln!("skip: SUPERSONIC_QWEN36_LAYER0_ABC_JSON is not set");
        return Ok(());
    };
    let output_path = PathBuf::from(
        std::env::var_os("SUPERSONIC_QWEN36_LAYER0_D_OUTPUT")
            .context("SUPERSONIC_QWEN36_LAYER0_D_OUTPUT must be set with the A/B/C input")?,
    );
    let abc_path = PathBuf::from(abc_path);
    let abc_bytes = std::fs::read(&abc_path)?;
    let report: Value = serde_json::from_slice(&abc_bytes).context("parse A/B/C report")?;
    if report["schema"].as_str() != Some("qwen36-layer0-mode-diagnostic/v2") {
        bail!("unexpected A/B/C report schema");
    }
    validate_prompt_contract(&report)?;
    let prompt_path = PathBuf::from(
        report["prompt"]["path"]
            .as_str()
            .context("prompt fixture path in A/B/C report")?,
    );
    let prompt_hash = sha256_file(&prompt_path)?;
    if prompt_hash != EXACT_PROMPT_SHA256
        || report["prompt"]["sha256"].as_str() != Some(&prompt_hash)
    {
        bail!("durable prompt fixture no longer matches its pinned SHA-256");
    }
    let execution_c = &report["executions"]["C"];
    if execution_c["mode"].as_str() != Some("recurrent")
        || execution_c["position"].as_u64() != Some(321)
        || execution_c["positions_executed"].as_u64() != Some(322)
        || execution_c["initial_state"].as_str() != Some("zero")
    {
        bail!("C is not the required zero-state recurrent execution");
    }

    let geometry = Geometry::from_report(&report)?;
    let artifact_path = PathBuf::from(
        report["artifact"]["path"]
            .as_str()
            .context("artifact path in A/B/C report")?,
    );
    let artifact_hash = sha256_file(&artifact_path)?;
    if report["artifact"]["sha256"].as_str() != Some(&artifact_hash) {
        bail!("artifact SHA-256 no longer matches the A/B/C report");
    }

    set_backend(Backend::Hip);
    let source = FlmModelSource::open_with_options(
        &artifact_path,
        FlmModelSourceOptions {
            int4_runtime: true,
            verify_block_hashes: true,
        },
    )?;
    let weights = load_weights(&source, 0, geometry)?;
    let abi_validation = abi_validation_evidence()?;
    let int4_storage_views = json!({
        "in_proj_qkv": descriptor_evidence(&weights.qkv_int4)?,
        "in_proj_z": descriptor_evidence(&weights.z_int4)?,
        "out_proj": descriptor_evidence(&weights.out_int4)?,
    });
    let input_rows = artifact_input_sequence(&source, &report, geometry)?;
    let input_sequence_bytes = input_rows.concat();
    let input_bytes = input_rows
        .last()
        .context("artifact input sequence is empty")?
        .clone();
    let recurrence = run_production_recurrence(geometry, &input_rows, &weights)?;
    let conv_state_before = &recurrence.conv_state_before_final;
    let recurrent_state_before = &recurrence.recurrent_state_before_final;
    let input = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[geometry.hidden], &input_bytes)?;
    let norm_weight = weights.input_norm.to_host_bytes()?;
    let conv_weight = weights.conv1d.to_host_bytes()?;

    let stage1 = run_stage(
        geometry,
        1,
        &input,
        &weights,
        conv_state_before,
        recurrent_state_before,
    )?;
    let stage2 = run_stage(
        geometry,
        2,
        &input,
        &weights,
        conv_state_before,
        recurrent_state_before,
    )?;
    let stage3 = run_stage(
        geometry,
        3,
        &input,
        &weights,
        conv_state_before,
        recurrent_state_before,
    )?;
    let stage4 = run_stage(
        geometry,
        4,
        &input,
        &weights,
        conv_state_before,
        recurrent_state_before,
    )?;
    let stage5 = run_stage(
        geometry,
        5,
        &input,
        &weights,
        conv_state_before,
        recurrent_state_before,
    )?;

    let qkv_dim = geometry.qkv_dim();
    let value_dim = geometry.value_dim();
    let value_key_dim = geometry.num_v_heads * geometry.head_k_dim;
    let z_offset = qkv_dim;
    let a_offset = z_offset + value_dim;
    let b_offset = a_offset + geometry.num_v_heads;
    let q_norm_offset = b_offset + geometry.num_v_heads;
    let k_norm_offset = q_norm_offset + geometry.key_dim();
    let q_offset = k_norm_offset + geometry.key_dim();
    let k_offset = q_offset + value_key_dim;
    let beta_offset = k_offset + value_key_dim;
    let decay_offset = beta_offset + geometry.num_v_heads;
    let core_offset = decay_offset + geometry.num_v_heads;

    let qkv_bytes = stage1.output[..qkv_dim * 2].to_vec();
    let post_residual = recurrence.final_stage.output[..geometry.hidden * 2].to_vec();
    let input_values = bf16_values(&input_bytes);
    let post_values = bf16_values(&post_residual);
    let out_proj = bf16_bytes(
        post_values
            .iter()
            .zip(input_values)
            .map(|(post, input)| post - input),
    );
    let mut boundaries = Map::new();
    boundaries.insert(
        "embedding".into(),
        payload("bfloat16", &[geometry.hidden], &input_bytes),
    );
    boundaries.insert(
        "layer_input".into(),
        payload("bfloat16", &[geometry.hidden], &input_bytes),
    );
    boundaries.insert(
        "input_rmsnorm".into(),
        payload(
            "bfloat16",
            &[geometry.hidden],
            &input_rmsnorm(&input_bytes, &norm_weight, geometry.rms_norm_eps),
        ),
    );
    boundaries.insert(
        "in_proj_qkv".into(),
        payload("bfloat16", &[qkv_dim], &qkv_bytes),
    );
    boundaries.insert(
        "in_proj_z".into(),
        payload(
            "bfloat16",
            &[value_dim],
            &bf16_workspace(&stage1.workspace, z_offset, value_dim),
        ),
    );
    boundaries.insert(
        "in_proj_a".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads],
            &bf16_workspace(&stage1.workspace, a_offset, geometry.num_v_heads),
        ),
    );
    boundaries.insert(
        "in_proj_b".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads],
            &bf16_workspace(&stage1.workspace, b_offset, geometry.num_v_heads),
        ),
    );
    boundaries.insert(
        "conv_output".into(),
        payload(
            "bfloat16",
            &[qkv_dim],
            &conv_output(&qkv_bytes, conv_state_before, &conv_weight, geometry),
        ),
    );
    boundaries.insert(
        "conv_silu".into(),
        payload("bfloat16", &[qkv_dim], &stage2.output[..qkv_dim * 2]),
    );
    boundaries.insert(
        "q".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads, geometry.head_k_dim],
            &stage3.output[..value_key_dim * 2],
        ),
    );
    boundaries.insert(
        "k".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads, geometry.head_k_dim],
            &stage3.output[value_key_dim * 2..value_key_dim * 4],
        ),
    );
    boundaries.insert(
        "v".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads, geometry.head_v_dim],
            &stage3.output[value_key_dim * 4..(2 * value_key_dim + value_dim) * 2],
        ),
    );
    boundaries.insert(
        "beta".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads],
            &bf16_workspace(&stage4.workspace, beta_offset, geometry.num_v_heads),
        ),
    );
    boundaries.insert(
        "decay".into(),
        payload(
            "float32",
            &[geometry.num_v_heads],
            &stage4.workspace[decay_offset..decay_offset + geometry.num_v_heads]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
    );
    boundaries.insert(
        "recurrent_state_update".into(),
        payload(
            "float32",
            &[
                geometry.num_v_heads,
                geometry.head_k_dim,
                geometry.head_v_dim,
            ],
            &recurrence.final_stage.recurrent_state,
        ),
    );
    boundaries.insert(
        "core_output".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads, geometry.head_v_dim],
            &stage4.output[..value_dim * 2],
        ),
    );
    boundaries.insert(
        "gated_rmsnorm".into(),
        payload(
            "bfloat16",
            &[geometry.num_v_heads, geometry.head_v_dim],
            &bf16_workspace(&recurrence.final_stage.workspace, core_offset, value_dim),
        ),
    );
    boundaries.insert(
        "out_proj".into(),
        payload("bfloat16", &[geometry.hidden], &out_proj),
    );
    boundaries.insert(
        "post_attn_residual".into(),
        payload("bfloat16", &[geometry.hidden], &post_residual),
    );
    if boundaries
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>()
        != BOUNDARY_ORDER.into_iter().collect()
    {
        bail!("D boundary schema is incomplete");
    }
    if stage2.conv_state.len() != conv_state_before.len()
        || recurrence.final_stage.conv_state.len() != conv_state_before.len()
    {
        bail!("staged conv state shape changed");
    }
    let stage2_matches_recurrence_conv_state =
        stage2.conv_state == recurrence.final_stage.conv_state;
    let stage4_matches_recurrence_state =
        stage4.recurrent_state == recurrence.final_stage.recurrent_state;
    let stage5_matches_recurrence_output = stage5.output == recurrence.final_stage.output
        && stage5.conv_state == recurrence.final_stage.conv_state
        && stage5.recurrent_state == recurrence.final_stage.recurrent_state;

    let result = json!({
        "schema": "qwen36-layer0-supersonic-diagnostic/v2",
        "source_abc": {
            "path": abc_path,
            "sha256": format!("{:x}", Sha256::digest(&abc_bytes)),
            "schema": report["schema"],
        },
        "artifact": {
            "path": artifact_path,
            "sha256": artifact_hash,
            "size": std::fs::metadata(&source.path)?.len(),
            "profile": "supersonic",
            "payload_hashes_verified": true,
        },
        "abi_validation": abi_validation,
        "execution": {
            "label": "D_supersonic_flm_v1_full_zero_state_recurrence",
            "mode": "recurrent",
            "position": 321,
            "positions_executed": 322,
            "initial_state": "zero",
            "layer_input_checksum": format!("{:x}", Sha256::digest(&input_bytes)),
            "input_sequence_payload": payload(
                "bfloat16",
                &[322, geometry.hidden],
                &input_sequence_bytes,
            ),
            "initial_state_payloads": {
                "conv_state": payload(
                    "bfloat16",
                    &[geometry.qkv_dim(), geometry.conv_kernel_dim - 1],
                    &recurrence.initial_conv_state,
                ),
                "recurrent_state": payload(
                    "float32",
                    &[
                        geometry.num_v_heads,
                        geometry.head_k_dim,
                        geometry.head_v_dim,
                    ],
                    &recurrence.initial_recurrent_state,
                ),
            },
            "boundary_order": BOUNDARY_ORDER,
            "boundary_payloads": boundaries,
            "state_before_final": {
                "conv_state": payload(
                    "bfloat16",
                    &[geometry.qkv_dim(), geometry.conv_kernel_dim - 1],
                    &recurrence.conv_state_before_final,
                ),
                "recurrent_state": payload(
                    "float32",
                    &[
                        geometry.num_v_heads,
                        geometry.head_k_dim,
                        geometry.head_v_dim,
                    ],
                    &recurrence.recurrent_state_before_final,
                ),
            },
            "state_after_final": {
                "conv_state": payload(
                    "bfloat16",
                    &[geometry.qkv_dim(), geometry.conv_kernel_dim - 1],
                    &recurrence.final_stage.conv_state,
                ),
                "recurrent_state": payload(
                    "float32",
                    &[
                        geometry.num_v_heads,
                        geometry.head_k_dim,
                        geometry.head_v_dim,
                    ],
                    &recurrence.final_stage.recurrent_state,
                ),
            },
        },
        "capture_methods": {
            "input_rmsnorm": "host reconstruction of staged kernel BF16 formula",
            "conv_output": "host reconstruction from stage1 qkv, D recurrence state, and FLM conv weights",
            "out_proj": "BF16 residual difference; kernel publishes only post-residual",
            "all_other_boundaries": "direct stage output, workspace, artifact input, or production-mutated state",
            "prompt_fixture": {
                "path": prompt_path,
                "sha256": prompt_hash,
                "token_count": 322,
            },
            "int4_storage_views": int4_storage_views,
        },
        "recurrence": {
            "production_stage": 5,
            "positions_executed": 322,
            "first_position": 0,
            "final_position": 321,
            "state_seed": "gpu_zero_buffers",
            "input_source": "artifact_embedding_rows_from_prompt_token_ids",
        },
        "stages": {
            "executed": [1, 2, 3, 4, 5],
            "fresh_state_per_stage": true,
            "stage2_matches_recurrence_conv_state": stage2_matches_recurrence_conv_state,
            "stage4_matches_recurrence_state": stage4_matches_recurrence_state,
            "stage5_matches_recurrence_output": stage5_matches_recurrence_output,
        },
    });
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output_path, serde_json::to_vec_pretty(&result)?)?;
    eprintln!(
        "wrote D layer-0 diagnostic to {} using artifact {}",
        output_path.display(),
        source.path.display()
    );
    Ok(())
}
