use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use base64::Engine;
use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::qwen36_moe::{
    linear_step_launch, Qwen36MoeLinearStepInt4, Qwen36MoeLinearStepParams,
    Qwen36MoeLinearStepWeights,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use supersonic_runtime::flm_model_source::{FlmModelSource, FlmModelSourceOptions};
use supersonic_runtime::qwen36_moe::layer_loader::{
    load_to_gpu, QWEN36_MOE_INT4_GROUP_SIZE, QWEN36_MOE_LOWBIT_NATIVE_INT4,
};

const LAYER_PREFIX: &str = "model.language_model.layers.0";
const LINEAR_PREFIX: &str = "model.language_model.layers.0.linear_attn";
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
    qkv_scale: GpuBuffer,
    qkv_zero: GpuBuffer,
    z_scale: GpuBuffer,
    z_zero: GpuBuffer,
    out_scale: GpuBuffer,
    out_zero: GpuBuffer,
}

struct StageResult {
    output: Vec<u8>,
    workspace: Vec<f32>,
    conv_state: Vec<u8>,
    recurrent_state: Vec<u8>,
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    Ok(value[key]
        .as_u64()
        .with_context(|| format!("missing integer {key}"))? as usize)
}

fn payload_bytes(payload: &Value, expected_dtype: &str) -> Result<(Vec<usize>, Vec<u8>)> {
    let dtype = payload["dtype"].as_str().context("payload dtype")?;
    if dtype != expected_dtype {
        bail!("payload dtype {dtype} != {expected_dtype}");
    }
    let shape = payload["shape"]
        .as_array()
        .context("payload shape")?
        .iter()
        .map(|dim| {
            dim.as_u64()
                .map(|value| value as usize)
                .context("payload shape dimension")
        })
        .collect::<Result<Vec<_>>>()?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload["base64"].as_str().context("payload base64")?)
        .context("decode payload")?;
    let element_bytes = match dtype {
        "bfloat16" => 2,
        "float32" => 4,
        _ => unreachable!(),
    };
    if bytes.len() != shape.iter().product::<usize>() * element_bytes {
        bail!("payload byte length does not match its shape");
    }
    Ok((shape, bytes))
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

fn load_weights(source: &FlmModelSource, ordinal: usize) -> Result<LayerWeights> {
    let store = &source.store;
    let load = |name: &str| load_to_gpu(store, ordinal, name);
    Ok(LayerWeights {
        input_norm: load(&format!("{LAYER_PREFIX}.input_layernorm.weight"))?,
        in_proj_qkv: load(&format!("{LINEAR_PREFIX}.in_proj_qkv.weight"))?,
        in_proj_z: load(&format!("{LINEAR_PREFIX}.in_proj_z.weight"))?,
        in_proj_a: load(&format!("{LINEAR_PREFIX}.in_proj_a.weight"))?,
        in_proj_b: load(&format!("{LINEAR_PREFIX}.in_proj_b.weight"))?,
        conv1d: load(&format!("{LINEAR_PREFIX}.conv1d.weight"))?,
        dt_bias: load(&format!("{LINEAR_PREFIX}.dt_bias"))?,
        a_log: load(&format!("{LINEAR_PREFIX}.A_log"))?,
        norm: load(&format!("{LINEAR_PREFIX}.norm.weight"))?,
        out_proj: load(&format!("{LINEAR_PREFIX}.out_proj.weight"))?,
        qkv_scale: load(&format!("{LINEAR_PREFIX}.in_proj_qkv.weight_int4_scale"))?,
        qkv_zero: load(&format!("{LINEAR_PREFIX}.in_proj_qkv.weight_int4_zero"))?,
        z_scale: load(&format!("{LINEAR_PREFIX}.in_proj_z.weight_int4_scale"))?,
        z_zero: load(&format!("{LINEAR_PREFIX}.in_proj_z.weight_int4_zero"))?,
        out_scale: load(&format!("{LINEAR_PREFIX}.out_proj.weight_int4_scale"))?,
        out_zero: load(&format!("{LINEAR_PREFIX}.out_proj.weight_int4_zero"))?,
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
    let int4 = Qwen36MoeLinearStepInt4 {
        group_size: QWEN36_MOE_INT4_GROUP_SIZE,
        in_proj_qkv_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
        in_proj_qkv_scale: weights.qkv_scale.as_ptr(),
        in_proj_qkv_zero: weights.qkv_zero.as_ptr(),
        in_proj_z_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
        in_proj_z_scale: weights.z_scale.as_ptr(),
        in_proj_z_zero: weights.z_zero.as_ptr(),
        out_proj_type: QWEN36_MOE_LOWBIT_NATIVE_INT4,
        out_proj_scale: weights.out_scale.as_ptr(),
        out_proj_zero: weights.out_zero.as_ptr(),
    };
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
    if report["schema"].as_str() != Some("qwen36-layer0-mode-diagnostic/v1") {
        bail!("unexpected A/B/C report schema");
    }
    if report["prompt"]["token_count"].as_u64() != Some(322)
        || report["prompt"]["final_position"].as_u64() != Some(321)
    {
        bail!("D requires the exact 322-token prompt at final position 321");
    }
    let execution_c = &report["executions"]["C"];
    if execution_c["mode"].as_str() != Some("recurrent")
        || execution_c["position"].as_u64() != Some(321)
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

    let (_, input_bytes) =
        payload_bytes(&execution_c["boundary_payloads"]["layer_input"], "bfloat16")?;
    let (_, conv_state_before) =
        payload_bytes(&execution_c["state_before_final"]["conv_state"], "bfloat16")?;
    let (_, recurrent_state_before) = payload_bytes(
        &execution_c["state_before_final"]["recurrent_state"],
        "float32",
    )?;

    set_backend(Backend::Hip);
    let source = FlmModelSource::open_with_options(
        &artifact_path,
        FlmModelSourceOptions {
            int4_runtime: true,
            verify_block_hashes: true,
        },
    )?;
    let weights = load_weights(&source, 0)?;
    let input = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[geometry.hidden], &input_bytes)?;
    let norm_weight = weights.input_norm.to_host_bytes()?;
    let conv_weight = weights.conv1d.to_host_bytes()?;

    let stage1 = run_stage(
        geometry,
        1,
        &input,
        &weights,
        &conv_state_before,
        &recurrent_state_before,
    )?;
    let stage2 = run_stage(
        geometry,
        2,
        &input,
        &weights,
        &conv_state_before,
        &recurrent_state_before,
    )?;
    let stage3 = run_stage(
        geometry,
        3,
        &input,
        &weights,
        &conv_state_before,
        &recurrent_state_before,
    )?;
    let stage4 = run_stage(
        geometry,
        4,
        &input,
        &weights,
        &conv_state_before,
        &recurrent_state_before,
    )?;
    let stage5 = run_stage(
        geometry,
        5,
        &input,
        &weights,
        &conv_state_before,
        &recurrent_state_before,
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
    let post_residual = stage5.output[..geometry.hidden * 2].to_vec();
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
            &conv_output(&qkv_bytes, &conv_state_before, &conv_weight, geometry),
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
            &stage4.recurrent_state,
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
            &bf16_workspace(&stage5.workspace, core_offset, value_dim),
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
        || stage5.conv_state.len() != conv_state_before.len()
    {
        bail!("staged conv state shape changed");
    }

    let result = json!({
        "schema": "qwen36-layer0-supersonic-diagnostic/v1",
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
        "execution": {
            "label": "D_supersonic_flm_v1_isolated_stages",
            "mode": "recurrent",
            "position": 321,
            "initial_state": "zero",
            "layer_input_checksum": execution_c["layer_input_checksum"],
            "boundary_order": BOUNDARY_ORDER,
            "boundary_payloads": boundaries,
            "state_before_final": execution_c["state_before_final"],
        },
        "capture_methods": {
            "input_rmsnorm": "host reconstruction of staged kernel BF16 formula",
            "conv_output": "host reconstruction from stage1 qkv, C state, and FLM conv weights",
            "out_proj": "BF16 residual difference; kernel publishes only post-residual",
            "all_other_boundaries": "direct stage output, workspace, or mutated state",
        },
        "stages": {
            "executed": [1, 2, 3, 4, 5],
            "fresh_state_per_stage": true,
            "stage2_conv_state_bytes": stage2.conv_state.len(),
            "stage5_recurrent_state_bytes": stage5.recurrent_state.len(),
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
