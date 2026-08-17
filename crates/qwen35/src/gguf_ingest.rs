//! Qwen3.8 GGUF → SuperSonic role map and packed weight load. MTP `blk.64` is ignored.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use model_store::gguf::GgufFile;
use model_store::gqh::{self, GqhHeader, GqhRung};

use crate::config::TextConfig;
use crate::weights::{
    ggml_k_row_bytes, FullWeights, LayerKind, LayerWeights, LinearWeights, Qwen35Weights,
    LOWBIT_GGML_Q2_K, LOWBIT_GGML_Q4_K, LOWBIT_GGML_Q5_K, LOWBIT_GGML_Q6_K, LOWBIT_GGML_Q8_0,
};

#[derive(Clone)]
pub struct MappedTensor {
    pub role: String,
    pub gguf_name: String,
    pub kind: LayerKind,
    pub layer: Option<usize>,
}

pub fn load_text_config(hf_model_dir: &Path) -> Result<TextConfig, String> {
    Ok(crate::config::load_config(hf_model_dir)?.text_config)
}

/// True when `path` is a GGUF with at least one GQH tensor header.
pub fn is_gqh_gguf(path: &Path) -> Result<bool, model_store::Error> {
    let file = GgufFile::open(path)?;
    Ok(file.gqh_header_count() > 0)
}

/// Expected GGUF tensor names for the 64 language layers (no MTP).
pub fn expected_gguf_names(config: &TextConfig) -> Vec<MappedTensor> {
    let mut out = vec![
        map("token_embd.weight", "token_embd.weight", LayerKind::Full, None),
        map("output_norm.weight", "output_norm.weight", LayerKind::Full, None),
        map("output.weight", "output.weight", LayerKind::Full, None),
    ];
    for idx in 0..config.num_hidden_layers {
        let prefix = format!("blk.{idx}");
        out.push(map(
            &format!("layers.{idx}.input_layernorm"),
            &format!("{prefix}.attn_norm.weight"),
            LayerKind::Full,
            Some(idx),
        ));
        out.push(map(
            &format!("layers.{idx}.post_attention_layernorm"),
            &format!("{prefix}.post_attention_norm.weight"),
            LayerKind::Full,
            Some(idx),
        ));
        out.push(map(
            &format!("layers.{idx}.mlp.gate_proj"),
            &format!("{prefix}.ffn_gate.weight"),
            LayerKind::Full,
            Some(idx),
        ));
        out.push(map(
            &format!("layers.{idx}.mlp.up_proj"),
            &format!("{prefix}.ffn_up.weight"),
            LayerKind::Full,
            Some(idx),
        ));
        out.push(map(
            &format!("layers.{idx}.mlp.down_proj"),
            &format!("{prefix}.ffn_down.weight"),
            LayerKind::Full,
            Some(idx),
        ));
        if config.is_full_attention(idx) {
            out.push(map(
                &format!("layers.{idx}.self_attn.q_proj"),
                &format!("{prefix}.attn_q.weight"),
                LayerKind::Full,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.self_attn.k_proj"),
                &format!("{prefix}.attn_k.weight"),
                LayerKind::Full,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.self_attn.v_proj"),
                &format!("{prefix}.attn_v.weight"),
                LayerKind::Full,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.self_attn.o_proj"),
                &format!("{prefix}.attn_output.weight"),
                LayerKind::Full,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.self_attn.q_norm"),
                &format!("{prefix}.attn_q_norm.weight"),
                LayerKind::Full,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.self_attn.k_norm"),
                &format!("{prefix}.attn_k_norm.weight"),
                LayerKind::Full,
                Some(idx),
            ));
        } else {
            out.push(map(
                &format!("layers.{idx}.linear_attn.in_proj_qkv"),
                &format!("{prefix}.attn_qkv.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.in_proj_z"),
                &format!("{prefix}.attn_gate.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.in_proj_b"),
                &format!("{prefix}.ssm_alpha.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.in_proj_a"),
                &format!("{prefix}.ssm_beta.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.out_proj"),
                &format!("{prefix}.ssm_out.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.conv1d"),
                &format!("{prefix}.ssm_conv1d.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.A_log"),
                &format!("{prefix}.ssm_a"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.dt_bias"),
                &format!("{prefix}.ssm_dt.bias"),
                LayerKind::Linear,
                Some(idx),
            ));
            out.push(map(
                &format!("layers.{idx}.linear_attn.norm"),
                &format!("{prefix}.ssm_norm.weight"),
                LayerKind::Linear,
                Some(idx),
            ));
        }
    }
    out
}

fn map(role: &str, gguf_name: &str, kind: LayerKind, layer: Option<usize>) -> MappedTensor {
    MappedTensor {
        role: role.to_string(),
        gguf_name: gguf_name.to_string(),
        kind,
        layer,
    }
}

pub fn check_mapping(file: &GgufFile, config: &TextConfig) -> Result<Vec<MappedTensor>, String> {
    let expected = expected_gguf_names(config);
    let mut missing = Vec::new();
    for item in &expected {
        if file.tensor(&item.gguf_name).is_none() {
            missing.push(item.gguf_name.clone());
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "GGUF missing {} mapped tensors, first: {}",
            missing.len(),
            missing[..missing.len().min(8)].join(", ")
        ));
    }
    let qkv_out = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let z_out = config.linear_value_dim();
    let q_out = config.num_attention_heads * config.head_dim;
    let kv_out = config.num_key_value_heads * config.head_dim;
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;

    let t = file.tensor("blk.0.attn_qkv.weight").ok_or("missing blk.0 qkv")?;
    if t.dims != [hidden, qkv_out] {
        return Err(format!(
            "blk.0.attn_qkv.weight dims {:?} != [{hidden}, {qkv_out}]",
            t.dims
        ));
    }
    let t = file.tensor("blk.0.attn_gate.weight").ok_or("missing blk.0 gate")?;
    if t.dims != [hidden, z_out] {
        return Err(format!(
            "blk.0.attn_gate.weight dims {:?} != [{hidden}, {z_out}]",
            t.dims
        ));
    }
    let t = file.tensor("blk.3.attn_q.weight").ok_or("missing blk.3 q")?;
    if t.dims != [hidden, q_out * 2] {
        return Err(format!(
            "blk.3.attn_q.weight dims {:?} != [{hidden}, {}] (Q||gate)",
            t.dims,
            q_out * 2
        ));
    }
    let t = file.tensor("blk.3.attn_k.weight").ok_or("missing blk.3 k")?;
    if t.dims != [hidden, kv_out] {
        return Err(format!(
            "blk.3.attn_k.weight dims {:?} != [{hidden}, {kv_out}]",
            t.dims
        ));
    }
    let t = file.tensor("blk.0.ffn_up.weight").ok_or("missing ffn_up")?;
    if t.dims != [hidden, inter] {
        return Err(format!(
            "blk.0.ffn_up.weight dims {:?} != [{hidden}, {inter}]",
            t.dims
        ));
    }
    let embed = file.tensor("token_embd.weight").ok_or("missing embed")?;
    if embed.dims != [hidden, config.vocab_size] {
        return Err(format!(
            "token_embd.weight dims {:?} != [{hidden}, {}]",
            embed.dims, config.vocab_size
        ));
    }
    Ok(expected)
}

fn err(msg: impl Into<String>) -> model_store::Error {
    model_store::Error::Other(msg.into())
}

fn f32_to_bf16_bytes(data: &[u8]) -> Result<Vec<u8>, model_store::Error> {
    if data.len() % 4 != 0 {
        return Err(err("F32 tensor length is not a multiple of 4"));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(4) {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    Ok(out)
}

fn upload_f32(file: &GgufFile, name: &str, ordinal: usize) -> Result<GpuBuffer, model_store::Error> {
    let tensor = file.tensor(name).ok_or_else(|| err(format!("missing {name}")))?;
    if tensor.tensor_type != 0 {
        return Err(err(format!("{name} expected F32, got type {}", tensor.tensor_type)));
    }
    let data = file.tensor_bytes(name)?;
    GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &tensor.dims, data).map_err(Into::into)
}

fn upload_f32_as_bf16(
    file: &GgufFile,
    name: &str,
    ordinal: usize,
    shape: &[usize],
) -> Result<GpuBuffer, model_store::Error> {
    let data = file.tensor_bytes(name)?;
    let bf = f32_to_bf16_bytes(data)?;
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, shape, &bf).map_err(Into::into)
}

fn upload_packed(
    file: &GgufFile,
    name: &str,
    ordinal: usize,
    headers: &mut BTreeMap<String, GqhHeader>,
    header_key: &str,
) -> Result<GpuBuffer, model_store::Error> {
    let tensor = file.tensor(name).ok_or_else(|| err(format!("missing {name}")))?;
    if tensor.dims.len() != 2 {
        return Err(err(format!("{name} must be rank-2, got {:?}", tensor.dims)));
    }
    let cols = tensor.dims[0];
    let rows = tensor.dims[1];
    let data = file.tensor_bytes(name)?;
    let (upload, row_bytes) = if let Some(rung) = GqhRung::from_ggml_type(tensor.tensor_type) {
        if let Some(h) = file.gqh_header(name) {
            headers.insert(header_key.to_string(), h.clone());
        } else if rung.has_header() {
            return Err(err(format!("{name} is GQH but has no header")));
        }
        let file_row = gqh::packed_nbytes(rung, 1, cols)?;
        if data.len() != rows * file_row {
            return Err(err(format!(
                "{name} packed size {} != {rows}*{file_row}",
                data.len()
            )));
        }
        let aligned = gqh::planarize(rung, rows, cols, data)?;
        let row_bytes = gqh::device_row_bytes(rung, cols).ok_or_else(|| {
            err(format!("{name} invalid GQH device row cols={cols}"))
        })?;
        (aligned, row_bytes)
    } else {
        let qtype = match tensor.tensor_type {
            8 => LOWBIT_GGML_Q8_0,
            10 => LOWBIT_GGML_Q2_K,
            12 => LOWBIT_GGML_Q4_K,
            13 => LOWBIT_GGML_Q5_K,
            14 => LOWBIT_GGML_Q6_K,
            other => {
                return Err(err(format!(
                    "{name} has unsupported packed type {other}"
                )))
            }
        };
        let row_bytes = ggml_k_row_bytes(qtype, cols).ok_or_else(|| {
            err(format!("{name} invalid packed row for type {qtype} cols={cols}"))
        })?;
        if data.len() != rows * row_bytes {
            return Err(err(format!(
                "{name} packed size {} != {rows}*{row_bytes}",
                data.len()
            )));
        }
        (data.to_vec(), row_bytes)
    };
    let buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[rows, row_bytes], &upload)
        .map_err(model_store::Error::from)?;
    if let Some(h) = headers.get(header_key) {
        kernel_ffi::gqh::register_header(buf.as_ptr(), h.tensor_scale, h.grid_code);
    }
    Ok(buf)
}

/// 8-byte `{f32 tensor_scale, i32 grid_code}` sidecar for the 4B megakernel.
fn upload_gqh_sidecars(
    ordinal: usize,
    headers: &BTreeMap<String, GqhHeader>,
) -> Result<BTreeMap<String, GpuBuffer>, model_store::Error> {
    let mut out = BTreeMap::new();
    for (role, header) in headers {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&header.tensor_scale.to_le_bytes());
        bytes[4..8].copy_from_slice(&(i32::from(header.grid_code)).to_le_bytes());
        let buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[8], &bytes)
            .map_err(model_store::Error::from)?;
        out.insert(role.clone(), buf);
    }
    Ok(out)
}

fn load_q2k_embed(
    file: &GgufFile,
    ordinal: usize,
    hidden: usize,
    vocab: usize,
) -> Result<GpuBuffer, model_store::Error> {
    let packed = file.tensor_bytes("token_embd.weight")?;
    let row_bytes = model_store::q2k::row_bytes(hidden)?;
    if packed.len() != row_bytes * vocab {
        return Err(err(format!(
            "token_embd packed {} != {}*{}",
            packed.len(),
            row_bytes,
            vocab
        )));
    }
    let mut bf = vec![0u8; vocab * hidden * 2];
    let mut row = vec![0.0f32; hidden];
    for token in 0..vocab {
        model_store::q2k::decode_row(
            &packed[token * row_bytes..(token + 1) * row_bytes],
            hidden,
            &mut row,
        )?;
        let dst = &mut bf[token * hidden * 2..(token + 1) * hidden * 2];
        for (c, v) in row.iter().enumerate() {
            dst[c * 2..c * 2 + 2].copy_from_slice(&bf16::from_f32(*v).to_le_bytes());
        }
    }
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[vocab, hidden], &bf).map_err(Into::into)
}

fn load_a_log_exp(file: &GgufFile, name: &str, ordinal: usize) -> Result<GpuBuffer, model_store::Error> {
    let data = file.tensor_bytes(name)?;
    if data.len() % 4 != 0 {
        return Err(err(format!("{name} is not F32")));
    }
    let mut bf = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(4) {
        let v = f32::from_le_bytes(chunk.try_into().unwrap()).exp();
        bf.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[bf.len() / 2], &bf).map_err(Into::into)
}

/// Load packed Qwen3.8 GQH GGUF weights. Skips MTP `blk.64`.
pub fn load_weights(
    file: &GgufFile,
    config: &TextConfig,
    ordinal: usize,
) -> Result<Qwen35Weights, model_store::Error> {
    check_mapping(file, config).map_err(err)?;
    let hidden = config.hidden_size;
    let mut headers = BTreeMap::new();

    let embed_tokens = Arc::new(load_q2k_embed(
        file,
        ordinal,
        hidden,
        config.vocab_size,
    )?);
    let lm_head = Arc::new(upload_packed(
        file,
        "output.weight",
        ordinal,
        &mut headers,
        "lm_head.weight",
    )?);
    let norm_weight =
        upload_f32_as_bf16(file, "output_norm.weight", ordinal, &[hidden])?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for idx in 0..config.num_hidden_layers {
        let blk = format!("blk.{idx}");
        let input_norm_w =
            upload_f32_as_bf16(file, &format!("{blk}.attn_norm.weight"), ordinal, &[hidden])?;
        let post_attn_norm_w = upload_f32_as_bf16(
            file,
            &format!("{blk}.post_attention_norm.weight"),
            ordinal,
            &[hidden],
        )?;
        let gate_proj_w = upload_packed(
            file,
            &format!("{blk}.ffn_gate.weight"),
            ordinal,
            &mut headers,
            &format!("layers.{idx}.mlp.gate_proj"),
        )?;
        let up_proj_w = upload_packed(
            file,
            &format!("{blk}.ffn_up.weight"),
            ordinal,
            &mut headers,
            &format!("layers.{idx}.mlp.up_proj"),
        )?;
        let down_proj_w = upload_packed(
            file,
            &format!("{blk}.ffn_down.weight"),
            ordinal,
            &mut headers,
            &format!("layers.{idx}.mlp.down_proj"),
        )?;

        let (kind, linear, full) = if config.is_full_attention(idx) {
            let q_proj_w = upload_packed(
                file,
                &format!("{blk}.attn_q.weight"),
                ordinal,
                &mut headers,
                &format!("layers.{idx}.self_attn.q_proj"),
            )?;
            let k_proj_w = upload_packed(
                file,
                &format!("{blk}.attn_k.weight"),
                ordinal,
                &mut headers,
                &format!("layers.{idx}.self_attn.k_proj"),
            )?;
            let v_proj_w = upload_packed(
                file,
                &format!("{blk}.attn_v.weight"),
                ordinal,
                &mut headers,
                &format!("layers.{idx}.self_attn.v_proj"),
            )?;
            let o_proj_w = upload_packed(
                file,
                &format!("{blk}.attn_output.weight"),
                ordinal,
                &mut headers,
                &format!("layers.{idx}.self_attn.o_proj"),
            )?;
            let q_norm_w = Some(upload_f32_as_bf16(
                file,
                &format!("{blk}.attn_q_norm.weight"),
                ordinal,
                &[config.head_dim],
            )?);
            let k_norm_w = Some(upload_f32_as_bf16(
                file,
                &format!("{blk}.attn_k_norm.weight"),
                ordinal,
                &[config.head_dim],
            )?);
            (
                LayerKind::Full,
                None,
                Some(FullWeights {
                    q_proj_w,
                    k_proj_w,
                    v_proj_w,
                    o_proj_w,
                    q_norm_w,
                    k_norm_w,
                    q_proj_scale: None,
                    k_proj_scale: None,
                    v_proj_scale: None,
                    o_proj_scale: None,
                    q_proj_int8_scale: None,
                    k_proj_int8_scale: None,
                    v_proj_int8_scale: None,
                    o_proj_int8_scale: None,
                    q_proj_int4_scale: None,
                    q_proj_int4_zero: None,
                    q_proj_awq_inv_scale: None,
                    k_proj_int4_scale: None,
                    k_proj_int4_zero: None,
                    k_proj_awq_inv_scale: None,
                    v_proj_int4_scale: None,
                    v_proj_int4_zero: None,
                    v_proj_awq_inv_scale: None,
                    o_proj_int4_scale: None,
                    o_proj_int4_zero: None,
                    o_proj_awq_inv_scale: None,
                }),
            )
        } else {
            let qkv_out = config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim;
            let conv1d_w = upload_f32_as_bf16(
                file,
                &format!("{blk}.ssm_conv1d.weight"),
                ordinal,
                &[qkv_out, 1, config.linear_conv_kernel_dim],
            )?;
            let norm_f32 = upload_f32(file, &format!("{blk}.ssm_norm.weight"), ordinal)?;
            let norm_w_bf16 = upload_f32_as_bf16(
                file,
                &format!("{blk}.ssm_norm.weight"),
                ordinal,
                &[config.linear_key_head_dim],
            )?;
            (
                LayerKind::Linear,
                Some(LinearWeights {
                    qkv_proj_w: upload_packed(
                        file,
                        &format!("{blk}.attn_qkv.weight"),
                        ordinal,
                        &mut headers,
                        &format!("layers.{idx}.linear_attn.in_proj_qkv"),
                    )?,
                    z_proj_w: upload_packed(
                        file,
                        &format!("{blk}.attn_gate.weight"),
                        ordinal,
                        &mut headers,
                        &format!("layers.{idx}.linear_attn.in_proj_z"),
                    )?,
                    qkvz_proj_w: None,
                    b_proj_w: upload_packed(
                        file,
                        &format!("{blk}.ssm_alpha.weight"),
                        ordinal,
                        &mut headers,
                        &format!("layers.{idx}.linear_attn.in_proj_b"),
                    )?,
                    a_proj_w: upload_packed(
                        file,
                        &format!("{blk}.ssm_beta.weight"),
                        ordinal,
                        &mut headers,
                        &format!("layers.{idx}.linear_attn.in_proj_a"),
                    )?,
                    ba_proj_w: None,
                    conv1d_w,
                    out_proj_w: upload_packed(
                        file,
                        &format!("{blk}.ssm_out.weight"),
                        ordinal,
                        &mut headers,
                        &format!("layers.{idx}.linear_attn.out_proj"),
                    )?,
                    dt_bias: upload_f32_as_bf16(
                        file,
                        &format!("{blk}.ssm_dt.bias"),
                        ordinal,
                        &[config.linear_num_value_heads],
                    )?,
                    a_log_exp: load_a_log_exp(file, &format!("{blk}.ssm_a"), ordinal)?,
                    norm_w: norm_f32,
                    norm_w_bf16,
                    qkv_proj_scale: None,
                    z_proj_scale: None,
                    b_proj_scale: None,
                    a_proj_scale: None,
                    out_proj_scale: None,
                    qkv_proj_int8_scale: None,
                    z_proj_int8_scale: None,
                    b_proj_int8_scale: None,
                    a_proj_int8_scale: None,
                    out_proj_int8_scale: None,
                    qkv_proj_int4_scale: None,
                    qkv_proj_int4_zero: None,
                    qkv_proj_awq_inv_scale: None,
                    z_proj_int4_scale: None,
                    z_proj_int4_zero: None,
                    z_proj_awq_inv_scale: None,
                    out_proj_int4_scale: None,
                    out_proj_int4_zero: None,
                    out_proj_awq_inv_scale: None,
                }),
                None,
            )
        };

        layers.push(LayerWeights {
            kind,
            input_norm_w,
            post_attn_norm_w,
            gate_proj_w,
            up_proj_w,
            down_proj_w,
            gate_proj_scale: None,
            up_proj_scale: None,
            down_proj_scale: None,
            gate_proj_int8_scale: None,
            up_proj_int8_scale: None,
            down_proj_int8_scale: None,
            gate_proj_int4_scale: None,
            gate_proj_int4_zero: None,
            gate_proj_awq_inv_scale: None,
            up_proj_int4_scale: None,
            up_proj_int4_zero: None,
            up_proj_awq_inv_scale: None,
            down_proj_int4_scale: None,
            down_proj_int4_zero: None,
            down_proj_awq_inv_scale: None,
            linear,
            full,
        });
    }

    let gqh_sidecars = upload_gqh_sidecars(ordinal, &headers)?;

    Ok(Qwen35Weights {
        config: config.clone(),
        weight_prefix: "gguf".to_string(),
        embed_tokens,
        lm_head,
        lm_head_scale: None,
        lm_head_int4_scale: None,
        lm_head_int4_zero: None,
        lm_head_awq_inv_scale: None,
        norm_weight,
        layers,
        gqh_headers: headers,
        gqh_sidecars,
        is_fp8: false,
        fp8_block_size: 0,
        is_int4: false,
        int4_group_size: 0,
        is_int8: false,
        int8_baked_store: None,
        int8_outlier_threshold: 0.0,
    })
}
