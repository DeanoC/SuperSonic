use std::fs;
use std::path::Path;

use anyhow::Result;
use model_store::gqh::GqhRung;

use crate::Cli;

const QWEN38_VOCAB_SIZE: usize = 248_320;
const QWEN38_HIDDEN_SIZE: usize = 5_120;
const QWEN38_INTERMEDIATE_SIZE: usize = 17_408;
const QWEN38_NUM_HIDDEN_LAYERS: usize = 64;
const QWEN38_NUM_ATTENTION_HEADS: usize = 24;
const QWEN38_NUM_KEY_VALUE_HEADS: usize = 4;
const QWEN38_MAX_POSITION_EMBEDDINGS: usize = 262_144;
const QWEN38_HEAD_DIM: usize = 256;
const QWEN38_LINEAR_CONV_KERNEL_DIM: usize = 4;
const QWEN38_LINEAR_KEY_HEAD_DIM: usize = 128;
const QWEN38_LINEAR_VALUE_HEAD_DIM: usize = 128;
const QWEN38_LINEAR_NUM_KEY_HEADS: usize = 16;
const QWEN38_LINEAR_NUM_VALUE_HEADS: usize = 48;
const QWEN38_ROPE_THETA: f64 = 10_000_000.0;
const QWEN38_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

const MTP_EH_PROJ: &str = "blk.64.nextn.eh_proj.weight";
const MTP_TENSORS: [&str; 15] = [
    "blk.64.nextn.enorm.weight",
    "blk.64.nextn.hnorm.weight",
    MTP_EH_PROJ,
    "blk.64.nextn.shared_head_norm.weight",
    "blk.64.attn_norm.weight",
    "blk.64.post_attention_norm.weight",
    "blk.64.ffn_gate.weight",
    "blk.64.ffn_up.weight",
    "blk.64.ffn_down.weight",
    "blk.64.attn_q.weight",
    "blk.64.attn_k.weight",
    "blk.64.attn_v.weight",
    "blk.64.attn_output.weight",
    "blk.64.attn_q_norm.weight",
    "blk.64.attn_k_norm.weight",
];

#[derive(Clone, Copy)]
enum TensorEncoding {
    F32,
    Packed,
    TokenEmbedding,
}

pub(crate) fn load_tokenizer(tokenizer_path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))
}

/// Validate every file needed by the public Qwen3.8 GQH startup contract.
///
/// This function intentionally performs only host-side reads. Callers should
/// invoke it before selecting an accelerator or allocating any GPU buffers so
/// a typo in a model directory or artifact fails with an actionable message.
pub fn validate_input_contract(cli: &Cli) -> Result<()> {
    if cli.model != "qwen3.8-27b" {
        anyhow::bail!(
            "unsupported model {:?}; the startup contract requires --model qwen3.8-27b",
            cli.model
        );
    }
    if cli.temperature != 0.0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--temperature 0)");
    }
    if cli.top_k != 0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--top-k 0)");
    }
    if cli.top_p != 1.0 {
        anyhow::bail!("Qwen3.8 inference supports greedy generation only (--top-p 1)");
    }

    let config_path = cli.model_dir.join("config.json");
    require_file(&config_path, "Qwen3.8 config.json")?;
    let config = qwen35::config::load_config(&cli.model_dir).map_err(|e| {
        anyhow::anyhow!("invalid Qwen3.8 config.json {}: {e}", config_path.display())
    })?;
    validate_qwen38_geometry(&config.text_config)?;

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    require_file(&tokenizer_path, "Qwen3.8 tokenizer data")?;
    load_tokenizer(&tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "invalid Qwen3.8 tokenizer data {}: {e}",
            tokenizer_path.display()
        )
    })?;

    if cli.chat {
        let tokenizer_config_path = cli.model_dir.join("tokenizer_config.json");
        require_file(&tokenizer_config_path, "Qwen3.8 chat-template metadata")?;
        supersonic_runtime::chat_template::ChatTemplate::try_load(&cli.model_dir)
            .map_err(|e| {
                anyhow::anyhow!(
                    "invalid Qwen3.8 chat-template metadata {}: {e}",
                    tokenizer_config_path.display()
                )
            })?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.8 chat-template metadata {} does not define a chat template",
                    tokenizer_config_path.display()
                )
            })?;
    }

    let gguf_path = cli.gguf_file.as_deref().ok_or_else(|| {
        anyhow::anyhow!("missing Qwen3.8 GQH GGUF artifact: --gguf-file is required")
    })?;
    require_file(gguf_path, "Qwen3.8 GQH GGUF artifact")?;
    let gguf = model_store::gguf::GgufFile::open(gguf_path).map_err(|e| {
        anyhow::anyhow!(
            "invalid Qwen3.8 GQH GGUF artifact {}: {e}",
            gguf_path.display()
        )
    })?;

    if gguf.gqh_header_count() == 0 {
        anyhow::bail!(
            "GGUF {} is not a custom GQH artifact: required GQH headers are absent",
            gguf_path.display()
        );
    }
    match gguf.kv("general.architecture") {
        Some("qwen35") => {}
        Some(architecture) => anyhow::bail!(
            "GQH GGUF {} has unsupported architecture {:?}; expected qwen35 for Qwen3.8",
            gguf_path.display(),
            architecture
        ),
        None => anyhow::bail!(
            "GQH GGUF {} is missing required general.architecture metadata",
            gguf_path.display()
        ),
    }

    if let Some(basename) = gguf.kv("general.basename") {
        if basename != "qwen38" {
            anyhow::bail!(
                "GQH GGUF {} has unsupported basename {:?}; expected qwen38 for Qwen3.8-27B",
                gguf_path.display(),
                basename
            );
        }
    }

    validate_qwen38_artifact(&gguf, &config.text_config, cli.speculative_decode).map_err(|e| {
        anyhow::anyhow!(
            "GQH GGUF {} is incompatible with Qwen3.8 geometry or qtypes: {e}",
            gguf_path.display()
        )
    })?;

    qwen35::gguf_ingest::check_mapping(&gguf, &config.text_config).map_err(|e| {
        anyhow::anyhow!(
            "GQH GGUF {} is incompatible with Qwen3.8 geometry or qtypes: {e}",
            gguf_path.display()
        )
    })?;
    Ok(())
}

fn validate_qwen38_geometry(config: &qwen35::config::TextConfig) -> Result<()> {
    let checks = [
        ("vocab_size", config.vocab_size, QWEN38_VOCAB_SIZE),
        ("hidden_size", config.hidden_size, QWEN38_HIDDEN_SIZE),
        (
            "intermediate_size",
            config.intermediate_size,
            QWEN38_INTERMEDIATE_SIZE,
        ),
        (
            "num_hidden_layers",
            config.num_hidden_layers,
            QWEN38_NUM_HIDDEN_LAYERS,
        ),
        (
            "num_attention_heads",
            config.num_attention_heads,
            QWEN38_NUM_ATTENTION_HEADS,
        ),
        (
            "num_key_value_heads",
            config.num_key_value_heads,
            QWEN38_NUM_KEY_VALUE_HEADS,
        ),
        (
            "max_position_embeddings",
            config.max_position_embeddings,
            QWEN38_MAX_POSITION_EMBEDDINGS,
        ),
        ("head_dim", config.head_dim, QWEN38_HEAD_DIM),
        (
            "linear_conv_kernel_dim",
            config.linear_conv_kernel_dim,
            QWEN38_LINEAR_CONV_KERNEL_DIM,
        ),
        (
            "linear_key_head_dim",
            config.linear_key_head_dim,
            QWEN38_LINEAR_KEY_HEAD_DIM,
        ),
        (
            "linear_value_head_dim",
            config.linear_value_head_dim,
            QWEN38_LINEAR_VALUE_HEAD_DIM,
        ),
        (
            "linear_num_key_heads",
            config.linear_num_key_heads,
            QWEN38_LINEAR_NUM_KEY_HEADS,
        ),
        (
            "linear_num_value_heads",
            config.linear_num_value_heads,
            QWEN38_LINEAR_NUM_VALUE_HEADS,
        ),
    ];
    for (name, actual, expected) in checks {
        if actual != expected {
            anyhow::bail!(
                "unsupported Qwen3.8-27B geometry: {name}={actual}, expected fixed value {expected}"
            );
        }
    }
    if (config.rms_norm_eps - 1e-6).abs() > f64::EPSILON {
        anyhow::bail!(
            "unsupported Qwen3.8-27B geometry: rms_norm_eps={}, expected 0.000001",
            config.rms_norm_eps
        );
    }
    if config.layer_types.len() != QWEN38_NUM_HIDDEN_LAYERS {
        anyhow::bail!(
            "unsupported Qwen3.8-27B geometry: layer_types has {}, expected {} entries",
            config.layer_types.len(),
            QWEN38_NUM_HIDDEN_LAYERS
        );
    }
    for (idx, layer_type) in config.layer_types.iter().enumerate() {
        let expected = if (idx + 1) % 4 == 0 {
            "full_attention"
        } else {
            "linear_attention"
        };
        if layer_type != expected {
            anyhow::bail!(
                "unsupported Qwen3.8-27B geometry: layer_types[{idx}]={layer_type:?}, expected {expected:?}"
            );
        }
    }
    if (config.rope_theta() - QWEN38_ROPE_THETA).abs() > f64::EPSILON {
        anyhow::bail!(
            "unsupported Qwen3.8-27B geometry: rope_theta={}, expected {QWEN38_ROPE_THETA}",
            config.rope_theta()
        );
    }
    if (config.partial_rotary_factor() - QWEN38_PARTIAL_ROTARY_FACTOR).abs() > f64::EPSILON {
        anyhow::bail!(
            "unsupported Qwen3.8-27B geometry: partial_rotary_factor={}, expected {QWEN38_PARTIAL_ROTARY_FACTOR}",
            config.partial_rotary_factor()
        );
    }
    if config
        .rope_parameters
        .as_ref()
        .map(|params| params.rope_type.as_str())
        .unwrap_or("default")
        != "default"
    {
        anyhow::bail!("unsupported Qwen3.8-27B geometry: rope_type must be \"default\"");
    }
    Ok(())
}

fn validate_qwen38_artifact(
    file: &model_store::gguf::GgufFile,
    config: &qwen35::config::TextConfig,
    speculative_decode: bool,
) -> Result<()> {
    validate_gqh_header_inventory(file)?;
    validate_mtp_block(file, config, speculative_decode)?;

    let expected = qwen35::gguf_ingest::expected_gguf_names(config);
    let mut missing = Vec::new();
    for item in expected {
        let Some((dims, encoding)) = expected_tensor_contract(&item.gguf_name, config) else {
            anyhow::bail!("unsupported mapped tensor role {}", item.gguf_name);
        };
        if file.tensor(&item.gguf_name).is_none() {
            missing.push(item.gguf_name);
            continue;
        }
        validate_tensor(file, &item.gguf_name, &dims, encoding)?;
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "GGUF missing {} mapped tensors, first: {}",
            missing.len(),
            missing[..missing.len().min(8)].join(", ")
        );
    }
    Ok(())
}

fn validate_gqh_header_inventory(file: &model_store::gguf::GgufFile) -> Result<()> {
    let mut required_headers = 0usize;
    for name in file.tensor_names() {
        let tensor = file
            .tensor(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor metadata for {name}"))?;
        if let Some(rung) = GqhRung::from_ggml_type(tensor.tensor_type) {
            if rung.has_header() {
                required_headers += 1;
                if file.gqh_header(name).is_none() {
                    anyhow::bail!(
                        "{name} GQH qtype {} is missing its required GQH header",
                        tensor.tensor_type
                    );
                }
            } else if file.gqh_header(name).is_some() {
                anyhow::bail!(
                    "{name} GQH qtype {} must not have a GQH header",
                    tensor.tensor_type
                );
            }
        } else if file.gqh_header(name).is_some() {
            anyhow::bail!(
                "{name} qtype {} has an unexpected GQH header",
                tensor.tensor_type
            );
        }
    }
    if file.gqh_header_count() != required_headers {
        anyhow::bail!(
            "GGUF contains {} GQH headers but {} mapped qtypes require headers",
            file.gqh_header_count(),
            required_headers
        );
    }
    Ok(())
}

fn expected_tensor_contract(
    name: &str,
    config: &qwen35::config::TextConfig,
) -> Option<(Vec<usize>, TensorEncoding)> {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let q_out = config.num_attention_heads * config.head_dim;
    let kv_out = config.num_key_value_heads * config.head_dim;
    let qkv_out = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let z_out = config.linear_value_dim();
    let value_dim = config.linear_value_dim();
    let dims = match name {
        "token_embd.weight" => {
            return Some((
                vec![hidden, config.vocab_size],
                TensorEncoding::TokenEmbedding,
            ));
        }
        "output_norm.weight" => {
            return Some((vec![hidden], TensorEncoding::F32));
        }
        "output.weight" => {
            return Some((vec![hidden, config.vocab_size], TensorEncoding::Packed));
        }
        _ => {
            let rest = name.strip_prefix("blk.")?;
            let (layer, suffix) = rest.split_once('.')?;
            let layer = layer.parse::<usize>().ok()?;
            if layer >= config.num_hidden_layers {
                return None;
            }
            match suffix {
                "attn_norm.weight" | "post_attention_norm.weight" => {
                    return Some((vec![hidden], TensorEncoding::F32));
                }
                "ffn_gate.weight" | "ffn_up.weight" => vec![hidden, inter],
                "ffn_down.weight" => vec![inter, hidden],
                "attn_qkv.weight" => vec![hidden, qkv_out],
                "attn_gate.weight" => vec![hidden, z_out],
                "ssm_beta.weight" | "ssm_alpha.weight" => {
                    vec![hidden, config.linear_num_value_heads]
                }
                "ssm_out.weight" => vec![value_dim, hidden],
                "ssm_conv1d.weight" => {
                    return Some((
                        vec![config.linear_conv_kernel_dim, qkv_out],
                        TensorEncoding::F32,
                    ));
                }
                "ssm_a" | "ssm_dt.bias" => {
                    return Some((vec![config.linear_num_value_heads], TensorEncoding::F32));
                }
                "ssm_norm.weight" => {
                    return Some((vec![config.linear_key_head_dim], TensorEncoding::F32));
                }
                "attn_q.weight" => vec![hidden, q_out * 2],
                "attn_k.weight" | "attn_v.weight" => vec![hidden, kv_out],
                "attn_output.weight" => vec![q_out, hidden],
                "attn_q_norm.weight" | "attn_k_norm.weight" => {
                    return Some((vec![config.head_dim], TensorEncoding::F32));
                }
                _ => return None,
            }
        }
    };
    Some((dims, TensorEncoding::Packed))
}

fn validate_tensor(
    file: &model_store::gguf::GgufFile,
    name: &str,
    expected_dims: &[usize],
    encoding: TensorEncoding,
) -> Result<()> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("missing mapped tensor {name}"))?;
    match encoding {
        TensorEncoding::F32 => {
            if tensor.tensor_type != 0 {
                anyhow::bail!("{name} is an F32 role but has qtype {}", tensor.tensor_type);
            }
            if file.gqh_header(name).is_some() || file.mix_header(name).is_some() {
                anyhow::bail!("{name} is an F32 role but has a quantization sidecar");
            }
            validate_wire_size(file, name, tensor.tensor_type)?;
        }
        TensorEncoding::TokenEmbedding => {
            if tensor.tensor_type != 10 && tensor.tensor_type != 11 {
                anyhow::bail!(
                    "{name} token embedding has unsupported qtype {}; expected Q2_K (10) or Q3_K (11)",
                    tensor.tensor_type
                );
            }
            if file.gqh_header(name).is_some() || file.mix_header(name).is_some() {
                anyhow::bail!("{name} token embedding must not have a GQH or mix sidecar");
            }
            if tensor.dims.len() != 2 {
                anyhow::bail!(
                    "{name} token embedding must be rank-2, got {:?}",
                    tensor.dims
                );
            }
            validate_wire_size(file, name, tensor.tensor_type)?;
        }
        TensorEncoding::Packed => validate_packed_tensor(file, name)?,
    }
    if tensor.dims != expected_dims {
        anyhow::bail!(
            "{name} dimensions {:?} do not match fixed Qwen3.8 geometry {:?}",
            tensor.dims,
            expected_dims
        );
    }
    Ok(())
}

fn validate_packed_tensor(file: &model_store::gguf::GgufFile, name: &str) -> Result<()> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("missing packed tensor {name}"))?;
    if tensor.dims.len() != 2 {
        anyhow::bail!("{name} packed tensor must be rank-2, got {:?}", tensor.dims);
    }
    let qtype = tensor.tensor_type;
    if let Some(rung) = GqhRung::from_ggml_type(qtype) {
        let header = file.gqh_header(name);
        if rung.has_header() && header.is_none() {
            anyhow::bail!("{name} GQH qtype {qtype} is missing its required GQH header");
        }
        if !rung.has_header() && header.is_some() {
            anyhow::bail!("{name} GQH qtype {qtype} must not have a GQH header");
        }
        if let Some(header) = header {
            if header.qtype != qtype {
                anyhow::bail!(
                    "{name} tensor qtype {qtype} disagrees with GQH header qtype {}",
                    header.qtype
                );
            }
        }
        let want = model_store::gqh::packed_nbytes(rung, tensor.dims[1], tensor.dims[0])?;
        if tensor.nbytes != want {
            anyhow::bail!(
                "{name} GQH wire size {} does not match qtype {qtype} geometry {want}",
                tensor.nbytes
            );
        }
        return Ok(());
    }
    if qtype == model_store::dmix2::GGML_TYPE_Q3_1_ROCMFP3_MIX
        || qtype == model_store::dmix2::GGML_TYPE_Q2_1_ROCMFP2_MIX
    {
        if file.gqh_header(name).is_some() {
            anyhow::bail!("{name} mix qtype {qtype} must not have a GQH header");
        }
        let header = file.mix_header(name).ok_or_else(|| {
            anyhow::anyhow!("{name} mix qtype {qtype} is missing its mix sidecar")
        })?;
        if header.qtype != qtype {
            anyhow::bail!(
                "{name} tensor qtype {qtype} disagrees with mix sidecar qtype {}",
                header.qtype
            );
        }
        validate_wire_size(file, name, qtype)?;
        return Ok(());
    }
    if !matches!(qtype, 8 | 10 | 12 | 13 | 14) {
        anyhow::bail!(
            "{name} has unsupported packed qtype {qtype}; expected Qwen3.8 GQH/K-quant encoding"
        );
    }
    if file.gqh_header(name).is_some() || file.mix_header(name).is_some() {
        anyhow::bail!("{name} standard packed qtype {qtype} has an unexpected sidecar");
    }
    validate_wire_size(file, name, qtype)
}

fn validate_wire_size(file: &model_store::gguf::GgufFile, name: &str, qtype: u32) -> Result<()> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
    let want = if let Some(rung) = GqhRung::from_ggml_type(qtype) {
        model_store::gqh::packed_nbytes(rung, tensor.dims[1], tensor.dims[0])?
    } else if qtype == model_store::dmix2::GGML_TYPE_Q3_1_ROCMFP3_MIX
        || qtype == model_store::dmix2::GGML_TYPE_Q2_1_ROCMFP2_MIX
    {
        model_store::dmix2::row_bytes(qtype, tensor.dims[0])?
            .checked_mul(tensor.dims[1])
            .ok_or_else(|| anyhow::anyhow!("{name} packed byte length overflows"))?
    } else if qtype == 0 {
        tensor
            .dims
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("{name} F32 byte length overflows"))?
    } else if qtype == 11 {
        model_store::q3k::row_bytes(tensor.dims[0])?
            .checked_mul(tensor.dims[1])
            .ok_or_else(|| anyhow::anyhow!("{name} Q3_K byte length overflows"))?
    } else {
        qwen35::weights::ggml_k_row_bytes(qtype as i32, tensor.dims[0])
            .and_then(|row| row.checked_mul(tensor.dims[1]))
            .ok_or_else(|| anyhow::anyhow!("{name} qtype {qtype} has invalid packed geometry"))?
    };
    if tensor.nbytes != want {
        anyhow::bail!(
            "{name} wire size {} does not match qtype {qtype} geometry {want}",
            tensor.nbytes
        );
    }
    Ok(())
}

fn validate_mtp_block(
    file: &model_store::gguf::GgufFile,
    config: &qwen35::config::TextConfig,
    speculative_decode: bool,
) -> Result<()> {
    let present = MTP_TENSORS
        .iter()
        .filter(|name| file.tensor(name).is_some())
        .count();
    if present == 0 {
        if speculative_decode {
            anyhow::bail!(
                "--speculative-decode requires the complete Qwen3.8 NextN MTP block, including {MTP_EH_PROJ}"
            );
        }
        return Ok(());
    }
    let missing: Vec<_> = MTP_TENSORS
        .iter()
        .filter(|name| file.tensor(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "incomplete Qwen3.8 NextN MTP block: missing {}",
            missing.join(", ")
        );
    }

    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let q_out = config.num_attention_heads * config.head_dim;
    let kv_out = config.num_key_value_heads * config.head_dim;
    let specs = [
        (
            "blk.64.nextn.enorm.weight",
            vec![hidden],
            TensorEncoding::F32,
        ),
        (
            "blk.64.nextn.hnorm.weight",
            vec![hidden],
            TensorEncoding::F32,
        ),
        (
            MTP_EH_PROJ,
            vec![hidden * 2, hidden],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.nextn.shared_head_norm.weight",
            vec![hidden],
            TensorEncoding::F32,
        ),
        ("blk.64.attn_norm.weight", vec![hidden], TensorEncoding::F32),
        (
            "blk.64.post_attention_norm.weight",
            vec![hidden],
            TensorEncoding::F32,
        ),
        (
            "blk.64.ffn_gate.weight",
            vec![hidden, inter],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.ffn_up.weight",
            vec![hidden, inter],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.ffn_down.weight",
            vec![inter, hidden],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.attn_q.weight",
            vec![hidden, q_out * 2],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.attn_k.weight",
            vec![hidden, kv_out],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.attn_v.weight",
            vec![hidden, kv_out],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.attn_output.weight",
            vec![q_out, hidden],
            TensorEncoding::Packed,
        ),
        (
            "blk.64.attn_q_norm.weight",
            vec![config.head_dim],
            TensorEncoding::F32,
        ),
        (
            "blk.64.attn_k_norm.weight",
            vec![config.head_dim],
            TensorEncoding::F32,
        ),
    ];
    for (name, dims, encoding) in specs {
        validate_tensor(file, name, &dims, encoding)?;
    }
    Ok(())
}

fn require_file(path: &Path, role: &str) -> Result<()> {
    let metadata = fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("missing {role} at {}: {e}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{role} at {} is not a regular file", path.display());
    }
    fs::File::open(path)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{role} at {} is not readable: {e}", path.display()))
}
