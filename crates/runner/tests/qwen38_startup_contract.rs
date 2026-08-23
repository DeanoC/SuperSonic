use std::fs;
use std::path::Path;

use runner::{parse_cli_from, validate_input_contract};
use tempfile::TempDir;

fn cli(model_dir: &Path, gguf_file: &Path) -> runner::Cli {
    parse_cli_from([
        "supersonic".to_string(),
        "--model".to_string(),
        "qwen3.8-27b".to_string(),
        "--model-dir".to_string(),
        model_dir.display().to_string(),
        "--gguf-file".to_string(),
        gguf_file.display().to_string(),
    ])
    .expect("test CLI should parse")
}

fn cli_with_speculative(model_dir: &Path, gguf_file: &Path) -> runner::Cli {
    parse_cli_from([
        "supersonic".to_string(),
        "--model".to_string(),
        "qwen3.8-27b".to_string(),
        "--model-dir".to_string(),
        model_dir.display().to_string(),
        "--gguf-file".to_string(),
        gguf_file.display().to_string(),
        "--speculative-decode".to_string(),
    ])
    .expect("test CLI should parse")
}

fn write_config(model_dir: &Path) {
    write_config_with_hidden(model_dir, 5120);
}

fn write_config_with_hidden(model_dir: &Path, hidden_size: usize) {
    let layer_types: Vec<&str> = (0..64)
        .map(|idx| {
            if (idx + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let config = serde_json::json!({
        "text_config": {
            "vocab_size": 248320,
            "hidden_size": hidden_size,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 262144,
            "rms_norm_eps": 0.000001,
            "tie_word_embeddings": false,
            "eos_token_id": 248044,
            "head_dim": 256,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "layer_types": layer_types,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            }
        }
    });
    fs::write(
        model_dir.join("config.json"),
        serde_json::to_vec(&config).expect("serialize config.json"),
    )
    .expect("write config.json");
}

fn write_tokenizer(model_dir: &Path) {
    fs::write(
        model_dir.join("tokenizer.json"),
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"hello": 0},
                "unk_token": "[UNK]"
            }
        }"#,
    )
    .expect("write tokenizer.json");
}

fn write_non_gqh_gguf(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    fs::write(path, bytes).expect("write non-GQH GGUF");
}

struct FixtureTensor<'a> {
    name: &'a str,
    dims: Vec<u64>,
    tensor_type: u32,
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_gqh_gguf(path: &Path, tensors: &[FixtureTensor<'_>], headers: &[(&str, u32)]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());

    write_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    write_string(&mut bytes, "qwen35");
    write_string(&mut bytes, "general.basename");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    write_string(&mut bytes, "qwen38");
    write_string(&mut bytes, model_store::gqh::GQH_HEADERS_KV);
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    let mut header_blob = Vec::new();
    header_blob.extend_from_slice(b"GQHh1\0\0\0");
    header_blob.extend_from_slice(&(headers.len() as u32).to_le_bytes());
    header_blob.extend_from_slice(&0u32.to_le_bytes());
    for (name, qtype) in headers {
        header_blob.extend_from_slice(&(name.len() as u32).to_le_bytes());
        header_blob.extend_from_slice(name.as_bytes());
        header_blob.extend_from_slice(&qtype.to_le_bytes());
        header_blob.extend_from_slice(&1.0f32.to_le_bytes());
        header_blob.push(0);
        header_blob.extend_from_slice(&[0, 0, 0]);
    }
    bytes.extend_from_slice(&(header_blob.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header_blob);

    let mut offset = 0usize;
    let mut payloads = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let dims: Vec<usize> = tensor.dims.iter().map(|dim| *dim as usize).collect();
        let nbytes = model_store::gguf::tensor_nbytes(&dims, tensor.tensor_type)
            .expect("fixture tensor type and shape");
        write_string(&mut bytes, tensor.name);
        bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor.tensor_type.to_le_bytes());
        bytes.extend_from_slice(&(offset as u64).to_le_bytes());
        payloads.push(nbytes);
        offset = (offset + nbytes + 31) & !31;
    }

    let data_offset = (bytes.len() + 31) & !31;
    bytes.resize(data_offset, 0);
    let mut payload_offset = 0usize;
    for nbytes in payloads {
        let start = data_offset + payload_offset;
        if bytes.len() < start + nbytes {
            bytes.resize(start + nbytes, 0);
        }
        payload_offset = (payload_offset + nbytes + 31) & !31;
    }
    fs::write(path, bytes).expect("write GQH fixture");
}

fn gqh_tensor<'a>(name: &'a str, dims: &[u64], tensor_type: u32) -> FixtureTensor<'a> {
    FixtureTensor {
        name,
        dims: dims.to_vec(),
        tensor_type,
    }
}

#[test]
fn rejects_missing_config_with_required_artifact_role() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    let gguf = temp.path().join("weights.gqh.gguf");

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("missing config must fail preflight")
        .to_string();

    assert!(
        error.contains(&model_dir.join("config.json").display().to_string()),
        "{error}"
    );
    assert!(error.contains("config.json"), "{error}");
}

#[test]
fn rejects_missing_tokenizer_data_with_required_artifact_role() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    let gguf = temp.path().join("weights.gqh.gguf");

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("missing tokenizer data must fail preflight")
        .to_string();

    assert!(
        error.contains(&model_dir.join("tokenizer.json").display().to_string()),
        "{error}"
    );
    assert!(error.contains("tokenizer"), "{error}");
}

#[test]
fn rejects_missing_gguf_with_required_artifact_role() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("weights.gqh.gguf");

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("missing GGUF must fail preflight")
        .to_string();

    assert!(error.contains(&gguf.display().to_string()), "{error}");
    assert!(error.contains("GGUF"), "{error}");
}

#[test]
fn rejects_readable_non_gqh_gguf_with_required_artifact_role() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("weights.gguf");
    write_non_gqh_gguf(&gguf);

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("non-GQH GGUF must fail preflight")
        .to_string();

    assert!(error.contains(&gguf.display().to_string()), "{error}");
    assert!(error.contains("GQH"), "{error}");
}

#[test]
fn rejects_self_consistent_but_non_qwen38_geometry() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config_with_hidden(&model_dir, 256);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("weights.gqh.gguf");

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("non-Qwen3.8 geometry must fail before artifact use")
        .to_string();

    assert!(error.contains("hidden_size"), "{error}");
    assert!(error.contains("Qwen3.8-27B"), "{error}");
}

#[test]
fn rejects_gqh_header_tensor_qtype_mismatch_before_mapping() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("mismatch.gguf");
    write_gqh_gguf(
        &gguf,
        &[gqh_tensor("output.weight", &[256, 1], 109)],
        &[("output.weight", 108)],
    );

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("header/tensor qtype mismatch must fail preflight")
        .to_string();

    assert!(error.contains("output.weight"), "{error}");
    assert!(error.contains("qtype"), "{error}");
}

#[test]
fn rejects_gqh_tensor_without_required_header_before_mapping() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("missing-header.gguf");
    write_gqh_gguf(
        &gguf,
        &[
            gqh_tensor("output.weight", &[256, 1], 108),
            gqh_tensor("blk.0.ffn_gate.weight", &[256, 1], 109),
        ],
        &[("blk.0.ffn_gate.weight", 109)],
    );

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("GQH tensor without its header must fail preflight")
        .to_string();

    assert!(error.contains("output.weight"), "{error}");
    assert!(error.contains("missing its required GQH header"), "{error}");
}

#[test]
fn rejects_packed_tensor_in_f32_role_before_mapping() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("f32-role.gguf");
    write_gqh_gguf(
        &gguf,
        &[gqh_tensor("output_norm.weight", &[256, 1], 108)],
        &[("output_norm.weight", 108)],
    );

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("F32 role with packed qtype must fail preflight")
        .to_string();

    assert!(error.contains("output_norm.weight"), "{error}");
    assert!(error.contains("F32"), "{error}");
}

#[test]
fn rejects_speculative_decode_without_nextn_block() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("without-mtp.gguf");
    write_gqh_gguf(
        &gguf,
        &[gqh_tensor("output.weight", &[256, 1], 108)],
        &[("output.weight", 108)],
    );

    let error = validate_input_contract(&cli_with_speculative(&model_dir, &gguf))
        .expect_err("speculative decode must require NextN tensors")
        .to_string();

    assert!(error.contains("speculative-decode"), "{error}");
    assert!(error.contains("NextN"), "{error}");
}

#[test]
fn rejects_incomplete_optional_nextn_block() {
    let temp = TempDir::new().expect("tempdir");
    let model_dir = temp.path().join("model");
    fs::create_dir(&model_dir).expect("model dir");
    write_config(&model_dir);
    write_tokenizer(&model_dir);
    let gguf = temp.path().join("incomplete-mtp.gguf");
    write_gqh_gguf(
        &gguf,
        &[gqh_tensor("blk.64.nextn.eh_proj.weight", &[256, 1], 108)],
        &[("blk.64.nextn.eh_proj.weight", 108)],
    );

    let error = validate_input_contract(&cli(&model_dir, &gguf))
        .expect_err("incomplete optional NextN block must fail preflight")
        .to_string();

    assert!(error.contains("NextN"), "{error}");
    assert!(error.contains("incomplete"), "{error}");
}

#[test]
fn accepts_real_qwen38_artifact_when_available() {
    let model_dir = Path::new("/data/models/Qwen3.8-27B");
    let gguf = Path::new("/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf");
    if !model_dir.join("config.json").is_file()
        || !model_dir.join("tokenizer.json").is_file()
        || !gguf.is_file()
    {
        return;
    }

    validate_input_contract(&cli(model_dir, gguf))
        .expect("the approved Qwen3.8 GQH artifact must pass host preflight");
}

#[test]
fn accepts_q3k_token_embedding_variant_when_available() {
    let model_dir = Path::new("/data/models/Qwen3.8-27B");
    let gguf = Path::new("/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq-8192.gguf");
    if !model_dir.join("config.json").is_file()
        || !model_dir.join("tokenizer.json").is_file()
        || !gguf.is_file()
    {
        return;
    }

    validate_input_contract(&cli(model_dir, gguf))
        .expect("the Q3_K token-embedding Qwen3.8 variant must pass host preflight");
}
