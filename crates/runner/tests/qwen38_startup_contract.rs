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

fn write_config(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "text_config": {
                "vocab_size": 1,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "rms_norm_eps": 0.000001,
                "head_dim": 128
            }
        }"#,
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
