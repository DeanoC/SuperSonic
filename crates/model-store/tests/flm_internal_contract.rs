use std::fs;
use std::path::Path;

use model_store::{codec, flm, gqh};

#[test]
fn gqh_ids_are_consistent_between_gguf_and_internal_flm() {
    for ids in codec::GQH_CODEC_IDS {
        let rung = gqh::GqhRung::from_ggml_type(ids.gguf_qtype)
            .unwrap_or_else(|| panic!("missing GGUF qtype {}", ids.gguf_qtype));
        assert_eq!(rung.flm_codec(), ids.flm_codec);

        let round_trip = gqh::GqhRung::from_flm_codec(ids.flm_codec)
            .unwrap_or_else(|| panic!("missing FLM codec {}", ids.flm_codec));
        assert_eq!(round_trip.ggml_type(), ids.gguf_qtype);
    }
}

#[test]
fn internal_flm_tensor_descriptor_round_trips_without_gpu() {
    let descriptor = flm::FlmTensorDescriptor {
        tensor_id: 7,
        name: "blk.0.attn_q.weight".to_string(),
        role_id: flm::LOGICAL_TENSOR_ROLE_QUANTIZED_WEIGHT,
        rank: 2,
        shape: [4096, 5120, 0, 0],
        value_format_id: flm::VALUE_FORMAT_SYM_INT4,
        reconstruction_dtype: flm::FLM_DTYPE_BF16,
        storage_binding_start: 3,
        storage_binding_count: 2,
        flags: flm::LOGICAL_TENSOR_FLAG_REQUIRED,
    };

    let wire = descriptor.encode().expect("encode internal FLM descriptor");
    let decoded = flm::FlmTensorDescriptor::decode(&wire).expect("decode internal FLM descriptor");
    assert_eq!(decoded, descriptor);
}

#[test]
fn runner_has_no_public_flm_startup_route() {
    let runner_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runner/src");
    let mut source = String::new();
    collect_rust_source(&runner_src, &mut source);

    for forbidden in [
        "--flm-file",
        "effective_flm",
        "open_flm",
        "FlmModel",
        "model_store::flm",
        "flm_file",
    ] {
        assert!(
            !source.contains(forbidden),
            "runner source still exposes FLM startup surface: {forbidden}"
        );
    }
}

#[test]
fn model_store_has_no_legacy_bake_or_distribution_surface() {
    let model_store_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_rs =
        fs::read_to_string(model_store_root.join("src/lib.rs")).expect("read model-store lib");
    for forbidden_module in [
        "pub mod baker;",
        "pub mod fetch;",
        "pub mod manifest;",
        "pub mod store;",
        "pub mod transforms;",
    ] {
        assert!(
            !lib_rs.contains(forbidden_module),
            "model-store still exposes legacy module: {forbidden_module}"
        );
    }
    for deleted in [
        "baker.rs",
        "fetch.rs",
        "manifest.rs",
        "store.rs",
        "transforms.rs",
    ] {
        assert!(
            !model_store_root.join("src").join(deleted).exists(),
            "legacy model-store source still exists: {deleted}"
        );
    }

    let qwen38_weights = fs::read_to_string(model_store_root.join("../qwen38/src/weights.rs"))
        .expect("read qwen38 weights");
    for forbidden in ["BakedStore", "LayoutTag", "load_baked"] {
        assert!(
            !qwen38_weights.contains(forbidden),
            "Qwen3.8 weights still carry removed bake surface: {forbidden}"
        );
    }
}

fn collect_rust_source(path: &Path, out: &mut String) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("read runner source") {
            collect_rust_source(&entry.expect("runner source entry").path(), out);
        }
    } else if path.extension().is_some_and(|ext| ext == "rs") {
        out.push_str(&fs::read_to_string(path).expect("read runner Rust source"));
    }
}
