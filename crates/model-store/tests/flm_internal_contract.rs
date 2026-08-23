use std::fs;
use std::path::Path;

use model_store::gqh;

#[test]
fn public_gqh_mapping_covers_all_product_qtypes() {
    for (qtype, expected) in [(108, "Gqh3"), (109, "Gqh2H"), (110, "Gqh2C"), (111, "Gqh4")] {
        let rung = gqh::GqhRung::from_ggml_type(qtype)
            .unwrap_or_else(|| panic!("missing public GGUF qtype {qtype}"));
        assert_eq!(format!("{rung:?}"), expected);
        assert_eq!(rung.ggml_type(), qtype);
    }
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

#[test]
fn flm_foundations_and_flm_terms_are_not_public_api() {
    let model_store_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_rs =
        fs::read_to_string(model_store_root.join("src/lib.rs")).expect("read model-store lib");
    assert!(lib_rs.contains("mod codec;"));
    assert!(lib_rs.contains("mod flm;"));
    assert!(!lib_rs.contains("pub mod codec;"));
    assert!(!lib_rs.contains("pub mod flm;"));

    let gqh_rs = fs::read_to_string(model_store_root.join("src/gqh.rs")).expect("read GQH source");
    assert!(!gqh_rs.contains("pub fn from_flm_codec"));
    assert!(!gqh_rs.contains("pub fn flm_codec"));

    let flm_rs =
        fs::read_to_string(model_store_root.join("src/flm.rs")).expect("read internal FLM source");
    assert!(!flm_rs.contains("FTD1"));
    assert!(!flm_rs.contains("pub type FlmTensorDescriptor"));
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
