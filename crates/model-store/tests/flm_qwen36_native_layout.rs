use std::path::Path;

use model_store::flm::ASSET_CHAT_TEMPLATE_UTF8;
use model_store::manifest::LayoutTag;
use model_store::{BakedStore, FlmLoadOptions};

#[test]
fn qwen36_35b_native_flm_linear_attention_aliases_match_supersonic_runtime_layout() {
    let Ok(flm_path) = std::env::var("SUPERSONIC_QWEN36_35B_NATIVE_INT4_FLM") else {
        eprintln!("skip: SUPERSONIC_QWEN36_35B_NATIVE_INT4_FLM unset");
        return;
    };
    let Ok(bake_dir) = std::env::var("SUPERSONIC_QWEN36_35B_NATIVE_INT4_BAKE_DIR") else {
        eprintln!("skip: SUPERSONIC_QWEN36_35B_NATIVE_INT4_BAKE_DIR unset");
        return;
    };

    let flm = BakedStore::open_flm_with_options(
        Path::new(&flm_path),
        FlmLoadOptions {
            flm_int4_logical_aliases: true,
            verify_block_hashes: false,
        },
    )
    .expect("open native Qwen3.6 MoE FLM");
    let bake = BakedStore::open(Path::new(&bake_dir)).expect("open SuperSonic native bake");

    let runtime = flm.flm_runtime().expect("FLM runtime directory");
    let template = runtime
        .required_chat_template_source()
        .expect("native chat template asset");
    assert!(
        !template.trim().is_empty(),
        "native chat template source must be non-empty"
    );
    assert_eq!(
        runtime
            .assets
            .values()
            .filter(|asset| asset.kind_id == ASSET_CHAT_TEMPLATE_UTF8)
            .count(),
        1,
        "native runtime must contain exactly one chat template asset"
    );

    for (name, expected_layout, expected_shape) in [
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            LayoutTag::DepthwiseConvSqueezed,
            vec![8192, 4],
        ),
        (
            "model.language_model.layers.0.linear_attn.dt_bias",
            LayoutTag::HeadBiasReshaped,
            vec![1, 1, 32],
        ),
        (
            "model.language_model.layers.0.linear_attn.A_log",
            LayoutTag::HeadExpReshaped,
            vec![1, 1, 32],
        ),
    ] {
        let flm_meta = flm
            .meta(name)
            .unwrap_or_else(|| panic!("FLM missing {name}"));
        let bake_meta = bake
            .meta(name)
            .unwrap_or_else(|| panic!("bake missing {name}"));

        assert_eq!(bake_meta.layout, expected_layout, "bake layout for {name}");
        assert_eq!(bake_meta.shape, expected_shape, "bake shape for {name}");
        assert_eq!(flm_meta.layout, expected_layout, "FLM layout for {name}");
        assert_eq!(flm_meta.shape, expected_shape, "FLM shape for {name}");
        assert_eq!(
            flm.raw_bytes(name),
            bake.raw_bytes(name),
            "FLM bytes should match SuperSonic runtime bake for {name}",
        );
    }
}
