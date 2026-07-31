use std::path::Path;

use model_store::flm::{
    FlmStage3DirectWeightKind, ASSET_CHAT_TEMPLATE_UTF8, VALUE_FORMAT_SYM_INT4,
};
use model_store::manifest::LayoutTag;
use model_store::store::Int4StorageKind;
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

    let declares_row_group_codec = runtime
        .codecs()
        .iter()
        .any(|codec| codec.semantic_id as u16 == model_store::flm::CODEC_ROW_GROUP_INT4_BF16_SYM);
    let mut row_group_views = 0usize;

    for logical in runtime
        .logical_tensors()
        .iter()
        .filter(|logical| logical.value_format_id == VALUE_FORMAT_SYM_INT4)
    {
        match runtime
            .stage3_direct_weight_kind(&logical.name)
            .expect("classify native INT4 storage")
        {
            Some(FlmStage3DirectWeightKind::NativeInt4) => {
                let view = flm
                    .int4_storage_view(&logical.name)
                    .expect("tile-v1 storage view");
                assert_eq!(view.kind, Int4StorageKind::TileV1);
                assert!(view.zero_tensor.is_some());
                assert_eq!(flm.layout(&logical.name), Some(&LayoutTag::Int4Quantized));
            }
            Some(FlmStage3DirectWeightKind::RowGroupInt4) => {
                row_group_views += 1;
                let view = flm
                    .int4_storage_view(&logical.name)
                    .expect("row-group storage view");
                assert_eq!(view.kind, Int4StorageKind::RowGroupSymmetric);
                assert_eq!(view.zero_tensor, None);
                assert_eq!(view.implicit_zero_code, Some(8));
                assert_eq!(flm.layout(&logical.name), Some(&LayoutTag::Int4RowGroup));
            }
            _ => {}
        }
    }
    if declares_row_group_codec {
        assert!(
            row_group_views > 0,
            "an artifact declaring semantic codec 11 must expose at least one typed row-group view"
        );
    } else {
        assert_eq!(
            row_group_views, 0,
            "legacy canonical artifacts without semantic codec 11 must not claim row-group coverage"
        );
    }

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
