use std::path::PathBuf;

use supersonic_runtime::dflash::{parse_tap_layers, DFlashOptions};
use supersonic_runtime::registry::ModelVariant;

#[test]
fn parses_comma_separated_tap_layers() {
    assert_eq!(parse_tap_layers("1, 8,15", 32).unwrap(), vec![1, 8, 15]);
    assert!(parse_tap_layers("", 32).is_err());
    assert!(parse_tap_layers("1,32", 32).is_err());
}

#[test]
fn dflash_options_pick_model_appropriate_default_block_size() {
    let options = DFlashOptions {
        draft_dir: PathBuf::from("/tmp/dflash"),
        block: None,
        tap_layers: None,
    };

    assert_eq!(
        options
            .effective_block_size(&ModelVariant::Qwen3_6_27B, 16)
            .unwrap(),
        16
    );
    assert_eq!(
        options
            .effective_block_size(&ModelVariant::Qwen3_5_9B, 16)
            .unwrap(),
        3
    );
}

#[test]
fn dflash_options_validate_block_override_against_draft_block() {
    let options = DFlashOptions {
        draft_dir: PathBuf::from("/tmp/dflash"),
        block: Some(8),
        tap_layers: None,
    };
    assert_eq!(
        options
            .effective_block_size(&ModelVariant::Qwen3_6_27B, 16)
            .unwrap(),
        8
    );

    let options = DFlashOptions {
        block: Some(17),
        ..options
    };
    assert!(options
        .effective_block_size(&ModelVariant::Qwen3_6_27B, 16)
        .is_err());
}
