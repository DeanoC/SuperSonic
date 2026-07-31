use serde_json::Value;
use supersonic_runtime::qwen36_moe::weights::dequant_int4_to_bf16_bytes;

const FIXTURE: &str =
    include_str!("../../../oracle/fixtures/qwen36_native_int4_v1_known_bytes.json");

fn hex_bytes(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0, "hex fixture must contain byte pairs");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).expect("ASCII hex");
            u8::from_str_radix(pair, 16).expect("valid fixture hex")
        })
        .collect()
}

fn hex_u16(value: &Value) -> u16 {
    u16::from_str_radix(value.as_str().expect("hex string"), 16).expect("valid u16 fixture hex")
}

fn output_bits(decoded: &[u8], row: usize, col: usize, cols: usize) -> u16 {
    let offset = (row * cols + col) * 2;
    u16::from_le_bytes([decoded[offset], decoded[offset + 1]])
}

#[test]
fn production_decoder_matches_independent_known_byte_abi_fixture() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("parse known-byte fixture");
    assert_eq!(fixture["schema"], "qwen36-native-int4-known-bytes/v1");
    assert!(
        fixture["provenance"]
            .as_str()
            .expect("fixture provenance")
            .contains("no producer packer"),
        "fixture must remain independent of the producer packer"
    );

    let rows = fixture["logical_shape"][0].as_u64().unwrap() as usize;
    let cols = fixture["logical_shape"][1].as_u64().unwrap() as usize;
    let group_size = fixture["group_size"].as_u64().unwrap() as usize;
    let row_pattern = hex_bytes(fixture["packed"]["row_pattern_hex"].as_str().unwrap());
    let row_repeats = fixture["packed"]["row_repeats"].as_u64().unwrap() as usize;
    let pattern_repeats = fixture["packed"]["pattern_repeats_per_row"]
        .as_u64()
        .unwrap() as usize;
    let packed_row = row_pattern.repeat(pattern_repeats);
    let packed = packed_row.repeat(row_repeats);
    assert_eq!(packed.len(), rows * cols / 2);

    let scale = hex_bytes(fixture["scale_bf16_le_hex"].as_str().unwrap());
    let zero = hex_bytes(fixture["zero_bf16_le_hex"].as_str().unwrap());
    assert_eq!(scale, [0x80, 0x3f, 0x00, 0x3f, 0x00, 0x40, 0x80, 0x3e]);
    assert_eq!(zero, [0x00, 0x00, 0x00, 0x40, 0x00, 0x41, 0x70, 0x41]);

    let decoded = dequant_int4_to_bf16_bytes(&packed, &scale, &zero, rows, cols, group_size);
    let expected_tiles = fixture["expected_bf16_bits_by_tile"]
        .as_array()
        .expect("expected tile tables");
    assert_eq!(expected_tiles.len(), 4);
    for row in 0..rows {
        let tile_row = row / group_size;
        for col in 0..cols {
            let tile_col = col / group_size;
            let nibble = col % 16;
            let expected = hex_u16(&expected_tiles[tile_row * 2 + tile_col][nibble]);
            assert_eq!(
                output_bits(&decoded, row, col, cols),
                expected,
                "decoded BF16 mismatch at row={row} col={col}"
            );
        }
    }

    for probe in fixture["probes"].as_array().expect("probe array") {
        let row = probe["row"].as_u64().unwrap() as usize;
        let col = probe["col"].as_u64().unwrap() as usize;
        let byte = packed[row * cols / 2 + col / 2];
        let nibble = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        let tile = (row / group_size) * 2 + col / group_size;
        let sidecar_bits =
            |bytes: &[u8]| u16::from_le_bytes([bytes[tile * 2], bytes[tile * 2 + 1]]);
        assert_eq!(byte, hex_u16(&probe["packed_byte"]) as u8);
        assert_eq!(nibble, probe["nibble"].as_u64().unwrap() as u8);
        assert_eq!(sidecar_bits(&scale), hex_u16(&probe["scale_bf16_bits"]));
        assert_eq!(sidecar_bits(&zero), hex_u16(&probe["zero_bf16_bits"]));
        assert_eq!(
            output_bits(&decoded, row, col, cols),
            hex_u16(&probe["output_bf16_bits"])
        );
    }
}
