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

#[test]
fn hip_descriptor_decoder_preserves_tile_v1_known_bytes() -> anyhow::Result<()> {
    use gpu_hal::{set_backend, Backend, GpuBuffer, ScalarType};
    use half::bf16;
    use kernel_ffi::qwen36_moe::{int4_descriptor_dequant_smoke_launch, Qwen36MoeInt4WeightDesc};

    set_backend(Backend::Hip);
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let rows = fixture["logical_shape"][0].as_u64().unwrap() as usize;
    let cols = fixture["logical_shape"][1].as_u64().unwrap() as usize;
    let row_pattern = hex_bytes(fixture["packed"]["row_pattern_hex"].as_str().unwrap());
    let pattern_repeats = fixture["packed"]["pattern_repeats_per_row"]
        .as_u64()
        .unwrap() as usize;
    let packed = row_pattern.repeat(pattern_repeats).repeat(rows);
    let scale = hex_bytes(fixture["scale_bf16_le_hex"].as_str().unwrap());
    let zero = hex_bytes(fixture["zero_bf16_le_hex"].as_str().unwrap());

    let packed_gpu = GpuBuffer::from_host_bytes(0, ScalarType::U8, &[packed.len()], &packed)?;
    let scale_gpu = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[scale.len() / 2], &scale)?;
    let zero_gpu = GpuBuffer::from_host_bytes(0, ScalarType::BF16, &[zero.len() / 2], &zero)?;
    let mut wide_gpu = GpuBuffer::zeros(0, ScalarType::F32, &[rows * cols])?;
    let mut scalar_gpu = GpuBuffer::zeros(0, ScalarType::F32, &[rows * cols])?;
    let desc = Qwen36MoeInt4WeightDesc {
        scale: scale_gpu.as_ptr(),
        zero: zero_gpu.as_ptr(),
        packed_row_stride_bytes: (cols / 2) as u64,
        packed_expert_stride_bytes: 0,
        scale_row_stride_elements: (cols / 128) as u64,
        scale_expert_stride_elements: 0,
        input_group_size: 128,
        output_group_size: 128,
        implicit_zero_code: -1,
        encoding: 1,
    };

    int4_descriptor_dequant_smoke_launch(
        0,
        &packed_gpu,
        &scale_gpu,
        Some(&zero_gpu),
        &desc,
        1,
        rows as i32,
        cols as i32,
        &mut wide_gpu,
        &mut scalar_gpu,
    )?;

    let wide = wide_gpu.to_host_bytes()?;
    let scalar = scalar_gpu.to_host_bytes()?;
    let expected_tiles = fixture["expected_bf16_bits_by_tile"].as_array().unwrap();
    for row in 0..rows {
        for col in 0..cols {
            let tile = (row / 128) * 2 + col / 128;
            let expected_bits = hex_u16(&expected_tiles[tile][col % 16]);
            let expected = f32::from(bf16::from_bits(expected_bits));
            let offset = (row * cols + col) * 4;
            let read =
                |bytes: &[u8]| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert_eq!(
                read(&wide),
                expected,
                "tile-v1 8-wide mismatch at row={row} col={col}"
            );
            assert_eq!(
                read(&scalar),
                expected,
                "tile-v1 scalar mismatch at row={row} col={col}"
            );
        }
    }
    Ok(())
}
