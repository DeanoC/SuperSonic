use half::bf16;

const HELPERS: &str =
    include_str!("../../../kernels/qwen36_moe_persistent/helpers.cuh");

#[derive(Clone, Copy, Debug)]
struct HostDesc {
    encoding: i32,
    zero_present: bool,
    packed_row_stride: usize,
    packed_expert_stride: usize,
    scale_row_stride: usize,
    scale_expert_stride: usize,
    input_group_size: usize,
    output_group_size: usize,
    implicit_zero_code: i32,
}

fn canonical_g32(desc: HostDesc, cols: usize) -> bool {
    desc.encoding == 2
        && !desc.zero_present
        && desc.input_group_size == 32
        && desc.output_group_size == 1
        && desc.implicit_zero_code == 8
        && cols >= 32
        && cols % 32 == 0
        && desc.packed_row_stride == cols / 2
        && desc.scale_row_stride == cols / 32
        && desc.packed_row_stride == desc.scale_row_stride * 16
        && ((desc.packed_expert_stride == 0 && desc.scale_expert_stride == 0)
            || (desc.packed_expert_stride > 0
                && desc.scale_expert_stride > 0
                && desc.packed_expert_stride % desc.packed_row_stride == 0
                && desc.scale_expert_stride % desc.scale_row_stride == 0
                && desc.packed_expert_stride / desc.packed_row_stride
                    == desc.scale_expert_stride / desc.scale_row_stride))
}

fn bf16_value(bits: &[u16], index: usize) -> f32 {
    f32::from(bf16::from_bits(bits[index]))
}

fn generic_value(
    packed: &[u8],
    scales: &[u16],
    zeros: Option<&[u16]>,
    desc: HostDesc,
    expert: usize,
    row: usize,
    col: usize,
) -> f32 {
    let packed_index = expert * desc.packed_expert_stride
        + row * desc.packed_row_stride
        + col / 2;
    let packed_byte = packed[packed_index];
    let code = if col & 1 == 0 {
        packed_byte & 0x0f
    } else {
        packed_byte >> 4
    };
    let scale_index = expert * desc.scale_expert_stride
        + (row / desc.output_group_size) * desc.scale_row_stride
        + col / desc.input_group_size;
    let scale = bf16_value(scales, scale_index);
    let zero = if desc.implicit_zero_code >= 0 {
        desc.implicit_zero_code as f32
    } else {
        bf16_value(zeros.expect("explicit zero plane"), scale_index)
    };
    (f32::from(code) - zero) * scale
}

fn generic_span8(
    packed: &[u8],
    scales: &[u16],
    zeros: Option<&[u16]>,
    desc: HostDesc,
    expert: usize,
    row: usize,
    col: usize,
) -> [f32; 8] {
    std::array::from_fn(|offset| {
        generic_value(packed, scales, zeros, desc, expert, row, col + offset)
    })
}

// Independent CPU oracle for the proposed hot span. It intentionally does not
// call the producer packer, runtime decoder, or the HIP helper under test.
fn g32_span8(
    packed: &[u8],
    scales: &[u16],
    desc: HostDesc,
    expert: usize,
    row: usize,
    col: usize,
) -> [f32; 8] {
    assert_eq!(desc.encoding, 2);
    assert!(!desc.zero_present);
    assert_eq!(desc.input_group_size, 32);
    assert_eq!(desc.output_group_size, 1);
    assert_eq!(desc.implicit_zero_code, 8);
    assert_eq!(col % 8, 0);
    assert!(col % 32 <= 24);
    let row_base = expert * desc.packed_expert_stride + row * desc.packed_row_stride;
    let scale_base = expert * desc.scale_expert_stride + row * desc.scale_row_stride;
    let byte_base = row_base + (col >> 1);
    let packed_word = u32::from_le_bytes(packed[byte_base..byte_base + 4].try_into().unwrap());
    let scale = bf16_value(scales, scale_base + (col >> 5));
    std::array::from_fn(|offset| {
        let code = ((packed_word >> (offset * 4)) & 0x0f) as f32;
        (code - 8.0) * scale
    })
}

fn g32_span16_bf16_bits(
    packed: &[u8],
    scales: &[u16],
    desc: HostDesc,
    expert: usize,
    row: usize,
    col: usize,
) -> [u16; 16] {
    assert_eq!(col % 16, 0);
    assert!(col % 32 <= 16);
    let first = g32_span8(packed, scales, desc, expert, row, col);
    let second = g32_span8(packed, scales, desc, expert, row, col + 8);
    std::array::from_fn(|index| {
        bf16::from_f32(if index < 8 {
            first[index]
        } else {
            second[index - 8]
        })
        .to_bits()
    })
}

fn fixture(cols: usize, rank3: bool) -> (Vec<u8>, Vec<u16>, HostDesc, usize, usize) {
    let experts = if rank3 { 2 } else { 1 };
    let rows = if rank3 { 3 } else { 4 };
    let packed_row_stride = cols / 2;
    let scale_row_stride = cols / 32;
    let packed_expert_stride = if rank3 { rows * packed_row_stride } else { 0 };
    let scale_expert_stride = if rank3 { rows * scale_row_stride } else { 0 };
    let desc = HostDesc {
        encoding: 2,
        zero_present: false,
        packed_row_stride,
        packed_expert_stride,
        scale_row_stride,
        scale_expert_stride,
        input_group_size: 32,
        output_group_size: 1,
        implicit_zero_code: 8,
    };
    let mut packed = vec![0u8; experts * rows * packed_row_stride.max(1)];
    for expert in 0..experts {
        for row in 0..rows {
            let base = expert * packed_expert_stride + row * packed_row_stride;
            for byte in 0..packed_row_stride {
                let low = ((1 + expert * 3 + row * 5 + byte * 7) & 0x0f) as u8;
                let high = ((2 + expert * 11 + row * 13 + byte * 3) & 0x0f) as u8;
                packed[base + byte] = low | (high << 4);
            }
        }
    }
    let mut scales = vec![0u16; experts * rows * scale_row_stride];
    let scale_pattern = [0x3f80, 0x3f00, 0x3e80, 0x3dcd];
    for expert in 0..experts {
        for row in 0..rows {
            let base = expert * scale_expert_stride + row * scale_row_stride;
            for group in 0..scale_row_stride {
                scales[base + group] = scale_pattern[(expert + row + group) % scale_pattern.len()];
            }
        }
    }
    (packed, scales, desc, experts, rows)
}

#[test]
fn helpers_declare_the_restricted_canonical_g32_hot_span_contract() {
    for required in [
        "qwen36_g32_descriptor_is_canonical",
        "qwen36_g32_dequant_span_8",
        "qwen36_g32_wmma_operand",
        "const hip_bfloat16* __restrict__ scale_row",
        "const uint32_t packed_word",
    ] {
        assert!(HELPERS.contains(required), "missing hot-path contract: {required}");
    }
    assert!(HELPERS.contains("desc.implicit_zero_code != 8"));
}

fn helper_region(signature: &str, next_signature: &str) -> &'static str {
    let start = HELPERS
        .find(signature)
        .unwrap_or_else(|| panic!("missing helper signature: {signature}"));
    let relative_end = HELPERS[start..]
        .find(next_signature)
        .unwrap_or_else(|| panic!("missing helper boundary: {next_signature}"));
    &HELPERS[start..start + relative_end]
}

#[test]
fn descriptor_scalar_pair_and_wmma_routes_have_g32_fast_branches() {
    for (signature, next_signature) in [
        (
            "qwen36_int4_dequant_8(",
            "qwen36_int4_pair_dequant_8(",
        ),
        (
            "qwen36_int4_pair_dequant_8(",
            "struct qwen36_float_pair",
        ),
        (
            "qwen36_int4_dq8_matvec_partial(",
            "qwen36_int4_dq8_pair_matvec_partial_same_row(",
        ),
        (
            "qwen36_int4_dq8_pair_matvec_partial_same_row(",
            "// ---------------------------------------------------------------------------",
        ),
    ] {
        let region = helper_region(signature, next_signature);
        assert!(region.contains("qwen36_g32_dequant_span_8"), "{signature} lacks G32 span route");
        assert!(region.contains("qwen36_int4_value") || region.contains("qwen36_int4_dequant_8"));
    }

    let wmma_operand = helper_region(
        "qwen36_wmma_int4_operand(",
        "template <typename Activation>",
    );
    assert!(wmma_operand.contains("qwen36_g32_wmma_operand"));
    let wmma_matvec = helper_region(
        "qwen36_wmma_int4_matvec_partial_16rows(",
        "template <typename Activation>\n__device__ inline qwen36_float8_pair",
    );
    assert!(wmma_matvec.contains("qwen36_g32_wmma_operand"));
    let wmma_pair = helper_region(
        "qwen36_wmma_int4_pair_matvec_partial_16rows(",
        "#endif",
    );
    assert!(wmma_pair.contains("qwen36_g32_wmma_operand"));
}

#[test]
fn independent_g32_spans_match_generic_for_k_and_rank_matrix() {
    for cols in [512, 2048, 4096] {
        for rank3 in [false, true] {
            let (packed, scales, desc, experts, rows) = fixture(cols, rank3);
            assert!(canonical_g32(desc, cols), "fixture must be canonical: {desc:?}");
            for expert in 0..experts {
                for row in 0..rows {
                    for col in (0..cols).step_by(8) {
                        let generic = generic_span8(
                            &packed, &scales, None, desc, expert, row, col,
                        );
                        let fast = g32_span8(&packed, &scales, desc, expert, row, col);
                        assert_eq!(generic, fast, "K={cols} rank3={rank3} e={expert} r={row} c={col}");
                    }
                    for col in (0..cols).step_by(16) {
                        let generic = (0..16)
                            .map(|offset| {
                                bf16::from_f32(generic_value(
                                    &packed,
                                    &scales,
                                    None,
                                    desc,
                                    expert,
                                    row,
                                    col + offset,
                                ))
                                .to_bits()
                            })
                            .collect::<Vec<_>>();
                        let fast = g32_span16_bf16_bits(
                            &packed, &scales, desc, expert, row, col,
                        );
                        assert_eq!(generic, fast, "WMMA K={cols} rank3={rank3} e={expert} r={row} c={col}");
                    }
                }
            }

            let mut mutated = packed.clone();
            let mutation_index = desc.packed_row_stride;
            mutated[mutation_index] ^= 0x01;
            let before = g32_span8(&packed, &scales, desc, 0, 1, 0);
            let after = g32_span8(&mutated, &scales, desc, 0, 1, 0);
            assert_ne!(before, after, "packed-nibble mutation must be observable");
            assert_eq!(
                after,
                generic_span8(&mutated, &scales, None, desc, 0, 1, 0),
                "mutated fast span must retain generic parity"
            );
        }
    }
}

#[test]
fn generic_fallback_remains_selected_for_tile_v1_and_unusual_geometry() {
    let (packed, scales, mut desc, _, rows) = fixture(512, false);
    desc.encoding = 1;
    desc.zero_present = true;
    desc.implicit_zero_code = -1;
    let zeros = vec![0x3f00; rows * desc.scale_row_stride];
    assert!(!canonical_g32(desc, 512));
    let expected = generic_span8(&packed, &scales, Some(&zeros), desc, 0, 0, 0);
    assert_eq!(expected.len(), 8);

    let mut unusual = desc;
    unusual.encoding = 2;
    unusual.zero_present = false;
    unusual.implicit_zero_code = 8;
    unusual.packed_row_stride += 16;
    assert!(!canonical_g32(unusual, 512));
}
