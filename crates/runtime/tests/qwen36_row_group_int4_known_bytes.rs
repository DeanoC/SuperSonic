use model_store::store::{Int4StorageKind, Int4StorageView};
use supersonic_runtime::qwen36_moe::weights::dequant_row_group_int4_to_bf16_bytes;

// Task 2's binding row-group fixture, copied verbatim rather than packed here.
const TASK2_PACKED: [u8; 16] = [
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f,
];

const TASK2_SCALE_BF16: [u8; 2] = [0x80, 0x3f];

const TASK2_EXPECTED_BF16: [u8; 64] = [
    0xe0, 0xc0, 0xc0, 0xc0, 0xa0, 0xc0, 0x80, 0xc0, 0x40, 0xc0, 0x00, 0xc0, 0x80, 0xbf, 0x00, 0x00,
    0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40, 0x00, 0x00,
    0xe0, 0xc0, 0xc0, 0xc0, 0xa0, 0xc0, 0x80, 0xc0, 0x40, 0xc0, 0x00, 0xc0, 0x80, 0xbf, 0x00, 0x00,
    0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40, 0x00, 0x00,
];

const RANK3_PACKED_MINIMAL: [u8; 84] = [
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f,
    0x00, 0x00, 0x00, 0x00, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f, 0x21, 0x43, 0x65, 0x87,
    0xa9, 0xcb, 0xed, 0x8f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f,
    0x00, 0x00, 0x00, 0x00, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x8f, 0x21, 0x43, 0x65, 0x87,
    0xa9, 0xcb, 0xed, 0x8f,
];

const RANK3_SCALE_BF16_MINIMAL: [u8; 16] = [
    0x80, 0x3f, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x3f,
];

fn row_group_view(logical_shape: Vec<usize>) -> Int4StorageView {
    Int4StorageView {
        kind: Int4StorageKind::RowGroupSymmetric,
        group_size: 32,
        packed_tensor: "opaque/packed".to_owned(),
        scale_tensor: "opaque/scale".to_owned(),
        zero_tensor: None,
        logical_shape,
        packed_row_stride_bytes: 16,
        packed_expert_stride_bytes: 0,
        scale_row_stride_elements: 1,
        scale_expert_stride_elements: 0,
        output_group_size: 1,
        implicit_zero_code: Some(8),
    }
}

#[test]
fn decodes_task2_binding_row_group_g32_known_bytes() {
    let decoded = dequant_row_group_int4_to_bf16_bytes(
        &TASK2_PACKED,
        &TASK2_SCALE_BF16,
        &row_group_view(vec![1, 32]),
    )
    .expect("Task 2's independent row-group fixture must decode");

    assert_eq!(decoded, TASK2_EXPECTED_BF16);
}

#[test]
fn decodes_rank3_minimal_and_trailing_padded_planes_identically() {
    let mut view = row_group_view(vec![2, 2, 32]);
    view.packed_row_stride_bytes = 20;
    view.packed_expert_stride_bytes = 48;
    view.scale_row_stride_elements = 2;
    view.scale_expert_stride_elements = 5;

    let minimal = dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL,
        &RANK3_SCALE_BF16_MINIMAL,
        &view,
    )
    .expect("rank-3 fixture ending at its final logical access must decode");

    let mut packed_with_trailing = RANK3_PACKED_MINIMAL.to_vec();
    packed_with_trailing.extend_from_slice(&[0xde, 0xad, 0xbe]);
    let mut scale_with_trailing = RANK3_SCALE_BF16_MINIMAL.to_vec();
    scale_with_trailing.extend_from_slice(&[0xfe, 0xca]);
    let trailing =
        dequant_row_group_int4_to_bf16_bytes(&packed_with_trailing, &scale_with_trailing, &view)
            .expect("harmless rank-3 carrier padding must not be rejected");

    assert_eq!(minimal.len(), 2 * 2 * 32 * 2);
    assert_eq!(&minimal[..TASK2_EXPECTED_BF16.len()], TASK2_EXPECTED_BF16);
    assert_eq!(trailing, minimal);
}

#[test]
fn rejects_malformed_row_group_views_and_planes() {
    let packed = TASK2_PACKED;
    let scale = TASK2_SCALE_BF16;

    let mut wrong_kind = row_group_view(vec![1, 32]);
    wrong_kind.kind = Int4StorageKind::TileV1;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_kind).is_err());

    let mut wrong_group = row_group_view(vec![1, 32]);
    wrong_group.group_size = 64;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_group).is_err());

    for shape in [
        vec![32],
        vec![1, 1, 1, 32],
        vec![0, 32],
        vec![1, 33],
        vec![1, 34],
    ] {
        let view = row_group_view(shape);
        assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &view).is_err());
    }

    let mut wrong_implicit_zero = row_group_view(vec![1, 32]);
    wrong_implicit_zero.implicit_zero_code = Some(7);
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_implicit_zero).is_err());

    let mut wrong_output_group = row_group_view(vec![1, 32]);
    wrong_output_group.output_group_size = 32;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_output_group).is_err());

    let mut unexpected_zero = row_group_view(vec![1, 32]);
    unexpected_zero.zero_tensor = Some("opaque/zero".to_owned());
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &unexpected_zero).is_err());

    let mut short_packed_stride = row_group_view(vec![1, 32]);
    short_packed_stride.packed_row_stride_bytes = 15;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &short_packed_stride).is_err());

    let mut short_scale_stride = row_group_view(vec![1, 32]);
    short_scale_stride.scale_row_stride_elements = 0;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &short_scale_stride).is_err());

    let mut rank2_expert_stride = row_group_view(vec![1, 32]);
    rank2_expert_stride.packed_expert_stride_bytes = 32;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &rank2_expert_stride).is_err());

    assert!(dequant_row_group_int4_to_bf16_bytes(
        &packed[..15],
        &scale,
        &row_group_view(vec![1, 32])
    )
    .is_err());
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &packed,
        &scale[..1],
        &row_group_view(vec![1, 32])
    )
    .is_err());

    let mut zero_code_packed = packed;
    zero_code_packed[0] = 0x10;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &zero_code_packed,
        &scale,
        &row_group_view(vec![1, 32])
    )
    .is_err());

    let overflowing = row_group_view(vec![usize::MAX, 32]);
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &overflowing).is_err());

    let mut rank3 = row_group_view(vec![2, 2, 32]);
    rank3.packed_row_stride_bytes = 20;
    rank3.packed_expert_stride_bytes = 48;
    rank3.scale_row_stride_elements = 2;
    rank3.scale_expert_stride_elements = 5;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL[..83],
        &RANK3_SCALE_BF16_MINIMAL,
        &rank3,
    )
    .is_err());
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL,
        &RANK3_SCALE_BF16_MINIMAL[..14],
        &rank3,
    )
    .is_err());

    let mut rank3_short_packed_row = rank3.clone();
    rank3_short_packed_row.packed_row_stride_bytes = 15;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL,
        &RANK3_SCALE_BF16_MINIMAL,
        &rank3_short_packed_row,
    )
    .is_err());

    let mut rank3_short_scale_row = rank3.clone();
    rank3_short_scale_row.scale_row_stride_elements = 0;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL,
        &RANK3_SCALE_BF16_MINIMAL,
        &rank3_short_scale_row,
    )
    .is_err());

    let mut rank3_short_expert = rank3.clone();
    rank3_short_expert.packed_expert_stride_bytes = 35;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL,
        &RANK3_SCALE_BF16_MINIMAL,
        &rank3_short_expert,
    )
    .is_err());

    let mut rank3_short_scale_expert = rank3;
    rank3_short_scale_expert.scale_expert_stride_elements = 2;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_MINIMAL,
        &RANK3_SCALE_BF16_MINIMAL,
        &rank3_short_scale_expert,
    )
    .is_err());
}
