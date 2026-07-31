use model_store::store::{Int4StorageKind, Int4StorageView};
use supersonic_runtime::qwen36_moe::weights::dequant_row_group_int4_to_bf16_bytes;

const RANK2_PACKED: [u8; 32] = [
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f,
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f,
];

const RANK2_SCALE_BF16: [u8; 4] = [0x80, 0x3f, 0x00, 0x3f];

const RANK2_EXPECTED_BF16: [u8; 128] = [
    0xe0, 0xc0, 0xc0, 0xc0, 0xa0, 0xc0, 0x80, 0xc0, 0x40, 0xc0, 0x00, 0xc0, 0x80, 0xbf, 0x00, 0x00,
    0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40, 0xe0, 0xc0,
    0xe0, 0xc0, 0xc0, 0xc0, 0xa0, 0xc0, 0x80, 0xc0, 0x40, 0xc0, 0x00, 0xc0, 0x80, 0xbf, 0x00, 0x00,
    0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40, 0xe0, 0xc0,
    0x60, 0xc0, 0x40, 0xc0, 0x20, 0xc0, 0x00, 0xc0, 0xc0, 0xbf, 0x80, 0xbf, 0x00, 0xbf, 0x00, 0x00,
    0x00, 0x3f, 0x80, 0x3f, 0xc0, 0x3f, 0x00, 0x40, 0x20, 0x40, 0x40, 0x40, 0x60, 0x40, 0x60, 0xc0,
    0x60, 0xc0, 0x40, 0xc0, 0x20, 0xc0, 0x00, 0xc0, 0xc0, 0xbf, 0x80, 0xbf, 0x00, 0xbf, 0x00, 0x00,
    0x00, 0x3f, 0x80, 0x3f, 0xc0, 0x3f, 0x00, 0x40, 0x20, 0x40, 0x40, 0x40, 0x60, 0x40, 0x60, 0xc0,
];

const RANK3_PACKED_WITH_PADDING: [u8; 40] = [
    0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f,
    0x00, 0x00, 0x00, 0x00, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x1f, 0x21, 0x43, 0x65, 0x87,
    0xa9, 0xcb, 0xed, 0x1f, 0x00, 0x00, 0x00, 0x00,
];

const RANK3_SCALE_BF16_WITH_PADDING: [u8; 12] = [
    0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
];

const RANK3_EXPECTED_BF16: [u8; 128] = [
    0xe0, 0xc0, 0xc0, 0xc0, 0xa0, 0xc0, 0x80, 0xc0, 0x40, 0xc0, 0x00, 0xc0, 0x80, 0xbf, 0x00, 0x00,
    0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40, 0xe0, 0xc0,
    0xe0, 0xc0, 0xc0, 0xc0, 0xa0, 0xc0, 0x80, 0xc0, 0x40, 0xc0, 0x00, 0xc0, 0x80, 0xbf, 0x00, 0x00,
    0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40, 0xe0, 0xc0,
    0x60, 0xc1, 0x40, 0xc1, 0x20, 0xc1, 0x00, 0xc1, 0xc0, 0xc0, 0x80, 0xc0, 0x00, 0xc0, 0x00, 0x00,
    0x00, 0x40, 0x80, 0x40, 0xc0, 0x40, 0x00, 0x41, 0x20, 0x41, 0x40, 0x41, 0x60, 0x41, 0x60, 0xc1,
    0x60, 0xc1, 0x40, 0xc1, 0x20, 0xc1, 0x00, 0xc1, 0xc0, 0xc0, 0x80, 0xc0, 0x00, 0xc0, 0x00, 0x00,
    0x00, 0x40, 0x80, 0x40, 0xc0, 0x40, 0x00, 0x41, 0x20, 0x41, 0x40, 0x41, 0x60, 0x41, 0x60, 0xc1,
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
fn decodes_independent_row_group_g32_known_bytes() {
    let decoded = dequant_row_group_int4_to_bf16_bytes(
        &RANK2_PACKED,
        &RANK2_SCALE_BF16,
        &row_group_view(vec![2, 32]),
    )
    .expect("independent row-group fixture must decode");

    assert_eq!(decoded, RANK2_EXPECTED_BF16);
}

#[test]
fn decodes_rank3_explicit_strides_without_reading_padding() {
    let mut view = row_group_view(vec![2, 1, 32]);
    view.packed_expert_stride_bytes = 20;
    view.scale_expert_stride_elements = 3;

    let decoded = dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_WITH_PADDING,
        &RANK3_SCALE_BF16_WITH_PADDING,
        &view,
    )
    .expect("rank-3 row-group fixture must decode");

    assert_eq!(decoded, RANK3_EXPECTED_BF16);
}

#[test]
fn rejects_malformed_row_group_views_and_planes() {
    let packed = RANK2_PACKED;
    let scale = RANK2_SCALE_BF16;

    let mut wrong_kind = row_group_view(vec![2, 32]);
    wrong_kind.kind = Int4StorageKind::TileV1;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_kind).is_err());

    let mut wrong_group = row_group_view(vec![2, 32]);
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

    let mut wrong_implicit_zero = row_group_view(vec![2, 32]);
    wrong_implicit_zero.implicit_zero_code = Some(7);
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_implicit_zero).is_err());

    let mut wrong_output_group = row_group_view(vec![2, 32]);
    wrong_output_group.output_group_size = 32;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &wrong_output_group).is_err());

    let mut unexpected_zero = row_group_view(vec![2, 32]);
    unexpected_zero.zero_tensor = Some("opaque/zero".to_owned());
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &unexpected_zero).is_err());

    let mut short_packed_stride = row_group_view(vec![2, 32]);
    short_packed_stride.packed_row_stride_bytes = 15;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &short_packed_stride).is_err());

    let mut short_scale_stride = row_group_view(vec![2, 32]);
    short_scale_stride.scale_row_stride_elements = 0;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &short_scale_stride).is_err());

    let mut rank2_expert_stride = row_group_view(vec![2, 32]);
    rank2_expert_stride.packed_expert_stride_bytes = 32;
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &rank2_expert_stride).is_err());

    assert!(dequant_row_group_int4_to_bf16_bytes(
        &packed[..31],
        &scale,
        &row_group_view(vec![2, 32])
    )
    .is_err());
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &[RANK2_PACKED.as_slice(), &[0xff]].concat(),
        &scale,
        &row_group_view(vec![2, 32])
    )
    .is_err());
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &packed,
        &scale[..3],
        &row_group_view(vec![2, 32])
    )
    .is_err());
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &packed,
        &[RANK2_SCALE_BF16.as_slice(), &[0, 0]].concat(),
        &row_group_view(vec![2, 32])
    )
    .is_err());

    let mut zero_code_packed = packed;
    zero_code_packed[0] = 0x10;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &zero_code_packed,
        &scale,
        &row_group_view(vec![2, 32])
    )
    .is_err());

    let overflowing = row_group_view(vec![usize::MAX, 32]);
    assert!(dequant_row_group_int4_to_bf16_bytes(&packed, &scale, &overflowing).is_err());

    let mut rank3 = row_group_view(vec![2, 1, 32]);
    rank3.packed_expert_stride_bytes = 20;
    rank3.scale_expert_stride_elements = 3;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_WITH_PADDING[..39],
        &RANK3_SCALE_BF16_WITH_PADDING,
        &rank3,
    )
    .is_err());
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_WITH_PADDING,
        &RANK3_SCALE_BF16_WITH_PADDING[..10],
        &rank3,
    )
    .is_err());

    let mut rank3_short_expert = rank3.clone();
    rank3_short_expert.packed_expert_stride_bytes = 15;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_WITH_PADDING,
        &RANK3_SCALE_BF16_WITH_PADDING,
        &rank3_short_expert,
    )
    .is_err());

    let mut rank3_short_scale_expert = rank3;
    rank3_short_scale_expert.scale_expert_stride_elements = 0;
    assert!(dequant_row_group_int4_to_bf16_bytes(
        &RANK3_PACKED_WITH_PADDING,
        &RANK3_SCALE_BF16_WITH_PADDING,
        &rank3_short_scale_expert,
    )
    .is_err());
}
