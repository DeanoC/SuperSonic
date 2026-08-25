use half::bf16;
use model_store::q6_bound::{
    activation_block_norms, argmax_f32_as_bf16, decode_q6_k_block, f32_to_bf16_rne_finite,
    q8_1_reconstruct, raw_q6_scalar_row_f32, required_exact_tiles, summarize_tile_counts,
    upward_f16_norm, weight_block_norms,
};

fn hand_selected_q6_row() -> (Vec<u8>, Vec<u16>) {
    let mut row = vec![0u8; 20 * 210];
    let block0 = &mut row[..210];
    let block0_scales: [i8; 16] = [
        3, -2, 5, -7, 11, -13, 17, -19, 23, -29, 31, -37, 41, -43, 47, -53,
    ];
    for (index, &scale) in block0_scales.iter().enumerate() {
        block0[192 + index] = scale as u8;
    }
    block0[208..210].copy_from_slice(&0x3555u16.to_le_bytes()); // FP16 0.33325195
    for (offset, value) in [
        (0, 0x1e),
        (1, 0x2d),
        (16, 0x3c),
        (17, 0x4b),
        (32, 0x5a),
        (33, 0x69),
        (48, 0x78),
        (49, 0x87),
        (64, 0x96),
        (65, 0xa5),
        (80, 0xb4),
        (81, 0xc3),
        (96, 0xd2),
        (97, 0xe1),
        (112, 0xf0),
        (113, 0x0f),
    ] {
        block0[offset] = value;
    }
    for (offset, value) in [
        (128, 0b11_10_01_00),
        (129, 0b00_01_10_11),
        (144, 0b10_11_00_01),
        (145, 0b01_00_11_10),
        (160, 0b01_11_10_00),
        (161, 0b10_00_01_11),
        (176, 0b00_10_11_01),
        (177, 0b11_01_00_10),
    ] {
        block0[offset] = value;
    }

    let block1 = &mut row[210..420];
    let block1_scales: [i8; 16] = [
        -3, 4, -6, 8, -10, 12, -14, 16, -18, 20, -22, 24, -26, 28, -30, 32,
    ];
    for (index, &scale) in block1_scales.iter().enumerate() {
        block1[192 + index] = scale as u8;
    }
    block1[208..210].copy_from_slice(&0x3a00u16.to_le_bytes()); // FP16 0.75
    for (offset, value) in [
        (0, 0xf1),
        (1, 0xe2),
        (16, 0xd3),
        (17, 0xc4),
        (32, 0xb5),
        (33, 0xa6),
        (48, 0x97),
        (49, 0x88),
        (64, 0x79),
        (65, 0x6a),
        (80, 0x5b),
        (81, 0x4c),
        (96, 0x3d),
        (97, 0x2e),
        (112, 0x1f),
        (113, 0x10),
    ] {
        block1[offset] = value;
    }
    for (offset, value) in [
        (128, 0b01_11_00_10),
        (129, 0b10_00_11_01),
        (144, 0b11_01_10_00),
        (145, 0b00_10_01_11),
        (160, 0b10_11_01_00),
        (161, 0b01_00_10_11),
        (176, 0b11_10_00_01),
        (177, 0b00_01_11_10),
    ] {
        block1[offset] = value;
    }

    let activation_values = [
        1.0f32, -0.75, 0.125, -2.5, 0.2, -0.3, 3.25, -4.5, 5.75, -6.125, 7.5, -8.25, 9.0, -10.5,
        11.75, -12.875,
    ];
    let activation_bf16 = (0..5120)
        .map(|index| bf16::from_f32(activation_values[index % activation_values.len()]).to_bits())
        .collect();
    (row, activation_bf16)
}

#[test]
fn raw_q6_scalar_row_matches_test_local_lane_reduction_bit_for_bit() {
    let (row, activation_bf16) = hand_selected_q6_row();
    let mut expected_lanes = [0.0f32; 32];
    for block_index in 0..20 {
        let block = decode_q6_k_block(&row[block_index * 210..(block_index + 1) * 210])
            .expect("valid hand-selected Q6_K block");
        for lane in 0..32 {
            for t in 0..8 {
                let coordinate = lane + 32 * t;
                let weight = block.d
                    * f32::from(block.scales[coordinate])
                    * f32::from(block.quants[coordinate]);
                let x = bf16::from_bits(activation_bf16[block_index * 256 + coordinate]).to_f32();
                expected_lanes[lane] = weight.mul_add(x, expected_lanes[lane]);
            }
        }
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let before = expected_lanes;
        for lane in 0..32 {
            expected_lanes[lane] = before[lane] + before[lane ^ offset];
        }
    }

    let actual = raw_q6_scalar_row_f32(&row, &activation_bf16).expect("finite scalar row");
    assert_eq!(actual.to_bits(), expected_lanes[0].to_bits());
}

#[test]
fn bf16_rne_argmax_preserves_strict_lowest_index_ties() {
    assert_eq!(f32_to_bf16_rne_finite(1.003_906_25).unwrap(), 0x3f80);
    assert!(f32_to_bf16_rne_finite(f32::INFINITY).is_err());
    assert_eq!(argmax_f32_as_bf16(&[1.0, 1.003_0, 0.0]).unwrap(), 0);
    assert_eq!(argmax_f32_as_bf16(&[-0.0, 0.0]).unwrap(), 0);
}

#[test]
fn bf16_rne_rejects_positive_finite_overflow_to_infinity() {
    let just_overflowing = f32::from_bits(0x7f7f_8000);

    assert!(just_overflowing.is_finite());
    assert!(f32_to_bf16_rne_finite(just_overflowing).is_err());
}

#[test]
fn bf16_rne_rejects_negative_finite_overflow_to_infinity() {
    let just_overflowing = f32::from_bits(0xff7f_8000);

    assert!(just_overflowing.is_finite());
    assert!(f32_to_bf16_rne_finite(just_overflowing).is_err());
}

#[test]
fn bf16_rne_accepts_largest_f32_values_that_round_to_finite_bf16_maxima() {
    assert_eq!(
        f32_to_bf16_rne_finite(f32::from_bits(0x7f7f_7fff)),
        Ok(0x7f7f)
    );
    assert_eq!(
        f32_to_bf16_rne_finite(f32::from_bits(0xff7f_7fff)),
        Ok(0xff7f)
    );
}

#[test]
fn raw_q6_scalar_row_rejects_wrong_shapes() {
    assert!(raw_q6_scalar_row_f32(&[0; 209], &[0; 5120]).is_err());
    assert!(raw_q6_scalar_row_f32(&[0; 20 * 210], &[0; 5119]).is_err());
}

#[test]
fn q6_zero_code_decodes_with_signed_scale_and_baseline_truncation() {
    let mut block = [0u8; 210];
    block[192..208].fill(1);
    block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // FP16 1.0

    let decoded = decode_q6_k_block(&block).expect("valid Q6_K block");
    assert_eq!(decoded.raw.len(), 256);
    assert_eq!(decoded.baseline_bf16.len(), 256);
    assert!(decoded.raw.iter().all(|&value| value == -32.0));
    assert!(decoded.baseline_bf16.iter().all(|&value| value == -32.0));

    let norms = weight_block_norms(&decoded);
    assert_eq!(norms.w_l2, 512.0); // sqrt(256 * 32^2)
    assert_eq!(norms.d_l2, 0.0);
}

#[test]
fn norm_encoding_rounds_toward_positive_infinity() {
    assert_eq!(upward_f16_norm(1.0).expect("exact"), 1.0);
    assert_eq!(
        upward_f16_norm(1.000_1).expect("rounded upward"),
        1.000_976_6
    );
    assert!(upward_f16_norm(-0.1).is_err());
    assert!(upward_f16_norm(65_505.0).is_err());
    assert!(upward_f16_norm(1.0e100).is_err());
}

#[test]
fn q6_layout_selects_each_nibble_bitplane_and_half_scale_group() {
    let mut block = [0u8; 210];
    block[0] = 0xa1;
    block[32] = 0xb2;
    block[64] = 0xc3;
    block[96] = 0xd4;
    block[128] = 0b11_10_01_00;
    block[160] = 0b11_10_01_00;
    for (index, scale) in (1u8..=16).enumerate() {
        block[192 + index] = scale;
    }
    block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // FP16 1.0

    let decoded = decode_q6_k_block(&block).expect("valid Q6_K block");
    // Hand-derived from ql low/high nibble, qh 2-bit planes 0/1/2/3,
    // scales 1/3/5/7 in half 0 and 9/11/13/15 in half 1.
    assert_eq!(
        [
            decoded.raw[0],
            decoded.raw[32],
            decoded.raw[64],
            decoded.raw[96],
            decoded.raw[128],
            decoded.raw[160],
            decoded.raw[192],
            decoded.raw[224],
        ],
        [-31.0, -42.0, 50.0, 189.0, -261.0, -132.0, 156.0, 435.0]
    );
}

#[test]
fn q6_baseline_operand_is_truncated_not_rounded_to_bf16() {
    let mut block = [0u8; 210];
    block[192..208].fill(1);
    block[0] = 1; // q=-31 for logical coordinate 0
    block[208..210].copy_from_slice(&0x2e66u16.to_le_bytes()); // FP16 0.0999755859375

    let decoded = decode_q6_k_block(&block).expect("valid Q6_K block");
    assert_eq!(decoded.raw[0], -3.099_243_2);
    assert_eq!(decoded.baseline_bf16[0], -3.093_75); // F32 bits c0465a00 -> c0460000
}

#[test]
fn q8_reconstruction_uses_fp16_stored_scale_in_32_value_groups() {
    let mut x = vec![0.0f32; 256];
    x[0] = 1.0;
    x[1] = -0.5;
    let reconstructed = q8_1_reconstruct(&x).expect("one Q6 block activation");

    assert_eq!(reconstructed.len(), 256);
    // 1/127 rounds to FP16 bits 0x2008 (0.00787353515625), which is the
    // scale consumed by MMVQ rather than the pre-store F32 quotient.
    assert_eq!(reconstructed[0], 0.999_938_96);
    assert_eq!(reconstructed[1], -0.503_906_25);
    assert!(reconstructed[2..].iter().all(|&value| value == 0.0));

    let norms = activation_block_norms(&x, &reconstructed).expect("matching activation");
    assert!(norms.e_l2 > 0.0039 && norms.e_l2 < 0.0040);
    assert!(norms.a_l2 > 1.119 && norms.a_l2 < 1.120);
    assert!((norms.x_l2 - 1.118_034).abs() < 0.000_001);
}

#[test]
fn exact_tile_selection_preserves_lower_index_bf16_ties() {
    let logits = [4.0, 4.0, 4.0, 5.0, 5.0, 0.0, 0.0, 0.0];
    let centers = [4.9, 4.0, 4.0, 5.0, 4.9, 4.9, 5.0, 0.0];
    let radii = [0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0];

    let selection = required_exact_tiles(&logits, &centers, &radii, 2)
        .expect("matching finite diagnostic rows");

    assert_eq!(selection.winner, 3, "the lower tied BF16 row must win");
    assert_eq!(selection.rows_not_excludable, 2);
    assert_eq!(selection.exact_tiles_required, 3);
}

#[test]
fn exact_tile_selection_starts_from_the_proposal_winner_tile() {
    let logits = [10.0, 0.0, 9.0, 0.0, 8.0, 0.0];
    let centers = [8.0, 0.0, 7.0, 0.0, 11.0, 0.0];
    let radii = [2.0, 0.0, 2.0, 0.0, 3.0, 0.0];

    let selection = required_exact_tiles(&logits, &centers, &radii, 2)
        .expect("matching finite diagnostic rows");

    assert_eq!(selection.winner, 0);
    assert_eq!(selection.exact_tiles_required, 3);
}

#[test]
fn tile_count_summary_reports_tail_and_strict_over_limit_fallbacks() {
    let summary = summarize_tile_counts(&[1, 2, 3, 4, 17], 16).expect("non-empty counts");

    assert_eq!(summary.p50, 3);
    assert_eq!(summary.p95, 17);
    assert_eq!(summary.p99, 17);
    assert_eq!(summary.max, 17);
    assert_eq!(summary.fallback_count, 1);
}

#[test]
fn tile_count_summary_uses_nearest_rank_for_small_corpus_tail() {
    let mut counts = vec![3; 63];
    counts.push(5);

    let summary = summarize_tile_counts(&counts, 16).expect("non-empty counts");

    assert_eq!(summary.p95, 3);
    assert_eq!(summary.p99, 5, "p99 of 64 samples must include sample 64");
}
