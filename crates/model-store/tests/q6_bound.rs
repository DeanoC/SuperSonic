use model_store::q6_bound::{
    activation_block_norms, decode_q6_k_block, q8_1_reconstruct, required_exact_tiles,
    summarize_tile_counts, upward_f16_norm, weight_block_norms,
};

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
