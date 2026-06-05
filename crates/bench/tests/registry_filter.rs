use supersonic_bench::matrix::{
    combos_for_arch, is_supported_combo, lower_precision_candidate,
    lower_precision_candidates_for_arch, BenchArch,
};

#[test]
fn gfx1100_includes_shipping_models() {
    let combos = combos_for_arch(BenchArch::Gfx1100);
    let model_quants: Vec<(&str, &str)> = combos.iter().map(|c| (c.model, c.quant)).collect();

    assert!(model_quants.contains(&("qwen3.5-0.8b", "bf16")));
    assert!(model_quants.contains(&("qwen3.5-0.8b", "int4")));
    assert!(model_quants.contains(&("gemma4-e2b", "bf16")));
    assert!(model_quants.contains(&("phi4-mini", "fp8r")));
    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4")));
    assert!(
        !model_quants.contains(&("qwen3.6-35b-a3b", "bf16")),
        "qwen3.6-35b-a3b BF16 is not supported on gfx1100 (24 GiB cap)"
    );
}

#[test]
fn min_vram_set_for_every_combo() {
    for arch in [BenchArch::Gfx1100, BenchArch::Sm86, BenchArch::AppleM5Max] {
        for c in combos_for_arch(arch) {
            assert!(c.min_vram_gib > 0.0, "combo {c:?} has zero min_vram_gib");
        }
    }
}

#[test]
fn apple_m5_max_includes_qwen_moe_metal_lanes() {
    let combos = combos_for_arch(BenchArch::AppleM5Max);
    let model_quants: Vec<(&str, &str)> = combos.iter().map(|c| (c.model, c.quant)).collect();

    assert_eq!(BenchArch::AppleM5Max.backend(), Some("metal"));
    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4")));
    assert!(model_quants.contains(&("qwen3.5-35b-a3b", "q4km-gptq")));
    assert!(model_quants.contains(&("qwen3.5-35b-a3b", "q4km")));
    assert!(!model_quants.contains(&("qwen3.6-35b-a3b", "kv-fp8")));
}

#[test]
fn apple_m5_max_tracks_lower_precision_candidates_separately() {
    let candidates = lower_precision_candidates_for_arch(BenchArch::AppleM5Max);
    let model_quants: Vec<(&str, &str)> = candidates.iter().map(|c| (c.model, c.quant)).collect();

    assert!(model_quants.contains(&("qwen3.5-0.8b", "int3")));
    assert!(model_quants.contains(&("qwen3.5-0.8b", "int2-4-mixed")));
    assert!(model_quants.contains(&("qwen3.5-0.8b", "mxfp4")));
    assert!(model_quants.contains(&("qwen3.5-35b-a3b", "int2-4-mixed")));
    assert!(
        !is_supported_combo("qwen3.5-0.8b", "int2-4-mixed", &BenchArch::AppleM5Max),
        "lower-precision probes must stay out of SUPPORTED_COMBOS until runtime support exists"
    );

    let artifact =
        lower_precision_candidate("qwen3.5-0.8b", "int2-4-mixed", &BenchArch::AppleM5Max)
            .expect("candidate should be registered")
            .quant_artifact();
    assert_eq!(artifact.profile, "autoround-int2-4-mixed");
    assert_eq!(artifact.average_bits_per_weight, Some(3.0));
    assert!(!artifact.runtime_supported);
}

#[test]
fn sm86_includes_qwen36_specprefill_lanes() {
    let combos = combos_for_arch(BenchArch::Sm86);
    let model_quants: Vec<(&str, &str)> = combos.iter().map(|c| (c.model, c.quant)).collect();

    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4")));
    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4-spec025")));
    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4-spec050")));
    assert!(model_quants.contains(&("qwen3.6-35b-a3b", "int4-spec075")));
}

#[test]
fn sm86_accepts_ad_hoc_qwen36_specprefill_lanes() {
    assert!(is_supported_combo(
        "qwen3.6-35b-a3b",
        "int4-spec070",
        &BenchArch::Sm86
    ));
    assert!(is_supported_combo(
        "qwen3.6-35b-a3b",
        "int4-spec100",
        &BenchArch::Sm86
    ));
    assert!(!is_supported_combo(
        "qwen3.6-35b-a3b",
        "int4-spec004",
        &BenchArch::Sm86
    ));
    assert!(!is_supported_combo(
        "qwen3.6-35b-a3b",
        "int4-spec070",
        &BenchArch::Gfx1100
    ));
    assert!(!is_supported_combo(
        "qwen3.5-9b",
        "int4-spec070",
        &BenchArch::Sm86
    ));
}
