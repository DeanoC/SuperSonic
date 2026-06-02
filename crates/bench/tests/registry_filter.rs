use supersonic_bench::matrix::{combos_for_arch, is_supported_combo, BenchArch};

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
    assert!(
        !model_quants.contains(&("qwen3.5-35b-a3b", "q4km")),
        "raw GGUF Q4_K_M should be added only after local correctness and 512-token benchmark evidence is recorded"
    );
    assert!(!model_quants.contains(&("qwen3.6-35b-a3b", "kv-fp8")));
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
