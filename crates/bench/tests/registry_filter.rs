use supersonic_bench::matrix::{combos_for_arch, BenchArch};

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
    for arch in [BenchArch::Gfx1100, BenchArch::Sm86] {
        for c in combos_for_arch(arch) {
            assert!(c.min_vram_gib > 0.0, "combo {c:?} has zero min_vram_gib");
        }
    }
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
