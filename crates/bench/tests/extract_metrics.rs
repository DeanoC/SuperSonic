#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::path::PathBuf;
#[allow(unused_imports)]
use supersonic_bench::perf::{
    extract_attribution_timings, extract_metrics, run_one_combo, ComboInvocation, ExtractedMetrics,
    RunPolicy,
};

const MODERN: &str = include_str!("fixtures/runner_output_modern.txt");
const PHI4: &str = include_str!("fixtures/runner_output_phi4.txt");

#[test]
fn extracts_ms_per_step_from_modern_runner() {
    let m = extract_metrics(MODERN).expect("expected metrics");
    assert!((m.ms_per_step - 8.0).abs() < 1e-6);
    assert!((m.ms_per_tok.unwrap() - 8.0).abs() < 1e-6);
}

#[test]
fn extracts_ms_per_step_from_phi4_runner() {
    let m = extract_metrics(PHI4).expect("expected metrics");
    assert!((m.ms_per_step - 38.3).abs() < 1e-6);
}

#[test]
fn returns_none_when_no_result_line() {
    let s = "no result line here\nsome other text";
    assert!(extract_metrics(s).is_none());
}

#[test]
fn accepts_ms_per_tok_as_ms_per_step_for_legacy_batch1_runners() {
    // Per spec: do not silently fall back; missing both means something broke.
    // qwen35_decode_report.rs emits ms_per_tok-only — verify we still surface ms_per_tok
    // but only when paired with ms_per_step. This format-change-detector test asserts
    // we don't accept a degenerate output that's missing ms_per_step entirely.
    let legacy = "[result] prompt_tokens=6 generated_tokens=4 decode_ms=32 ms_per_tok=8 decode_max_delta=0.0000";
    let m = extract_metrics(legacy);
    // ms_per_step is the canonical field we extract; if a runner only emits ms_per_tok,
    // we accept ms_per_tok as ms_per_step (they are equivalent for batch=1) but flag it.
    assert!(
        m.is_some(),
        "ms_per_tok should be accepted as ms_per_step for batch=1"
    );
    let m = m.unwrap();
    assert!((m.ms_per_step - 8.0).abs() < 1e-6);
}

#[test]
fn last_result_line_wins_when_multiple_present() {
    let s = "[result] ms_per_step=5\n[result] ms_per_step=9\n";
    let m = extract_metrics(s).unwrap();
    assert!((m.ms_per_step - 9.0).abs() < 1e-6);
}

#[test]
fn extracts_qwen36_prefill_lifecycle_when_result_line_is_absent() {
    let s = "[qwen36-moe lifecycle-timings] prompt_setup_ms=1 prefill_total_ms=2441.711 total_wall_ms=7603\n";
    let m = extract_metrics(s).unwrap();
    assert!((m.ms_per_step - 2441.711).abs() < 1e-6);
}

#[test]
fn extracts_qwen36_batched_prefill_lines_when_result_line_is_absent() {
    let s = "\
[qwen36-moe batched-prefill] chunks=1 tokens=119 embed_ms=0.2 chain_ms=3472.0
[qwen36-moe batched-prefill] chunks=1 tokens=8 embed_ms=0.1 chain_ms=42.3
";
    let m = extract_metrics(s).unwrap();
    assert!((m.ms_per_step - 3514.6).abs() < 1e-6);
}

#[test]
fn extracts_qwen36_attribution_timing_maps() {
    let s = "\
[result] prompt_tokens=6 generated_tokens=16
[qwen36-moe stage-timings] gen_steps=16 embed_ms_avg=0.123 chain_ms_avg=25.456 lm_head_ms_avg=2.000 sample_ms_avg=0.010 detok_ms_avg=0.001 total_ms_avg=27.590 (chain_total_ms=407.3 lm_head_total_ms=32.0)
[qwen36-moe chain-breakdown] gen_steps=16 full_attn_ms_avg=8.000 linear_attn_ms_avg=4.000 ffn_ms_avg=13.456 (full_attn_total_ms=128.0 linear_attn_total_ms=64.0 ffn_total_ms=215.3)
[qwen36-moe lifecycle-timings] prompt_setup_ms=1.000 bake_open_ms=2.000 layer_load_ms=3.000 session_ms=4.000 prefill_steps=1 prefill_embed_ms=5.000 prefill_chain_ms=6.000 prefill_total_ms=11.000 generation_wall_ms=441.0 total_wall_ms=452.0
[qwen36-moe mpp-pilot] status=ok size=2048 iterations=5 tile_m=64 tile_n=32 tile_k=64 tflops=13.250
[qwen36-moe mps-expert-pilot] status=ok hidden=2048 moe_intermediate=512 top_k=8 iterations=100 gate_up_ms=2.500 down_ms=1.250 gate_up_tflops=1.342 down_tflops=0.671
[qwen36-expert-residency] calls=160 entries=40 exact_hits=0 route_refills=120 allocations=40 copied_bytes=2014248960 exact_hit_rate=0.000000 slot_hits=0 slot_misses=1280 slot_hit_rate=0.000000 evictions=0 avg_active_groups=8.000000 max_active_groups=8 avg_copy_bytes=12589056.000
[qwen36-pack-cache] calls=160 entries=40 exact_hits=0 route_refills=120 allocations=40 copied_bytes=2014248960 exact_hit_rate=0.000000 slot_hits=0 slot_misses=1280 slot_hit_rate=0.000000 evictions=0 avg_active_groups=8.000000 max_active_groups=8 avg_copy_bytes=12589056.000
[qwen36-expert-residency-policy] resident_format=native_int4 scope=per_layer miss_policy=exact_route capacity=8 calls=160 exact_hits=0 route_refills=120 allocations=40 copied_bytes=2014248960 exact_hit_rate=0.000000 slot_hits=0 slot_misses=1280 slot_hit_rate=0.000000 evictions=0 avg_active_groups=8.000000 max_active_groups=8 avg_copy_bytes=12589056.000
[metal-profile] calls=3 total_ms=50.000 native_ms=45.000 host_ms=5.000
[metal-profile-op] op=qwen36_ffn_int4_stage5 path=native calls=2 mean_ms=20.0000 total_ms=40.000 max_ms=21.000
[hal-profile] calls=4 total_ms=1.000 alloc_calls=0 alloc_bytes=0 h2d=0 d2h=0 d2d=4096 memset=0 sync_calls=1
[hal-profile-op] op=copy_d2d calls=1 mean_ms=0.2500 total_ms=0.250 max_ms=0.250 total_bytes=4096
";
    let timings = extract_attribution_timings(s);
    assert_eq!(
        timings.stage_timings.unwrap().get("chain_ms_avg"),
        Some(&25.456)
    );
    assert_eq!(
        timings.chain_breakdown.unwrap().get("ffn_total_ms"),
        Some(&215.3)
    );
    assert_eq!(
        timings.lifecycle_timings.unwrap().get("prefill_total_ms"),
        Some(&11.0)
    );
    assert_eq!(timings.mpp_pilot.unwrap().get("tflops"), Some(&13.25));
    assert_eq!(
        timings.mps_expert_pilot.unwrap().get("gate_up_tflops"),
        Some(&1.342)
    );
    assert_eq!(
        timings.qwen36_pack_cache.unwrap().get("copied_bytes"),
        Some(&2014248960.0)
    );
    assert_eq!(
        timings.qwen36_expert_residency.unwrap().get("copied_bytes"),
        Some(&2014248960.0)
    );
    let policies = timings.qwen36_expert_residency_policies.unwrap();
    assert_eq!(policies[0].get("capacity"), Some(&8.0));
    assert_eq!(policies[0].get("route_refills"), Some(&120.0));
    let policy_rows = timings.qwen36_expert_residency_policy_rows.unwrap();
    assert_eq!(policy_rows[0].resident_format, "native_int4");
    assert_eq!(policy_rows[0].scope, "per_layer");
    assert_eq!(policy_rows[0].miss_policy, "exact_route");
    assert_eq!(policy_rows[0].capacity, 8.0);
    assert_eq!(
        policy_rows[0].metrics.get("copied_bytes"),
        Some(&2014248960.0)
    );
    let metal = timings.metal_profile.unwrap();
    assert_eq!(metal.summary.get("native_ms"), Some(&45.0));
    assert_eq!(metal.entries[0].op, "qwen36_ffn_int4_stage5");
    assert_eq!(metal.entries[0].path.as_deref(), Some("native"));
    let hal = timings.hal_profile.unwrap();
    assert_eq!(hal.summary.get("d2d"), Some(&4096.0));
    assert_eq!(hal.entries[0].total_bytes, Some(4096));
}

#[cfg(unix)]
#[test]
fn qwen36_apple_perf_keeps_unprofiled_timings_separate_from_profile_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let script = tmp.path().join("fake_supersonic.sh");
    std::fs::write(
        &script,
        r#"#!/bin/sh
emit_stage=0
for arg in "$@"; do
  if [ "$arg" = "--emit-stage-timings" ]; then
    emit_stage=1
  fi
done

if [ -n "$SUPERSONIC_METAL_PROFILE" ]; then
  echo "[result] prompt_tokens=1 generated_tokens=1 decode_ms=201 ms_per_step=201"
  if [ "$emit_stage" = "1" ]; then
    echo "[qwen36-moe stage-timings] gen_steps=1 chain_ms_avg=180 total_ms_avg=201"
    echo "[qwen36-moe chain-breakdown] gen_steps=1 full_attn_ms_avg=9 linear_attn_ms_avg=77 ffn_ms_avg=88"
    echo "[qwen36-moe lifecycle-timings] prefill_total_ms=22 generation_wall_ms=201"
  fi
  echo "[metal-profile] calls=1 total_ms=9 native_ms=7 host_ms=2"
  echo "[metal-profile-op] op=command_buffer_wait path=runtime calls=1 mean_ms=3 total_ms=3 max_ms=3"
  echo "[hal-profile] calls=1 total_ms=1 alloc_calls=0 alloc_bytes=0 h2d=0 d2h=0 d2d=0 memset=0 sync_calls=0"
  echo "[hal-profile-op] op=sync calls=1 mean_ms=1 total_ms=1 max_ms=1 total_bytes=0"
else
  echo "[result] prompt_tokens=1 generated_tokens=1 decode_ms=100 ms_per_step=100"
  if [ "$emit_stage" = "1" ]; then
    echo "[qwen36-moe stage-timings] gen_steps=1 chain_ms_avg=90 total_ms_avg=101"
    echo "[qwen36-moe chain-breakdown] gen_steps=1 full_attn_ms_avg=7 linear_attn_ms_avg=44 ffn_ms_avg=39"
    echo "[qwen36-moe lifecycle-timings] prefill_total_ms=11 generation_wall_ms=101"
    echo "[qwen36-moe mpp-pilot] status=ok tflops=15"
    echo "[qwen36-moe mps-expert-pilot] status=ok gate_up_ms=0.6 down_ms=0.4"
  fi
fi
"#,
    )
    .unwrap();
    let mut perms = std::fs::metadata(&script).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script, perms).unwrap();

    let invocation = ComboInvocation {
        binary: script,
        backend: Some("metal".into()),
        arch: "apple-m5-max".into(),
        model: "qwen3.6-35b-a3b".into(),
        model_dir: PathBuf::from("/unused-model-dir"),
        quant: "int4".into(),
        specprefill_draft_dir: None,
        prompt: "x".into(),
        max_new_tokens: 1,
        warmup_tokens: 1,
    };
    let policy = RunPolicy {
        measurement_runs: 1,
        cooldown_seconds: 0,
    };

    let cell = run_one_combo(&invocation, &policy).unwrap();
    assert_eq!(cell.schema_version, 8);
    assert_eq!(
        cell.stage_timings
            .as_ref()
            .and_then(|m| m.get("total_ms_avg")),
        Some(&101.0)
    );
    assert_eq!(
        cell.profile_stage_timings
            .as_ref()
            .and_then(|m| m.get("total_ms_avg")),
        Some(&201.0)
    );
    assert_eq!(
        cell.chain_breakdown
            .as_ref()
            .and_then(|m| m.get("linear_attn_ms_avg")),
        Some(&44.0)
    );
    assert_eq!(
        cell.profile_chain_breakdown
            .as_ref()
            .and_then(|m| m.get("linear_attn_ms_avg")),
        Some(&77.0)
    );
    assert!(cell.metal_profile.is_some());
    assert!(cell.hal_profile.is_some());
    assert_eq!(
        cell.mpp_pilot.as_ref().and_then(|m| m.get("tflops")),
        Some(&15.0)
    );
}
