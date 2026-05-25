use std::collections::BTreeMap;
use std::path::PathBuf;
use supersonic_bench::runs::{
    MetaJson, PerfCellJson, PerfStatus, Qwen36ExpertResidencyPolicyJson, RunDir,
};

#[test]
fn meta_json_round_trip() {
    let meta = MetaJson {
        schema_version: 2,
        run_id: "2026-05-05-abc1234".to_string(),
        timestamp_utc: "2026-05-05T12:00:00Z".to_string(),
        git_sha: "abc1234".to_string(),
        git_dirty: true,
        git_dirty_paths: vec!["crates/kernel-ffi/src/metal_native.mm".to_string()],
        git_diff_hash: Some("hash123".to_string()),
        hostname: "test-host".to_string(),
        arch: "gfx1100".to_string(),
        rocminfo: "Agent 1: gfx1100".to_string(),
        rocm_smi_u: "PID 1234 100%".to_string(),
        gpu_temp_c_pre: Some(45.0),
        gpu_temp_c_post: None,
        runner_version: "supersonic 0.1.0 (commit abc1234)".to_string(),
    };
    let s = serde_json::to_string(&meta).unwrap();
    let parsed: MetaJson = serde_json::from_str(&s).unwrap();
    assert_eq!(parsed.run_id, "2026-05-05-abc1234");
    assert!(parsed.git_dirty);
    assert_eq!(
        parsed.git_dirty_paths,
        vec!["crates/kernel-ffi/src/metal_native.mm".to_string()]
    );
    assert_eq!(parsed.git_diff_hash.as_deref(), Some("hash123"));
    assert_eq!(parsed.gpu_temp_c_pre, Some(45.0));
    assert_eq!(parsed.gpu_temp_c_post, None);
}

#[test]
fn perf_cell_json_status_variants() {
    let ok = PerfCellJson {
        schema_version: 2,
        model: "qwen3.5-0.8b".into(),
        quant: "bf16".into(),
        arch: "gfx1100".into(),
        backend: "hip".into(),
        prompt: "The quick brown fox jumps over".into(),
        max_new_tokens: 16,
        status: PerfStatus::Ok {
            ms_per_step: 8.0,
            ms_per_tok: 8.0,
            samples: vec![8.1, 8.0, 7.9],
        },
        stage_timings: None,
        chain_breakdown: None,
        lifecycle_timings: None,
        profile_stage_timings: None,
        profile_chain_breakdown: None,
        profile_lifecycle_timings: None,
        mpp_pilot: None,
        mps_expert_pilot: None,
        qwen36_pack_cache: None,
        qwen36_expert_residency: None,
        qwen36_expert_residency_policies: None,
        qwen36_expert_residency_policy_rows: None,
        metal_profile: None,
        hal_profile: None,
        gpu_temp_c_end: Some(60.0),
    };
    let s = serde_json::to_string(&ok).unwrap();
    let back: PerfCellJson = serde_json::from_str(&s).unwrap();
    match back.status {
        PerfStatus::Ok {
            ms_per_step,
            samples,
            ..
        } => {
            assert_eq!(ms_per_step, 8.0);
            assert_eq!(samples.len(), 3);
        }
        _ => panic!("expected Ok"),
    }

    let skipped = PerfCellJson {
        status: PerfStatus::Skipped {
            reason: "OOM at preflight".into(),
        },
        ..ok.clone()
    };
    let s = serde_json::to_string(&skipped).unwrap();
    assert!(s.contains("\"status\":\"skipped\""));
}

#[test]
fn perf_cell_preserves_qwen36_expert_residency_policy_labels() {
    let mut metrics = BTreeMap::new();
    metrics.insert("calls".to_string(), 160.0);
    metrics.insert("copied_bytes".to_string(), 2014248960.0);

    let cell = PerfCellJson {
        schema_version: 9,
        model: "qwen3.6-35b-a3b".into(),
        quant: "int4".into(),
        arch: "apple-m5-max".into(),
        backend: "metal".into(),
        prompt: "The quick brown fox jumps over".into(),
        max_new_tokens: 16,
        status: PerfStatus::Ok {
            ms_per_step: 150.6,
            ms_per_tok: 150.6,
            samples: vec![150.6],
        },
        stage_timings: None,
        chain_breakdown: None,
        lifecycle_timings: None,
        profile_stage_timings: None,
        profile_chain_breakdown: None,
        profile_lifecycle_timings: None,
        mpp_pilot: None,
        mps_expert_pilot: None,
        qwen36_pack_cache: None,
        qwen36_expert_residency: None,
        qwen36_expert_residency_policies: None,
        qwen36_expert_residency_policy_rows: Some(vec![Qwen36ExpertResidencyPolicyJson {
            resident_format: "native_int4".into(),
            scope: "per_layer".into(),
            miss_policy: "exact_route".into(),
            capacity: 8.0,
            metrics,
        }]),
        metal_profile: None,
        hal_profile: None,
        gpu_temp_c_end: None,
    };

    let s = serde_json::to_string(&cell).unwrap();
    assert!(s.contains("\"resident_format\":\"native_int4\""));
    assert!(s.contains("\"miss_policy\":\"exact_route\""));

    let parsed: PerfCellJson = serde_json::from_str(&s).unwrap();
    let rows = parsed.qwen36_expert_residency_policy_rows.unwrap();
    assert_eq!(rows[0].resident_format, "native_int4");
    assert_eq!(rows[0].scope, "per_layer");
    assert_eq!(rows[0].miss_policy, "exact_route");
    assert_eq!(rows[0].capacity, 8.0);
    assert_eq!(rows[0].metrics.get("copied_bytes"), Some(&2014248960.0));
}

#[test]
fn run_dir_paths() {
    let rd = RunDir::new(PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234"));
    assert_eq!(
        rd.meta_path(),
        PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234/meta.json")
    );
    assert_eq!(
        rd.perf_path("qwen3.5-0.8b", "bf16"),
        PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234/perf/qwen3.5-0.8b_bf16.json"),
    );
    assert_eq!(
        rd.external_path("hipfire", "qwen3.5-0.8b", "bf16"),
        PathBuf::from("/tmp/bench-runs/2026-05-05-abc1234/external/hipfire/qwen3.5-0.8b_bf16.json"),
    );
}
