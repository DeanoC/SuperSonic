use std::path::PathBuf;
use supersonic_bench::runs::{MetaJson, PerfCellJson, PerfStatus, RunDir};

#[test]
fn meta_json_round_trip() {
    let meta = MetaJson {
        schema_version: 2,
        run_id: "2026-05-05-abc1234".to_string(),
        timestamp_utc: "2026-05-05T12:00:00Z".to_string(),
        git_sha: "abc1234".to_string(),
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
        mpp_pilot: None,
        mps_expert_pilot: None,
        qwen36_pack_cache: None,
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
