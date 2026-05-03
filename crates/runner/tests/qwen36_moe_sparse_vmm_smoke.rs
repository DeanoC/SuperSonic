#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(target_os = "linux")]
use std::process::Command;

#[cfg(target_os = "linux")]
fn model_dir() -> Option<String> {
    std::env::var("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR")
        .or_else(|_| std::env::var("SUPERSONIC_QWEN36_MTP_MODEL_DIR"))
        .ok()
}

#[cfg(target_os = "linux")]
fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[cfg(target_os = "linux")]
fn generated_ids(output: &str) -> Vec<u32> {
    let line = output
        .lines()
        .find(|line| line.trim_start().starts_with("Generated ids: "))
        .expect("supersonic output should contain generated ids");
    let (_, ids) = line
        .split_once(':')
        .expect("generated ids line should contain ':'");
    serde_json::from_str(ids.trim()).expect("generated ids should parse as a JSON array")
}

#[cfg(target_os = "linux")]
fn hip_vmm_supported() -> bool {
    gpu_hal::set_backend(gpu_hal::Backend::Hip);
    gpu_hal::vmm_is_supported(gpu_hal::Backend::Hip, 0)
}

#[cfg(target_os = "linux")]
fn sparse_cap_experts() -> usize {
    std::env::var("SUPERSONIC_TEST_QWEN36_MOE_SPARSE_CAP_EXPERTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(256)
}

#[cfg(target_os = "linux")]
fn run_qwen36_moe(
    model_dir: &str,
    sparse_cap_experts: Option<usize>,
    telemetry_path: Option<&Path>,
) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.env("SUPERSONIC_BACKENDS", "hip")
        .env("SUPERSONIC_VMM_MOE_ISLANDS", "1")
        .env_remove("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS")
        .env_remove("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON")
        .args([
            "--backend",
            "hip",
            "--model",
            "qwen3.6-35b-a3b",
            "--model-dir",
            model_dir,
            "--int4",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "2",
            "--context-size",
            "16",
            "--temperature",
            "0",
            "--no-download",
        ]);
    if let Some(cap) = sparse_cap_experts {
        cmd.env("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS", cap.to_string());
    }
    if let Some(path) = telemetry_path {
        cmd.env("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON", path);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("run supersonic qwen36 sparse VMM smoke: {e}"))
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires HIP, VMM, and a local Qwen3.6-35B-A3B dir via SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"]
fn qwen36_moe_sparse_vmm_matches_dense_virtual_slabs() {
    if !hip_vmm_supported() {
        eprintln!("skipping: HIP VMM unsupported on this device/runtime");
        return;
    }

    let Some(model_dir) = model_dir() else {
        eprintln!(
            "skipping: SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR/SUPERSONIC_QWEN36_MTP_MODEL_DIR not set"
        );
        return;
    };

    let dense = run_qwen36_moe(&model_dir, None, None);
    let dense_combined = combined_output(&dense);
    assert!(
        dense.status.success(),
        "dense virtual-slab Qwen3.6-MoE smoke failed with status {:?}:\n{}",
        dense.status.code(),
        dense_combined
    );
    assert!(
        dense_combined.contains("[vmm] Qwen3.6-MoE routed expert slabs active"),
        "dense run did not report VMM expert slabs:\n{}",
        dense_combined
    );
    let dense_ids = generated_ids(&dense_combined);

    let temp = tempfile::tempdir().expect("create sparse telemetry tempdir");
    let telemetry_path = temp.path().join("qwen36_sparse_telemetry.json");
    let cap_experts = sparse_cap_experts();
    let sparse = run_qwen36_moe(&model_dir, Some(cap_experts), Some(&telemetry_path));
    let sparse_combined = combined_output(&sparse);
    assert!(
        sparse.status.success(),
        "sparse VMM Qwen3.6-MoE smoke failed with status {:?}:\n{}",
        sparse.status.code(),
        sparse_combined
    );
    assert!(
        sparse_combined.contains("[vmm] Qwen3.6-MoE sparse routed expert residency active")
            && sparse_combined.contains("peak_slices="),
        "sparse run did not report sparse residency telemetry:\n{}",
        sparse_combined
    );
    assert_eq!(
        generated_ids(&sparse_combined),
        dense_ids,
        "sparse VMM and dense virtual-slab VMM generated different token IDs"
    );

    let raw = std::fs::read_to_string(&telemetry_path).expect("read sparse telemetry JSON");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse sparse telemetry JSON");
    assert_eq!(
        json["schema"],
        "supersonic-qwen36-moe-sparse-vmm-telemetry-v1"
    );
    let summary = &json["summary"];
    let max_slices = (cap_experts * 2) as u64;
    assert!(
        summary["peak_resident_slices"].as_u64().unwrap() <= max_slices,
        "peak slices exceeded sparse cap: {summary:?}"
    );
    assert!(
        summary["final_resident_slices"].as_u64().unwrap() <= max_slices,
        "final slices exceeded sparse cap: {summary:?}"
    );
    assert!(
        summary["peak_resident_bytes"].as_u64().unwrap()
            < summary["reserved_bytes"].as_u64().unwrap(),
        "sparse resident bytes should stay below full VA reservation: {summary:?}"
    );
    assert!(
        summary["misses"].as_u64().unwrap() > 0 && summary["uploaded_bytes"].as_u64().unwrap() > 0,
        "sparse run did not exercise uploads/misses: {summary:?}"
    );
    assert!(
        json["steps"].as_array().unwrap().len() >= 2,
        "sparse telemetry should include multiple forward steps"
    );
    assert!(
        json["steps"]
            .as_array()
            .unwrap()
            .iter()
            .any(|step| step["kind"] == "generate"),
        "sparse telemetry should mark generation steps"
    );
}
