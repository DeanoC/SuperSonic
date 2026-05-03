#[cfg(target_os = "linux")]
use std::process::Command;

#[cfg(target_os = "linux")]
fn model_dir() -> Option<String> {
    std::env::var("SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR").ok()
}

#[cfg(target_os = "linux")]
struct SmokeCase {
    name: &'static str,
    prompt: String,
    max_new_tokens: &'static str,
    context_size: &'static str,
    prefill_chunk_size: Option<&'static str>,
}

#[cfg(target_os = "linux")]
fn run_supersonic(
    backend: &str,
    model_dir: &str,
    case: &SmokeCase,
    vmm: &str,
    evict_after_prefill: bool,
    restore_to_vmm: bool,
) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.env("SUPERSONIC_VMM_KV", vmm).args([
        "--backend",
        backend,
        "--model",
        "qwen3.5-0.8b",
        "--model-dir",
        model_dir,
        "--prompt",
        &case.prompt,
        "--max-new-tokens",
        case.max_new_tokens,
        "--context-size",
        case.context_size,
        "--no-bake",
        "--no-download",
    ]);
    if evict_after_prefill {
        cmd.env("SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL", "1");
    }
    if restore_to_vmm {
        cmd.env("SUPERSONIC_VMM_KV_RESTORE_TO_VMM", "1");
    }
    if let Some(chunk_size) = case.prefill_chunk_size {
        cmd.args(["--prefill-chunk-size", chunk_size]);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("run supersonic qwen35 {backend} smoke: {e}"))
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
fn tokens_line(output: &str) -> &str {
    output
        .lines()
        .find(|line| line.starts_with("[tokens] "))
        .expect("supersonic output should contain a [tokens] line")
}

#[cfg(target_os = "linux")]
fn backend_vmm_supported(backend: &str) -> bool {
    let backend = match backend {
        "hip" => gpu_hal::Backend::Hip,
        "cuda" => gpu_hal::Backend::Cuda,
        other => panic!("unsupported smoke backend {other}"),
    };
    gpu_hal::set_backend(backend);
    gpu_hal::vmm_is_supported(backend, 0)
}

#[cfg(target_os = "linux")]
fn qwen35_virtual_kv_matches_dense_tokens_for_backend(backend: &str) {
    if !backend_vmm_supported(backend) {
        eprintln!("skipping: {backend} VMM unsupported on this device/runtime");
        return;
    }

    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR is not set");
        return;
    };

    let cases = [
        SmokeCase {
            name: "short",
            prompt: "Hello".to_string(),
            max_new_tokens: "2",
            context_size: "32",
            prefill_chunk_size: None,
        },
        SmokeCase {
            name: "chunked_prefill",
            prompt: "SuperSonic virtual memory should preserve KV cache contents across chunked prefill. "
                .repeat(12),
            max_new_tokens: "2",
            context_size: "384",
            prefill_chunk_size: Some("8"),
        },
        SmokeCase {
            name: "long_chunked_prefill",
            prompt: "A low level virtual memory system needs stable addresses, sparse residency, deterministic decode, and repeatable validation. "
                .repeat(20),
            max_new_tokens: "2",
            context_size: "768",
            prefill_chunk_size: Some("64"),
        },
    ];

    for case in cases {
        let dense = run_supersonic(backend, &model_dir, &case, "0", false, false);
        let dense_combined = combined_output(&dense);
        assert!(
            dense.status.success(),
            "dense Qwen3.5 {backend} smoke case={} failed with status {:?}:\n{}",
            case.name,
            dense.status.code(),
            dense_combined
        );
        assert!(
            !dense_combined.contains("[vmm] Qwen3.5 BF16 dense KV"),
            "dense fallback case={} unexpectedly enabled VMM:\n{}",
            case.name,
            dense_combined
        );

        let virtual_kv = run_supersonic(backend, &model_dir, &case, "1", false, false);
        let virtual_combined = combined_output(&virtual_kv);
        assert!(
            virtual_kv.status.success(),
            "virtual-KV Qwen3.5 {backend} smoke case={} failed with status {:?}:\n{}",
            case.name,
            virtual_kv.status.code(),
            virtual_combined
        );
        assert!(
            virtual_combined.contains("[vmm] Qwen3.5 BF16 dense KV uses reserved virtual memory"),
            "virtual-KV case={} did not report VMM activation:\n{}",
            case.name,
            virtual_combined
        );
        assert!(
            virtual_combined.contains("[vmm] virtual KV logical="),
            "virtual-KV case={} did not report resident/reserved telemetry:\n{}",
            case.name,
            virtual_combined
        );

        assert_eq!(
            tokens_line(&virtual_combined),
            tokens_line(&dense_combined),
            "virtual-KV and dense fallback generated different token IDs for case={}",
            case.name
        );

        if case.name == "chunked_prefill" {
            let evicted = run_supersonic(backend, &model_dir, &case, "1", true, false);
            let evicted_combined = combined_output(&evicted);
            assert!(
                evicted.status.success(),
                "virtual-KV eviction smoke case={} failed with status {:?}:\n{}",
                case.name,
                evicted.status.code(),
                evicted_combined
            );
            assert!(
                evicted_combined.contains("[vmm] evicted virtual KV to host logical_backup=")
                    && evicted_combined.contains("resident=0.00MiB"),
                "virtual-KV eviction case={} did not report zero resident bytes:\n{}",
                case.name,
                evicted_combined
            );
            assert!(
                evicted_combined.contains("[vmm] restored virtual KV from host logical_resident="),
                "virtual-KV eviction case={} did not report restore:\n{}",
                case.name,
                evicted_combined
            );
            assert_eq!(
                tokens_line(&evicted_combined),
                tokens_line(&dense_combined),
                "evicted/restored virtual-KV and dense fallback generated different token IDs for case={}",
                case.name
            );

            let restored_vmm = run_supersonic(backend, &model_dir, &case, "1", true, true);
            let restored_vmm_combined = combined_output(&restored_vmm);
            assert!(
                restored_vmm.status.success(),
                "virtual-KV restore-to-VMM smoke case={} failed with status {:?}:\n{}",
                case.name,
                restored_vmm.status.code(),
                restored_vmm_combined
            );
            assert!(
                restored_vmm_combined
                    .contains("[vmm] restored virtual KV from host logical_resident=")
                    && restored_vmm_combined.contains("resident=24.00MiB"),
                "restore-to-VMM case={} did not report remapped resident bytes:\n{}",
                case.name,
                restored_vmm_combined
            );
            assert_eq!(
                tokens_line(&restored_vmm_combined),
                tokens_line(&dense_combined),
                "restore-to-VMM virtual-KV and dense fallback generated different token IDs for case={}",
                case.name
            );
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires HIP and a local Qwen3.5-0.8B model dir via SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR"]
fn qwen35_hip_virtual_kv_matches_dense_tokens() {
    qwen35_virtual_kv_matches_dense_tokens_for_backend("hip");
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires CUDA and a local Qwen3.5-0.8B model dir via SUPERSONIC_TEST_QWEN35_08B_MODEL_DIR"]
fn qwen35_cuda_virtual_kv_matches_dense_tokens() {
    qwen35_virtual_kv_matches_dense_tokens_for_backend("cuda");
}
