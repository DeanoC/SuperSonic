#[cfg(supersonic_backend_hip)]
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn hip_profile_passes_sanity_floors() {
    let profile = machine_profile::measure();
    let gpu = profile
        .gpus
        .first()
        .expect("HIP profiler should report >=1 GPU");
    assert!(
        gpu.vram_bw.read_gb_s.unwrap() > 50.0,
        "VRAM read {:.1} GB/s below floor of 50 GB/s",
        gpu.vram_bw.read_gb_s.unwrap()
    );
    assert!(
        gpu.mma_peak.bf16.as_ref().unwrap().measured_tflops > 1.0,
        "BF16 MMA {:.1} TFLOPS below floor of 1 TFLOPS",
        gpu.mma_peak.bf16.as_ref().unwrap().measured_tflops
    );
    assert!(
        gpu.lds_bw_aggregate_gb_s.unwrap() > 1000.0,
        "LDS aggregate {:.1} GB/s below floor of 1000 GB/s",
        gpu.lds_bw_aggregate_gb_s.unwrap()
    );
}

#[cfg(supersonic_backend_metal)]
#[test]
#[ignore = "requires Apple Metal GPU; run with --ignored"]
fn metal_profile_passes_sanity_floors() {
    let profile = machine_profile::measure();
    let gpu = profile
        .gpus
        .iter()
        .find(|g| g.backend == "Metal")
        .expect("Metal profiler should report a GPU");
    assert_eq!(gpu.memory_arch, "Unified");
    assert!(
        gpu.total_vram_bytes > 0,
        "Metal working set should be nonzero"
    );
    assert!(gpu.cu_count > 0, "Metal GPU core count should be nonzero");
    assert!(
        gpu.vram_bw.read_gb_s.unwrap_or(0.0) > 1.0,
        "Metal read bandwidth missing or too low"
    );
    assert!(
        gpu.microkernels
            .iter()
            .any(|m| m.name == "qwen36.int4_gemv.lm_head" && m.measured_gb_s.unwrap_or(0.0) > 1.0),
        "Qwen3.6-shaped INT4 GEMV microkernel missing or too low"
    );
    assert!(
        gpu.metal
            .as_ref()
            .and_then(|m| m.simdgroup_matrix_supported)
            .unwrap_or(false),
        "Metal simdgroup matrix support should compile on this target"
    );
    assert!(
        gpu.mma_peak
            .f16
            .as_ref()
            .map(|m| m.measured_tflops)
            .unwrap_or(0.0)
            > 1.0,
        "Metal F16 simdgroup MMA profile missing or too low"
    );
    assert!(
        gpu.metal
            .as_ref()
            .and_then(|m| m.recommended_working_set_bytes)
            .unwrap_or(0)
            > 0,
        "Metal recommended working set should be captured"
    );
}
