#[cfg(supersonic_backend_hip)]
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn hip_profile_passes_sanity_floors() {
    let profile = machine_profile::measure();
    let gpu = profile.gpus.first().expect("HIP profiler should report >=1 GPU");
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
