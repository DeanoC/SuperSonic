use crate::gpu::GpuProfileError;
use crate::schema::*;

pub struct MetalProfiler;
impl crate::gpu::GpuProfiler for MetalProfiler {
    fn profile(&self) -> Result<Vec<GpuProfile>, GpuProfileError> {
        profile()
    }
}

#[cfg(supersonic_backend_metal)]
mod ffi {
    use std::ffi::{c_char, c_int};

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct MpMetalDeviceInfo {
        pub total_vram_bytes: u64,
        pub recommended_working_set_bytes: u64,
        pub core_count: u32,
        pub wave_size: u32,
        pub max_threadgroup_memory_bytes: u64,
        pub max_threads_per_threadgroup: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct MpMetalMppProbeInfo {
        pub status: i32,
        pub tensor_write_value: f32,
        pub matmul_value: f32,
    }

    unsafe extern "C" {
        pub fn mp_metal_query_device_info(
            arch_name_out: *mut c_char,
            arch_name_len: usize,
            device_name_out: *mut c_char,
            device_name_len: usize,
            family_out: *mut c_char,
            family_len: usize,
            info_out: *mut MpMetalDeviceInfo,
        ) -> c_int;
        pub fn mp_metal_unified_read_gb_s(bytes: u64) -> f64;
        pub fn mp_metal_unified_write_gb_s(bytes: u64) -> f64;
        pub fn mp_metal_unified_copy_gb_s(bytes: u64) -> f64;
        pub fn mp_metal_threadgroup_gb_s(core_count: u32, iterations: u32) -> f64;
        pub fn mp_metal_simdgroup_matrix_probe() -> c_int;
        pub fn mp_metal_mpp_tensor_matmul_probe_detail(
            info_out: *mut MpMetalMppProbeInfo,
            status_out: *mut c_char,
            status_len: usize,
        ) -> c_int;
        pub fn mp_metal_simdgroup_mma_f16_tflops(
            core_count: u32,
            threadgroups_per_core: u32,
            iterations: u32,
        ) -> f64;
        pub fn mp_metal_simdgroup_mma_f16_accum_f16_tflops(
            core_count: u32,
            threadgroups_per_core: u32,
            iterations: u32,
        ) -> f64;
        pub fn mp_metal_simdgroup_mma_f16_sweep_tflops(
            core_count: u32,
            threadgroups_per_core: u32,
            simdgroups_per_threadgroup: u32,
            accumulators: u32,
            iterations: u32,
            f16_accum: u32,
        ) -> f64;
        pub fn mp_metal_simdgroup_gemm_f16_tflops(
            size: u32,
            iterations: u32,
            simdgroups_per_threadgroup: u32,
        ) -> f64;
        pub fn mp_metal_mpp_tensor_gemm_f16_tflops(size: u32, iterations: u32) -> f64;
        pub fn mp_metal_mps_gemm_f16_tflops(size: u32, iterations: u32) -> f64;
        pub fn mp_metal_int4_gemv_gb_s(
            in_dim: u32,
            out_dim: u32,
            group_size: u32,
            iterations: u32,
        ) -> f64;
    }
}

#[cfg(supersonic_backend_metal)]
#[derive(Debug, Clone)]
pub(crate) struct MetalDeviceInfo {
    pub arch_name: String,
    pub device_name: String,
    pub metal_family: String,
    pub total_vram_bytes: u64,
    pub recommended_working_set_bytes: u64,
    pub core_count: u32,
    pub wave_size: u32,
    pub max_threadgroup_memory_bytes: u64,
    pub max_threads_per_threadgroup: u64,
}

#[cfg(supersonic_backend_metal)]
pub(crate) fn enumerate_for_fingerprint() -> Result<Vec<MetalDeviceInfo>, GpuProfileError> {
    query_device_info().map(|info| vec![info])
}

#[cfg(supersonic_backend_metal)]
fn query_device_info() -> Result<MetalDeviceInfo, GpuProfileError> {
    let mut arch = [0i8; 128];
    let mut device = [0i8; 128];
    let mut family = [0i8; 128];
    let mut info = ffi::MpMetalDeviceInfo {
        total_vram_bytes: 0,
        recommended_working_set_bytes: 0,
        core_count: 0,
        wave_size: 0,
        max_threadgroup_memory_bytes: 0,
        max_threads_per_threadgroup: 0,
    };
    let status = unsafe {
        ffi::mp_metal_query_device_info(
            arch.as_mut_ptr(),
            arch.len(),
            device.as_mut_ptr(),
            device.len(),
            family.as_mut_ptr(),
            family.len(),
            &mut info,
        )
    };
    if status != 0 {
        return Err(GpuProfileError::Metal(format!(
            "mp_metal_query_device_info returned {status}"
        )));
    }

    Ok(MetalDeviceInfo {
        arch_name: c_buf_to_string(&arch),
        device_name: c_buf_to_string(&device),
        metal_family: c_buf_to_string(&family),
        total_vram_bytes: info.total_vram_bytes,
        recommended_working_set_bytes: info.recommended_working_set_bytes,
        core_count: info.core_count,
        wave_size: info.wave_size,
        max_threadgroup_memory_bytes: info.max_threadgroup_memory_bytes,
        max_threads_per_threadgroup: info.max_threads_per_threadgroup,
    })
}

#[cfg(supersonic_backend_metal)]
fn c_buf_to_string(buf: &[i8]) -> String {
    let nul = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..nul].iter().map(|&c| c as u8).collect::<Vec<_>>()).to_string()
}

#[cfg(supersonic_backend_metal)]
fn profile() -> Result<Vec<GpuProfile>, GpuProfileError> {
    let info = query_device_info()?;
    let core_count = info.core_count.max(1);
    let bytes = 64u64 << 20;
    let read = positive(unsafe { ffi::mp_metal_unified_read_gb_s(bytes) });
    let write = positive(unsafe { ffi::mp_metal_unified_write_gb_s(bytes) });
    let copy = positive(unsafe { ffi::mp_metal_unified_copy_gb_s(bytes) });
    let threadgroup = positive(unsafe { ffi::mp_metal_threadgroup_gb_s(core_count, 4096) });
    let simd_probe = unsafe { ffi::mp_metal_simdgroup_matrix_probe() } == 0;
    let mut mpp_probe_info = ffi::MpMetalMppProbeInfo {
        status: 0,
        tensor_write_value: 0.0,
        matmul_value: 0.0,
    };
    let mut mpp_probe_status = [0i8; 128];
    let mpp_probe_code = unsafe {
        ffi::mp_metal_mpp_tensor_matmul_probe_detail(
            &mut mpp_probe_info,
            mpp_probe_status.as_mut_ptr(),
            mpp_probe_status.len(),
        )
    };
    let mpp_probe = mpp_probe_code == 0;
    let mma_threadgroups_per_core = 256;
    let mma_iterations = 4096;
    let f16_mma = simd_probe
        .then(|| {
            positive(unsafe {
                ffi::mp_metal_simdgroup_mma_f16_tflops(
                    core_count,
                    mma_threadgroups_per_core,
                    mma_iterations,
                )
            })
        })
        .flatten();
    let f16_accum_f16_mma = simd_probe
        .then(|| {
            positive(unsafe {
                ffi::mp_metal_simdgroup_mma_f16_accum_f16_tflops(
                    core_count,
                    mma_threadgroups_per_core,
                    mma_iterations,
                )
            })
        })
        .flatten();
    let sweep_iterations = 4096;
    let sweep_configs = [
        ("f32acc.tgpc128.sg8.acc16", 128, 8, 16, false),
        ("f32acc.tgpc256.sg8.acc16", 256, 8, 16, false),
        ("f32acc.tgpc256.sg16.acc16", 256, 16, 16, false),
        ("f32acc.tgpc256.sg32.acc16", 256, 32, 16, false),
        ("f32acc.tgpc512.sg16.acc16", 512, 16, 16, false),
        ("f32acc.tgpc512.sg32.acc16", 512, 32, 16, false),
        ("f32acc.tgpc256.sg16.acc8", 256, 16, 8, false),
        ("f32acc.tgpc256.sg16.acc24", 256, 16, 24, false),
        ("f32acc.tgpc256.sg16.acc32", 256, 16, 32, false),
        ("f16acc.tgpc256.sg16.acc16", 256, 16, 16, true),
        ("f16acc.tgpc256.sg32.acc16", 256, 32, 16, true),
    ];
    let simd_sweep = if simd_probe {
        sweep_configs
            .into_iter()
            .map(|(label, tgpc, sgtg, acc, f16_accum)| {
                let tflops = positive(unsafe {
                    ffi::mp_metal_simdgroup_mma_f16_sweep_tflops(
                        core_count,
                        tgpc,
                        sgtg,
                        acc,
                        sweep_iterations,
                        u32::from(f16_accum),
                    )
                });
                (label, tgpc, sgtg, acc, f16_accum, tflops)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mps_gemm_size = 4096;
    let mps_gemm_iterations = 3;
    let simd_gemm_size = 2048;
    let simd_gemm_iterations = 2;
    let simd_gemm_configs = [8u32, 16, 32];
    let simd_gemm = if simd_probe {
        simd_gemm_configs
            .into_iter()
            .map(|simdgroups_per_threadgroup| {
                let tflops = positive(unsafe {
                    ffi::mp_metal_simdgroup_gemm_f16_tflops(
                        simd_gemm_size,
                        simd_gemm_iterations,
                        simdgroups_per_threadgroup,
                    )
                });
                (simdgroups_per_threadgroup, tflops)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mps_gemm_2048_f16 = positive(unsafe {
        ffi::mp_metal_mps_gemm_f16_tflops(simd_gemm_size, simd_gemm_iterations)
    });
    let mpp_tensor_gemm_2048_f16 = if mpp_probe {
        positive(unsafe {
            ffi::mp_metal_mpp_tensor_gemm_f16_tflops(simd_gemm_size, simd_gemm_iterations)
        })
    } else {
        None
    };
    let mpp_tensor_gemm_4096_f16 = if mpp_probe {
        positive(unsafe { ffi::mp_metal_mpp_tensor_gemm_f16_tflops(mps_gemm_size, 1) })
    } else {
        None
    };
    let mps_gemm_f16 =
        positive(unsafe { ffi::mp_metal_mps_gemm_f16_tflops(mps_gemm_size, mps_gemm_iterations) });
    let best_sweep_f16 = simd_sweep
        .iter()
        .filter_map(|(_, _, _, _, _, tflops)| *tflops)
        .reduce(f64::max);
    let best_simd_gemm_f16 = simd_gemm
        .iter()
        .filter_map(|(_, tflops)| *tflops)
        .reduce(f64::max);
    let best_f16_mma = [
        f16_mma,
        f16_accum_f16_mma,
        best_sweep_f16,
        best_simd_gemm_f16,
        mpp_tensor_gemm_2048_f16,
        mpp_tensor_gemm_4096_f16,
        mps_gemm_2048_f16,
        mps_gemm_f16,
    ]
    .into_iter()
    .flatten()
    .reduce(f64::max);

    let mut microkernels = Vec::new();
    microkernels.push(MicroKernelMeasurement {
        name: "metal.unified_read".into(),
        dtype: None,
        shape: Some(format!("{bytes} bytes")),
        measured_gb_s: read,
        measured_tflops: None,
        iterations: 1,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.unified_write".into(),
        dtype: None,
        shape: Some(format!("{bytes} bytes")),
        measured_gb_s: write,
        measured_tflops: None,
        iterations: 1,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.unified_copy".into(),
        dtype: None,
        shape: Some(format!("{bytes} bytes")),
        measured_gb_s: copy,
        measured_tflops: None,
        iterations: 1,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.threadgroup_rw".into(),
        dtype: None,
        shape: Some(format!("{core_count} gpu cores")),
        measured_gb_s: threadgroup,
        measured_tflops: None,
        iterations: 4096,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.simdgroup_mma_f16_f32acc".into(),
        dtype: Some("f16*f16->f32".into()),
        shape: Some(format!(
            "8x8x8, {} threadgroups/core",
            mma_threadgroups_per_core
        )),
        measured_gb_s: None,
        measured_tflops: f16_mma,
        iterations: mma_iterations as u64,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.simdgroup_mma_f16_f16acc".into(),
        dtype: Some("f16*f16->f16".into()),
        shape: Some(format!(
            "8x8x8, {} threadgroups/core",
            mma_threadgroups_per_core
        )),
        measured_gb_s: None,
        measured_tflops: f16_accum_f16_mma,
        iterations: mma_iterations as u64,
    });
    for (label, tgpc, sgtg, acc, f16_accum, tflops) in simd_sweep {
        microkernels.push(MicroKernelMeasurement {
            name: format!("metal.simdgroup_mma_f16_sweep.{label}"),
            dtype: Some(if f16_accum {
                "f16*f16->f16".into()
            } else {
                "f16*f16->f32".into()
            }),
            shape: Some(format!(
                "8x8x8, threadgroups/core={tgpc}, simdgroups/threadgroup={sgtg}, accumulators={acc}"
            )),
            measured_gb_s: None,
            measured_tflops: tflops,
            iterations: sweep_iterations as u64,
        });
    }
    for (simdgroups_per_threadgroup, tflops) in simd_gemm {
        microkernels.push(MicroKernelMeasurement {
            name: format!("metal.simdgroup_gemm_f16_f32acc.sg{simdgroups_per_threadgroup}"),
            dtype: Some("f16*f16->f32".into()),
            shape: Some(format!(
                "{simd_gemm_size}x{simd_gemm_size}x{simd_gemm_size}, 8x8 output tile, simdgroups/threadgroup={simdgroups_per_threadgroup}"
            )),
            measured_gb_s: None,
            measured_tflops: tflops,
            iterations: simd_gemm_iterations as u64,
        });
    }
    microkernels.push(MicroKernelMeasurement {
        name: "metal.mps_gemm_f16_2048".into(),
        dtype: Some("f16".into()),
        shape: Some(format!(
            "{simd_gemm_size}x{simd_gemm_size}x{simd_gemm_size}"
        )),
        measured_gb_s: None,
        measured_tflops: mps_gemm_2048_f16,
        iterations: simd_gemm_iterations as u64,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.mpp_tensor_gemm_f16_2048".into(),
        dtype: Some("f16*f16->f32".into()),
        shape: Some(format!(
            "{simd_gemm_size}x{simd_gemm_size}x{simd_gemm_size} equivalent, repeated 64x32x64 MPP tiles"
        )),
        measured_gb_s: None,
        measured_tflops: mpp_tensor_gemm_2048_f16,
        iterations: simd_gemm_iterations as u64,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.mpp_tensor_gemm_f16_4096".into(),
        dtype: Some("f16*f16->f32".into()),
        shape: Some(format!(
            "{mps_gemm_size}x{mps_gemm_size}x{mps_gemm_size} equivalent, repeated 64x32x64 MPP tiles"
        )),
        measured_gb_s: None,
        measured_tflops: mpp_tensor_gemm_4096_f16,
        iterations: 1,
    });
    microkernels.push(MicroKernelMeasurement {
        name: "metal.mps_gemm_f16".into(),
        dtype: Some("f16".into()),
        shape: Some(format!("{mps_gemm_size}x{mps_gemm_size}x{mps_gemm_size}")),
        measured_gb_s: None,
        measured_tflops: mps_gemm_f16,
        iterations: mps_gemm_iterations as u64,
    });
    for (name, in_dim, out_dim, iters) in [
        ("qwen36.int4_gemv.hidden_to_experts", 2048, 256 * 1024, 1),
        ("qwen36.int4_gemv.expert_down_topk8", 512, 2048 * 8, 16),
        ("qwen36.int4_gemv.lm_head", 2048, 248_320, 1),
    ] {
        let gb_s = positive(unsafe { ffi::mp_metal_int4_gemv_gb_s(in_dim, out_dim, 128, iters) });
        microkernels.push(MicroKernelMeasurement {
            name: name.into(),
            dtype: Some("int4->bf16".into()),
            shape: Some(format!("in={in_dim}, out={out_dim}, group=128")),
            measured_gb_s: gb_s,
            measured_tflops: None,
            iterations: iters as u64,
        });
    }

    Ok(vec![GpuProfile {
        backend: "Metal".into(),
        device_index: 0,
        arch_name: info.arch_name.clone(),
        pci_id: None,
        uuid: None,
        memory_arch: "Unified".into(),
        total_vram_bytes: info.total_vram_bytes,
        cu_count: core_count,
        wave_size: info.wave_size.max(32),
        lds_per_cu_bytes: info.max_threadgroup_memory_bytes,
        lds_bw_per_cu_gb_s: threadgroup.map(|x| x / core_count as f64),
        lds_bw_aggregate_gb_s: threadgroup,
        vram_bw: VramBandwidth {
            read_gb_s: read,
            write_gb_s: write,
            copy_gb_s: copy,
            theoretical_peak_gb_s: None,
            ratio_read: None,
        },
        mma_peak: MmaPeak {
            f16: best_f16_mma.map(|measured_tflops| MmaMeasurement {
                measured_tflops,
                theoretical_tflops: None,
                ratio: None,
            }),
            ..MmaPeak::default()
        },
        pcie: PcieProfile::default(),
        clock_rate_khz_measured: None,
        metal: Some(MetalProfile {
            device_name: Some(info.device_name),
            metal_family: Some(info.metal_family),
            gpu_core_count_source: Some("system_profiler SPDisplaysDataType".into()),
            max_threadgroup_memory_bytes: Some(info.max_threadgroup_memory_bytes),
            max_threads_per_threadgroup: Some(info.max_threads_per_threadgroup),
            recommended_working_set_bytes: Some(info.recommended_working_set_bytes),
            simdgroup_matrix_supported: Some(simd_probe),
            mpp_tensor_matmul_supported: Some(mpp_probe),
            mpp_tensor_matmul_probe_status: Some(c_buf_to_string(&mpp_probe_status)),
            mpp_tensor_matmul_probe_code: Some(mpp_probe_info.status),
            mpp_tensor_write_probe_value: finite_f32(mpp_probe_info.tensor_write_value),
            mpp_tensor_matmul_probe_value: finite_f32(mpp_probe_info.matmul_value),
        }),
        microkernels,
    }])
}

#[cfg(not(supersonic_backend_metal))]
fn profile() -> Result<Vec<GpuProfile>, GpuProfileError> {
    Err(GpuProfileError::NotImplemented("Metal"))
}

#[cfg(supersonic_backend_metal)]
fn positive(v: f64) -> Option<f64> {
    (v.is_finite() && v > 0.0).then_some(v)
}

#[cfg(supersonic_backend_metal)]
fn finite_f32(v: f32) -> Option<f32> {
    v.is_finite().then_some(v)
}
