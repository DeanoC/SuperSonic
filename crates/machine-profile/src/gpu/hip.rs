use crate::gpu::hip_ffi::*;
use crate::gpu::{GpuProfileError, GpuProfiler};
use crate::schema::*;
use gpu_hal::{set_device, Backend};

pub struct HipProfiler;

impl GpuProfiler for HipProfiler {
    fn profile(&self) -> Result<Vec<GpuProfile>, GpuProfileError> {
        if !gpu_hal::is_backend_compiled(Backend::Hip) {
            return Err(GpuProfileError::NotImplemented("HIP not compiled"));
        }
        gpu_hal::set_backend(Backend::Hip);
        let mut out = Vec::new();
        // gpu-hal currently exposes a single device. If multi-device support
        // lands, iterate here.
        for device_index in 0..1u32 {
            set_device(device_index as usize)
                .map_err(|e| GpuProfileError::Hip(e.to_string()))?;
            let info = query_device_info_hip(device_index as i32)
                .map_err(|e| GpuProfileError::Hip(e))?;
            out.push(profile_one(device_index, &info));
        }
        Ok(out)
    }
}

pub(crate) struct HipDeviceInfo {
    pub arch_name: String,
    pub total_vram_bytes: u64,
    pub warp_size: u32,
    pub clock_rate_khz: u32,
    pub pci_device_id: u32,
}

pub(crate) fn query_device_info_hip(device: i32) -> Result<HipDeviceInfo, String> {
    let mut arch_buf = [0u8; 64];
    let mut total_vram: u64 = 0;
    let mut warp_size: u32 = 0;
    let mut clock_khz: u32 = 0;
    let mut pci_device_id: u32 = 0;
    let status = unsafe {
        mp_query_device_info(
            device,
            arch_buf.as_mut_ptr(),
            arch_buf.len() as u32,
            &mut total_vram,
            &mut warp_size,
            &mut clock_khz,
            &mut pci_device_id,
        )
    };
    if status != 0 {
        return Err(format!("mp_query_device_info returned {status}"));
    }
    let nul = arch_buf.iter().position(|&b| b == 0).unwrap_or(arch_buf.len());
    let arch_name = String::from_utf8_lossy(&arch_buf[..nul]).to_string();
    Ok(HipDeviceInfo {
        arch_name,
        total_vram_bytes: total_vram,
        warp_size,
        clock_rate_khz: clock_khz,
        pci_device_id,
    })
}

/// Enumerate HIP devices and return device info for fingerprint-only purposes.
/// Does not launch any kernels — safe to call at startup without a full profile.
pub(crate) fn enumerate_for_fingerprint() -> Result<Vec<HipDeviceInfo>, super::GpuProfileError> {
    if !gpu_hal::is_backend_compiled(gpu_hal::Backend::Hip) {
        return Err(super::GpuProfileError::NotImplemented("HIP not compiled"));
    }
    gpu_hal::set_backend(gpu_hal::Backend::Hip);
    let mut out = Vec::new();
    for device_index in 0..1u32 {
        gpu_hal::set_device(device_index as usize)
            .map_err(|e| super::GpuProfileError::Hip(e.to_string()))?;
        let info = query_device_info_hip(device_index as i32)
            .map_err(|e| super::GpuProfileError::Hip(e))?;
        out.push(info);
    }
    Ok(out)
}

fn profile_one(device_index: u32, info: &HipDeviceInfo) -> GpuProfile {
    let cu_count = guess_cu_count(&info.arch_name);
    let lds_aggregate =
        unsafe { mp_lds_bandwidth_run(device_index as i32, cu_count, 1_000_000) };
    let read = unsafe { mp_hbm_bandwidth_read(device_index as i32, 1u64 << 28) };
    let write = unsafe { mp_hbm_bandwidth_write(device_index as i32, 1u64 << 28) };
    let copy = unsafe { mp_hbm_bandwidth_copy(device_index as i32, 1u64 << 28) };

    // WMMA microkernels only have real intrinsics for some archs. Probe the
    // device — the kernel itself uses __has_builtin per dtype, so future
    // arches that ship the builtin are auto-detected without code changes.
    // On unsupported (or unprobeable) archs we leave MMA fields as None
    // rather than reporting fake numbers from launch-overhead-only runs.
    let mut wmma_flags = [0i32; 3];
    let probe_status =
        unsafe { mp_wmma_probe(device_index as i32, wmma_flags.as_mut_ptr()) };
    let probe_ok = probe_status == 0;
    let f16 = if probe_ok && wmma_flags[0] == 1 {
        Some(unsafe { mp_wmma_peak_f16(device_index as i32, cu_count, 100_000) })
    } else {
        None
    };
    let bf16 = if probe_ok && wmma_flags[1] == 1 {
        Some(unsafe { mp_wmma_peak_bf16(device_index as i32, cu_count, 100_000) })
    } else {
        None
    };
    let i8_tops = if probe_ok && wmma_flags[2] == 1 {
        Some(unsafe { mp_wmma_peak_i8(device_index as i32, cu_count, 100_000) })
    } else {
        None
    };

    let mut h2d = vec![MpTransferSample { bytes: 0, gb_s: 0.0 }; 16];
    let n_h2d =
        unsafe { mp_pcie_h2d(device_index as i32, h2d.as_mut_ptr(), h2d.len() as i32) };
    h2d.truncate(n_h2d.max(0) as usize);

    let mut d2h = vec![MpTransferSample { bytes: 0, gb_s: 0.0 }; 16];
    let n_d2h =
        unsafe { mp_pcie_d2h(device_index as i32, d2h.as_mut_ptr(), d2h.len() as i32) };
    d2h.truncate(n_d2h.max(0) as usize);

    let duplex = unsafe { mp_pcie_duplex(device_index as i32, 1u64 << 27) };

    GpuProfile {
        backend: "HIP".into(),
        device_index,
        arch_name: info.arch_name.clone(),
        pci_id: Some(format!("0x{:04x}", info.pci_device_id)),
        uuid: None,
        memory_arch: format!("{:?}", gpu_hal::current_memory_architecture()),
        total_vram_bytes: info.total_vram_bytes,
        cu_count,
        wave_size: info.warp_size,
        lds_per_cu_bytes: 65536,
        lds_bw_per_cu_gb_s: Some(lds_aggregate / cu_count as f64),
        lds_bw_aggregate_gb_s: Some(lds_aggregate),
        vram_bw: VramBandwidth {
            read_gb_s: Some(read),
            write_gb_s: Some(write),
            copy_gb_s: Some(copy),
            theoretical_peak_gb_s: None,
            ratio_read: None,
        },
        mma_peak: MmaPeak {
            f16: f16.map(|measured_tflops| MmaMeasurement {
                measured_tflops,
                theoretical_tflops: None,
                ratio: None,
            }),
            bf16: bf16.map(|measured_tflops| MmaMeasurement {
                measured_tflops,
                theoretical_tflops: None,
                ratio: None,
            }),
            fp8_e4m3: None,
            i8: i8_tops.map(|measured_tflops| MmaMeasurement {
                measured_tflops,
                theoretical_tflops: None,
                ratio: None,
            }),
        },
        pcie: PcieProfile {
            generation: None,
            h2d_gb_s_by_size: h2d
                .into_iter()
                .map(|s| TransferSample {
                    bytes: s.bytes,
                    gb_s: s.gb_s,
                })
                .collect(),
            d2h_gb_s_by_size: d2h
                .into_iter()
                .map(|s| TransferSample {
                    bytes: s.bytes,
                    gb_s: s.gb_s,
                })
                .collect(),
            duplex_gb_s: Some(duplex),
        },
        clock_rate_khz_measured: Some(info.clock_rate_khz),
    }
}

fn guess_cu_count(arch: &str) -> u32 {
    match arch {
        "gfx1100" => 48,
        "gfx1101" => 32,
        "gfx1102" => 16,
        "gfx1150" => 16,
        "gfx90a" => 104,
        _ => 32,
    }
}

