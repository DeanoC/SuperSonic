use anyhow::{bail, Context, Result};
use clap::Parser;
use gpu_hal::{
    copy_h2d_async, hal_profile_reset, hal_profile_set_enabled, hal_profile_snapshot, sync,
    Backend, GpuBuffer, GpuStream, PinnedHostBuffer, ScalarType,
};
use model_store::{BakedStore, FlmLoadOptions};
use serde::Serialize;
use std::{ffi::c_void, path::PathBuf, time::Instant};

const MIB: f64 = 1024.0 * 1024.0;

#[derive(Debug, Parser)]
#[command(about = "Probe FLM/BakedStore H2D upload modes for selected tensors")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "hip")]
    backend: String,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, value_delimiter = ',', required = true)]
    tensor: Vec<String>,
    #[arg(long, default_value_t = 1)]
    iters: usize,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct UploadProbeRecord {
    tensor: String,
    mode: &'static str,
    dtype: String,
    bytes: usize,
    iters: usize,
    total_ms: f64,
    mib_per_s: f64,
    host_stage_ms: f64,
    host_pinned_alloc_ms: f64,
    device_wait_ms: f64,
    hal_total_ms: f64,
    copy_h2d_ms: f64,
    copy_h2d_async_ms: f64,
    copy_h2d_bytes: u64,
    copy_h2d_async_bytes: u64,
    alloc_ms: f64,
    alloc_bytes: u64,
}

fn parse_backend_arg(value: &str) -> Result<Backend> {
    Backend::parse(value).ok_or_else(|| anyhow::anyhow!("unknown backend: {value}"))
}

fn upload_shape_for_bytes(dtype: ScalarType, byte_len: usize) -> Result<Vec<usize>> {
    let elem_size = dtype.size_in_bytes();
    if byte_len % elem_size != 0 {
        bail!("byte_len={byte_len} is not divisible by dtype element size {elem_size}");
    }
    Ok(vec![byte_len / elem_size])
}

fn mib_per_second(bytes: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / MIB) / (elapsed_ms / 1000.0)
}

fn entry_total_ms(snapshot: &gpu_hal::HalProfileSnapshot, op: &str) -> f64 {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.op == op)
        .map(|entry| entry.total_ms)
        .unwrap_or(0.0)
}

fn entry_total_bytes(snapshot: &gpu_hal::HalProfileSnapshot, op: &str) -> u64 {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.op == op)
        .map(|entry| entry.total_bytes)
        .unwrap_or(0)
}

fn dtype_for_tensor(store: &BakedStore, tensor: &str) -> Result<ScalarType> {
    let meta = store
        .meta(tensor)
        .with_context(|| format!("tensor not found: {tensor}"))?;
    ScalarType::from_name(&meta.dtype).with_context(|| {
        format!(
            "tensor {tensor} has unsupported dtype {} for upload probe",
            meta.dtype
        )
    })
}

fn run_pageable_upload(
    store: &BakedStore,
    tensor: &str,
    device: usize,
    iters: usize,
) -> Result<UploadProbeRecord> {
    let meta = store
        .meta(tensor)
        .with_context(|| format!("tensor not found: {tensor}"))?;
    let bytes = usize::try_from(meta.byte_len)
        .with_context(|| format!("tensor {tensor} byte_len does not fit usize"))?;
    hal_profile_set_enabled(true);
    hal_profile_reset();
    let start = Instant::now();
    for _ in 0..iters {
        let buffer = store
            .load_to_gpu(tensor, device)
            .with_context(|| format!("pageable upload tensor {tensor}"))?;
        sync(device).context("sync after pageable upload")?;
        drop(buffer);
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let snapshot = hal_profile_snapshot();
    hal_profile_set_enabled(false);
    Ok(record_from_snapshot(
        tensor,
        "pageable",
        meta.dtype.clone(),
        bytes,
        iters,
        total_ms,
        0.0,
        0.0,
        0.0,
        &snapshot,
    ))
}

fn run_pinned_upload(
    store: &BakedStore,
    tensor: &str,
    device: usize,
    iters: usize,
) -> Result<UploadProbeRecord> {
    let meta = store
        .meta(tensor)
        .with_context(|| format!("tensor not found: {tensor}"))?;
    let bytes = store
        .raw_bytes(tensor)
        .with_context(|| format!("tensor {tensor} has no raw byte payload"))?;
    let dtype = dtype_for_tensor(store, tensor)?;
    let shape = upload_shape_for_bytes(dtype, bytes.len())?;
    let stream = GpuStream::new_nonblocking(device).context("create pinned upload stream")?;
    let mut host_stage_ms = 0.0;
    let mut host_pinned_alloc_ms = 0.0;
    let mut device_wait_ms = 0.0;

    hal_profile_set_enabled(true);
    hal_profile_reset();
    let start = Instant::now();
    let alloc_start = Instant::now();
    let mut staging =
        PinnedHostBuffer::new(device, bytes.len()).context("allocate pinned staging")?;
    host_pinned_alloc_ms += alloc_start.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..iters {
        let stage_start = Instant::now();
        staging.as_mut_slice().copy_from_slice(bytes);
        host_stage_ms += stage_start.elapsed().as_secs_f64() * 1000.0;
        let mut buffer =
            GpuBuffer::alloc(device, dtype, &shape).context("allocate probe device buffer")?;
        let wait_start = Instant::now();
        copy_h2d_async(
            device,
            &stream,
            buffer.as_mut_ptr(),
            staging.as_ptr() as *const c_void,
            bytes.len(),
        )
        .context("async pinned H2D upload")?;
        stream.synchronize().context("sync pinned upload stream")?;
        device_wait_ms += wait_start.elapsed().as_secs_f64() * 1000.0;
        drop(buffer);
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let snapshot = hal_profile_snapshot();
    hal_profile_set_enabled(false);
    Ok(record_from_snapshot(
        tensor,
        "pinned",
        meta.dtype.clone(),
        bytes.len(),
        iters,
        total_ms,
        host_stage_ms,
        host_pinned_alloc_ms,
        device_wait_ms,
        &snapshot,
    ))
}

fn record_from_snapshot(
    tensor: &str,
    mode: &'static str,
    dtype: String,
    bytes: usize,
    iters: usize,
    total_ms: f64,
    host_stage_ms: f64,
    host_pinned_alloc_ms: f64,
    device_wait_ms: f64,
    snapshot: &gpu_hal::HalProfileSnapshot,
) -> UploadProbeRecord {
    UploadProbeRecord {
        tensor: tensor.to_string(),
        mode,
        dtype,
        bytes,
        iters,
        total_ms,
        mib_per_s: mib_per_second(bytes * iters, total_ms),
        host_stage_ms,
        host_pinned_alloc_ms,
        device_wait_ms,
        hal_total_ms: snapshot.total_ms,
        copy_h2d_ms: entry_total_ms(snapshot, "copy_h2d"),
        copy_h2d_async_ms: entry_total_ms(snapshot, "copy_h2d_async"),
        copy_h2d_bytes: entry_total_bytes(snapshot, "copy_h2d"),
        copy_h2d_async_bytes: entry_total_bytes(snapshot, "copy_h2d_async"),
        alloc_ms: entry_total_ms(snapshot, "alloc"),
        alloc_bytes: entry_total_bytes(snapshot, "alloc"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.iters == 0 {
        bail!("--iters must be > 0");
    }
    let backend = parse_backend_arg(&args.backend)?;
    if !gpu_hal::is_backend_compiled(backend) {
        bail!("backend {backend} is not compiled into this build");
    }
    gpu_hal::set_backend(backend);
    gpu_hal::set_device(args.device)
        .with_context(|| format!("set {backend} device {}", args.device))?;
    let store = if args.model_dir.extension().and_then(|ext| ext.to_str()) == Some("flm") {
        BakedStore::open_flm_with_options(
            &args.model_dir,
            FlmLoadOptions {
                verify_block_hashes: false,
                flm_int4_logical_aliases: true,
            },
        )
    } else {
        BakedStore::open(&args.model_dir)
    }
    .with_context(|| format!("open store {}", args.model_dir.display()))?;

    let mut records = Vec::new();
    for tensor in &args.tensor {
        records.push(run_pageable_upload(
            &store,
            tensor,
            args.device,
            args.iters,
        )?);
        records.push(run_pinned_upload(&store, tensor, args.device, args.iters)?);
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&records)?);
    } else {
        for record in &records {
            println!(
                "[upload-probe] tensor={} mode={} dtype={} bytes={} iters={} total_ms={:.3} mib_s={:.1} host_stage_ms={:.3} host_pinned_alloc_ms={:.3} device_wait_ms={:.3} copy_h2d_ms={:.3} copy_h2d_async_ms={:.3} alloc_ms={:.3}",
                record.tensor,
                record.mode,
                record.dtype,
                record.bytes,
                record.iters,
                record.total_ms,
                record.mib_per_s,
                record.host_stage_ms,
                record.host_pinned_alloc_ms,
                record.device_wait_ms,
                record.copy_h2d_ms,
                record.copy_h2d_async_ms,
                record.alloc_ms,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_names_case_insensitively() {
        assert_eq!(parse_backend_arg("HIP").unwrap(), Backend::Hip);
        assert_eq!(parse_backend_arg("cuda").unwrap(), Backend::Cuda);
        assert!(parse_backend_arg("vulkan").is_err());
    }

    #[test]
    fn builds_linear_upload_shape_from_byte_len() {
        assert_eq!(
            upload_shape_for_bytes(ScalarType::BF16, 8).unwrap(),
            vec![4]
        );
        assert_eq!(upload_shape_for_bytes(ScalarType::U8, 5).unwrap(), vec![5]);
        assert!(upload_shape_for_bytes(ScalarType::BF16, 7).is_err());
    }

    #[test]
    fn reports_mib_per_second() {
        assert_eq!(mib_per_second(128 * 1024 * 1024, 64.0), 2000.0);
        assert_eq!(mib_per_second(128 * 1024 * 1024, 0.0), 0.0);
    }

    #[test]
    fn records_device_wait_wall_time() {
        let snapshot = gpu_hal::HalProfileSnapshot::default();
        let record = record_from_snapshot(
            "tensor",
            "pinned",
            "u8".to_string(),
            1024,
            1,
            10.0,
            2.0,
            1.5,
            7.5,
            &snapshot,
        );

        assert_eq!(record.host_pinned_alloc_ms, 1.5);
        assert_eq!(record.device_wait_ms, 7.5);
    }
}
