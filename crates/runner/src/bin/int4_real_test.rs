//! Real-weight INT4 matmul test.
//! Reads hidden + reference + packed INT4 / scale / zero from /tmp/int4_diag/,
//! runs `matmul_rhs_transposed_int4`, compares against the Python bf16 reference.

use anyhow::{anyhow, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::prefill_ffi;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Meta {
    m: usize,
    n: usize,
    k: usize,
    k_packed: usize,
    group_size: usize,
    scale_rows: usize,
    scale_cols: usize,
    #[serde(rename = "tensor")]
    _tensor: String,
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn main() -> Result<()> {
    let ordinal = 0usize;
    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    let root = PathBuf::from(std::env::var("INT4_DIAG_DIR").unwrap_or_else(|_| "/tmp/int4_diag".into()));
    let meta: Meta = serde_json::from_str(&fs::read_to_string(root.join("meta.json"))?)?;
    println!("[real-test] meta = {meta:?}");

    let hidden_bytes = fs::read(root.join("hidden_bf16.bin"))?;
    let ref_bytes = fs::read(root.join("ref_bf16.bin"))?;
    let packed_bytes = fs::read(root.join("packed_u8.bin"))?;
    let scale_bytes = fs::read(root.join("scale_bf16.bin"))?;
    let zero_bytes = fs::read(root.join("zero_bf16.bin"))?;

    assert_eq!(hidden_bytes.len(), meta.m * meta.k * 2);
    assert_eq!(ref_bytes.len(), meta.m * meta.n * 2);
    assert_eq!(packed_bytes.len(), meta.n * meta.k_packed);
    assert_eq!(scale_bytes.len(), meta.scale_rows * meta.scale_cols * 2);
    assert_eq!(zero_bytes.len(), meta.scale_rows * meta.scale_cols * 2);

    let hidden_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[meta.m, meta.k], &hidden_bytes)
        .map_err(|e| anyhow!("hidden upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::U8, &[meta.n, meta.k_packed], &packed_bytes)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;
    let scale_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[meta.scale_rows, meta.scale_cols], &scale_bytes)
        .map_err(|e| anyhow!("scale upload: {e}"))?;
    let zero_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[meta.scale_rows, meta.scale_cols], &zero_bytes)
        .map_err(|e| anyhow!("zero upload: {e}"))?;

    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[meta.m, meta.n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;

    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal, 1, meta.m, meta.n, meta.k,
        &hidden_gpu, &rhs_gpu, &scale_gpu, &zero_gpu,
        meta.group_size, &mut out_gpu,
    ).map_err(|e| anyhow!("matmul int4 call: {e}"))?;

    let out_host = bf16_bytes_to_f32(&out_gpu.to_host_bytes()
        .map_err(|e| anyhow!("d2h: {e}"))?);
    let ref_host = bf16_bytes_to_f32(&ref_bytes);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut nbad = 0usize;
    let mut first_bad: Option<(usize, usize, f32, f32, f32)> = None;
    let nref = ref_host.len();
    let mut sum_abs_g: f64 = 0.0;
    let mut sum_abs_r: f64 = 0.0;
    let mut sum_sq_g: f64 = 0.0;
    let mut sum_sq_r: f64 = 0.0;
    for i in 0..nref {
        let g = out_host[i];
        let r = ref_host[i];
        sum_abs_g += g.abs() as f64;
        sum_abs_r += r.abs() as f64;
        sum_sq_g += (g as f64) * (g as f64);
        sum_sq_r += (r as f64) * (r as f64);
        let ad = (g - r).abs();
        let rel = ad / r.abs().max(1e-6);
        if ad > max_abs { max_abs = ad; }
        if rel > max_rel { max_rel = rel; }
        if ad > 0.2 && rel > 0.05 {
            if first_bad.is_none() {
                let mi = i / meta.n;
                let ni = i % meta.n;
                first_bad = Some((mi, ni, g, r, ad));
            }
            nbad += 1;
        }
    }
    let rmse = ((sum_sq_g + sum_sq_r - 2.0 * {
        let mut s = 0f64;
        for i in 0..nref { s += out_host[i] as f64 * ref_host[i] as f64; }
        s
    }) / (nref as f64)).sqrt();

    println!("[real-test] n_elems = {nref}");
    println!("[real-test] max_abs = {max_abs:.5e}  max_rel = {max_rel:.5e}  bad = {nbad}");
    println!("[real-test] avg |gpu| = {:.4}, avg |ref| = {:.4}",
             sum_abs_g / nref as f64, sum_abs_r / nref as f64);
    println!("[real-test] rms(ref) = {:.4},  rms(gpu) = {:.4},  rmse = {:.6}",
             (sum_sq_r / nref as f64).sqrt(), (sum_sq_g / nref as f64).sqrt(), rmse);
    if let Some((mi, ni, g, r, ad)) = first_bad {
        println!("[real-test] first bad @ [{mi},{ni}]: gpu={g:.4}  ref={r:.4}  ad={ad:.4}");
    }
    println!("[real-test] samples: gpu[0,0]={:.4}  ref[0,0]={:.4}",
             out_host[0], ref_host[0]);

    if nbad == 0 {
        println!("[real-test] OK");
    } else {
        println!("[real-test] FAIL");
    }
    Ok(())
}
