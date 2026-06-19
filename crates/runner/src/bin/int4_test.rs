//! Minimal known-value test for `matmul_rhs_transposed_int4`.
//!
//! Compares the HIP INT4 dequant matmul against a CPU reference using
//! bit-exact dequant: `bf16(q*s - zf*s)`.

use anyhow::{anyhow, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use half::{bf16, f16};
use kernel_ffi::prefill_ffi;
use std::time::Instant;

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let b = bf16::from_f32(v).to_bits();
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// Bit-exact BF16 dequant that mirrors the baker: `bf16(q*s - zf*s)`.
fn dequant_bf16(nibble: u8, scale: f32, zero: f32) -> f32 {
    let q = nibble as f32;
    let dq = q * scale - zero * scale;
    bf16::from_f32(dq).to_f32()
}

/// Pack 2 nibbles per byte, low=even col first (matches baker).
fn pack_nibbles(nibbles: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 2, 0);
    let mut out = Vec::with_capacity(rows * cols / 2);
    for r in 0..rows {
        for c in (0..cols).step_by(2) {
            let lo = nibbles[r * cols + c] & 0xF;
            let hi = nibbles[r * cols + c + 1] & 0xF;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// CPU reference: out[m,n] = sum_k lhs[m,k] * dequant(nibble[n,k], scale[n/gs,k/gs], zero[n/gs,k/gs]).
/// `lhs` and scale/zero values are taken pre-bf16 rounding (caller should pass bf16-rounded floats
/// if matching the kernel's effective inputs).
fn reference_matmul(
    m: usize,
    n: usize,
    k: usize,
    gs: usize,
    lhs: &[f32],    // [m, k] BF16-rounded values
    nibbles: &[u8], // [n, k]
    scales: &[f32], // [n/gs, k/gs] BF16-rounded values
    zeros: &[f32],  // [n/gs, k/gs] BF16-rounded values
) -> Vec<f32> {
    let scale_cols = (k + gs - 1) / gs;
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                let sr = ni / gs;
                let sc = ki / gs;
                let si = sr * scale_cols + sc;
                let s = scales[si];
                let z = zeros[si];
                let w = dequant_bf16(nibbles[ni * k + ki], s, z);
                acc += lhs[mi * k + ki] * w;
            }
            out[mi * n + ni] = bf16::from_f32(acc).to_f32();
        }
    }
    out
}

/// Round an f32 to bf16 and back — mimics what the baker does when serialising scale/zero.
fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn push_f16_le(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

fn ggml_q4k_row(row: usize) -> (Vec<u8>, Vec<f32>) {
    let mut out = Vec::with_capacity(144);
    push_f16_le(&mut out, 1.0);
    push_f16_le(&mut out, 0.0);
    out.extend_from_slice(&[1u8; 12]);
    let mut vals = vec![0f32; 256];
    for g in 0..4 {
        for l in 0..32 {
            let lo = ((row * 7 + g * 3 + l) & 0x0f) as u8;
            let hi = ((row * 11 + g * 5 + l + 1) & 0x0f) as u8;
            out.push(lo | (hi << 4));
            vals[g * 64 + l] = lo as f32;
            vals[g * 64 + 32 + l] = hi as f32;
        }
    }
    (out, vals)
}

fn ggml_q5k_row(row: usize) -> (Vec<u8>, Vec<f32>) {
    let mut out = Vec::with_capacity(176);
    push_f16_le(&mut out, 1.0);
    push_f16_le(&mut out, 0.0);
    out.extend_from_slice(&[1u8; 12]);
    let qh_start = out.len();
    out.extend_from_slice(&[0u8; 32]);
    let mut qs = Vec::with_capacity(128);
    let mut vals = vec![0f32; 256];
    for g in 0..4 {
        for l in 0..32 {
            let v0 = ((row * 13 + g * 17 + l) & 0x1f) as u8;
            let v1 = ((row * 19 + g * 23 + l + 3) & 0x1f) as u8;
            qs.push((v0 & 0x0f) | ((v1 & 0x0f) << 4));
            if (v0 & 0x10) != 0 {
                out[qh_start + l] |= 1 << (2 * g);
            }
            if (v1 & 0x10) != 0 {
                out[qh_start + l] |= 2 << (2 * g);
            }
            vals[g * 64 + l] = v0 as f32;
            vals[g * 64 + 32 + l] = v1 as f32;
        }
    }
    out.extend_from_slice(&qs);
    (out, vals)
}

fn ggml_q6k_row(row: usize) -> (Vec<u8>, Vec<f32>) {
    let mut ql = vec![0u8; 128];
    let mut qh = vec![0u8; 64];
    let scales = [1i8; 16];
    let mut vals = vec![0f32; 256];
    for half in 0..2 {
        for l in 0..32 {
            let base = half * 128;
            let vs = [
                ((row * 5 + half * 7 + l) & 0x3f) as u8,
                ((row * 11 + half * 13 + l + 1) & 0x3f) as u8,
                ((row * 17 + half * 19 + l + 2) & 0x3f) as u8,
                ((row * 23 + half * 29 + l + 3) & 0x3f) as u8,
            ];
            ql[half * 64 + l] = (vs[0] & 0x0f) | ((vs[2] & 0x0f) << 4);
            ql[half * 64 + 32 + l] = (vs[1] & 0x0f) | ((vs[3] & 0x0f) << 4);
            qh[half * 32 + l] = ((vs[0] >> 4) & 3)
                | (((vs[1] >> 4) & 3) << 2)
                | (((vs[2] >> 4) & 3) << 4)
                | (((vs[3] >> 4) & 3) << 6);
            vals[base + l] = vs[0] as f32 - 32.0;
            vals[base + 32 + l] = vs[1] as f32 - 32.0;
            vals[base + 64 + l] = vs[2] as f32 - 32.0;
            vals[base + 96 + l] = vs[3] as f32 - 32.0;
        }
    }
    let mut out = Vec::with_capacity(210);
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(scales.iter().map(|v| *v as u8));
    push_f16_le(&mut out, 1.0);
    (out, vals)
}

fn ggml_q8_0_row(row: usize) -> (Vec<u8>, Vec<f32>) {
    let d = f16::from_f32(0.03125 + (row % 7) as f32 * 0.001953125).to_f32();
    let mut out = Vec::with_capacity(34);
    push_f16_le(&mut out, d);
    let mut vals = vec![0f32; 32];
    for (l, val) in vals.iter_mut().enumerate() {
        let q = (((row * 17 + l * 11 + 13) % 255) as i16 - 127) as i8;
        out.push(q as u8);
        *val = bf16_round(d * q as f32);
    }
    (out, vals)
}

fn run_ggml_case(
    ordinal: usize,
    name: &str,
    qtype: i32,
    row_fn: fn(usize) -> (Vec<u8>, Vec<f32>),
) -> Result<()> {
    run_ggml_case_shape(ordinal, name, qtype, row_fn, 3, 17, 256)
}

fn run_ggml_case_shape(
    ordinal: usize,
    name: &str,
    qtype: i32,
    row_fn: fn(usize) -> (Vec<u8>, Vec<f32>),
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    println!("=== {name}: m={m} n={n} k={k} ===");
    let mut lhs = vec![0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            lhs[mi * k + ki] = bf16_round((((mi + 1) as f32) * 0.01 + (ki as f32) * 0.003).sin());
        }
    }
    let mut rhs = Vec::new();
    let mut rows = Vec::new();
    for ni in 0..n {
        let mut vals = Vec::with_capacity(k);
        for block in 0..ggml_blocks_for(qtype, k)? {
            let (bytes, block_vals) = row_fn(ni.wrapping_mul(131).wrapping_add(block));
            rhs.extend_from_slice(&bytes);
            vals.extend_from_slice(&block_vals);
        }
        rows.push(vals);
    }
    let row_bytes = rhs.len() / n;
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, row_bytes], &rhs)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;
    let dummy_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, 1],
        &f32_to_bf16_bytes(&[0.0]),
    )
    .map_err(|e| anyhow!("dummy upload: {e}"))?;
    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;
    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        &rhs_gpu,
        &dummy_gpu,
        &dummy_gpu,
        None,
        128,
        qtype,
        &mut out_gpu,
    )
    .map_err(|e| anyhow!("ggml matmul: {e}"))?;
    let out_host = bf16_bytes_to_f32(
        &out_gpu
            .to_host_bytes()
            .map_err(|e| anyhow!("out d2h: {e}"))?,
    );
    let mut nbad = 0usize;
    let mut max_abs = 0f32;
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += lhs[mi * k + ki] * rows[ni][ki];
            }
            let r = bf16_round(acc);
            let g = out_host[mi * n + ni];
            let d = (g - r).abs();
            max_abs = max_abs.max(d);
            if d > 0.25 {
                nbad += 1;
            }
        }
    }
    println!("  max_abs={max_abs:.5e} bad={nbad}/{}", m * n);
    if nbad > 0 {
        return Err(anyhow!("{name} mismatches"));
    }
    Ok(())
}

fn run_ggml_residual_add_case(
    ordinal: usize,
    name: &str,
    qtype: i32,
    row_fn: fn(usize) -> (Vec<u8>, Vec<f32>),
) -> Result<()> {
    let m = 16usize;
    let n = 64usize;
    let k = 512usize;
    println!("=== {name} residual add: m={m} n={n} k={k} ===");
    let lhs = make_bench_lhs(m, k);
    let rhs = make_ggml_k_slab(n, k, qtype, row_fn)?;
    let row_bytes = ggml_row_bytes_for(qtype, k)?;
    let mut residual = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let x = (mi as f32 + 1.0) * 0.013 + (ni as f32) * 0.007;
            residual[mi * n + ni] = bf16_round(x.cos() * 0.5);
        }
    }

    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, row_bytes], &rhs)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;
    let dummy_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, 1],
        &f32_to_bf16_bytes(&[0.0]),
    )
    .map_err(|e| anyhow!("dummy upload: {e}"))?;
    let residual_bytes = f32_to_bf16_bytes(&residual);
    let residual_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, n], &residual_bytes)
            .map_err(|e| anyhow!("residual upload: {e}"))?;
    let mut proj_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("proj alloc: {e}"))?;
    let mut ref_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("ref alloc: {e}"))?;
    let mut fused_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, n], &residual_bytes)
            .map_err(|e| anyhow!("fused residual upload: {e}"))?;

    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        &rhs_gpu,
        &dummy_gpu,
        &dummy_gpu,
        None,
        128,
        qtype,
        &mut proj_gpu,
    )
    .map_err(|e| anyhow!("regular ggml matmul: {e}"))?;
    prefill_ffi::element_add(
        ordinal,
        ScalarType::BF16,
        m * n,
        &residual_gpu,
        &proj_gpu,
        &mut ref_gpu,
    )
    .map_err(|e| anyhow!("reference residual add: {e}"))?;

    let fused_residual_ref: &GpuBuffer = unsafe { &*(&fused_gpu as *const GpuBuffer) };
    let handled = prefill_ffi::matmul_rhs_transposed_int4_residual_add(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        &rhs_gpu,
        &dummy_gpu,
        &dummy_gpu,
        None,
        128,
        qtype,
        fused_residual_ref,
        &mut fused_gpu,
    )
    .map_err(|e| anyhow!("fused residual matmul: {e}"))?;
    if !handled {
        return Err(anyhow!(
            "{name} residual add was not handled by fused kernel"
        ));
    }

    let ref_bytes = ref_gpu
        .to_host_bytes()
        .map_err(|e| anyhow!("ref d2h: {e}"))?;
    let fused_bytes = fused_gpu
        .to_host_bytes()
        .map_err(|e| anyhow!("fused d2h: {e}"))?;
    let ref_host = bf16_bytes_to_f32(&ref_bytes);
    let fused_host = bf16_bytes_to_f32(&fused_bytes);
    let mut nbad = 0usize;
    let mut max_abs = 0f32;
    for i in 0..(m * n) {
        let d = (ref_host[i] - fused_host[i]).abs();
        max_abs = max_abs.max(d);
        if ref_bytes[2 * i..2 * i + 2] != fused_bytes[2 * i..2 * i + 2] {
            nbad += 1;
        }
    }
    println!("  max_abs={max_abs:.5e} byte_mismatch={nbad}/{}", m * n);
    if nbad > 0 {
        return Err(anyhow!("{name} residual add mismatches reference path"));
    }
    Ok(())
}

fn ggml_row_bytes_for(qtype: i32, k: usize) -> Result<usize> {
    qwen35::weights::ggml_k_row_bytes(qtype, k)
        .ok_or_else(|| anyhow!("unsupported GGML qtype {qtype} for k={k}"))
}

fn ggml_block_cols_for(qtype: i32) -> usize {
    if qtype == qwen35::weights::LOWBIT_GGML_Q8_0 {
        32
    } else {
        256
    }
}

fn ggml_blocks_for(qtype: i32, k: usize) -> Result<usize> {
    let block_cols = ggml_block_cols_for(qtype);
    if k % block_cols != 0 {
        return Err(anyhow!(
            "GGML qtype {qtype} bench requires k multiple of {block_cols}, got {k}"
        ));
    }
    Ok(k / block_cols)
}

fn make_ggml_k_slab(
    n: usize,
    k: usize,
    qtype: i32,
    row_fn: fn(usize) -> (Vec<u8>, Vec<f32>),
) -> Result<Vec<u8>> {
    let row_bytes = ggml_row_bytes_for(qtype, k)?;
    let blocks = ggml_blocks_for(qtype, k)?;
    let mut out = Vec::with_capacity(n * row_bytes);
    for row in 0..n {
        for block in 0..blocks {
            let (bytes, _) = row_fn(row.wrapping_mul(131).wrapping_add(block));
            out.extend_from_slice(&bytes);
        }
    }
    Ok(out)
}

fn make_bench_lhs(m: usize, k: usize) -> Vec<f32> {
    let mut lhs = vec![0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let x = (mi as f32 + 1.0) * 0.017 + (ki as f32) * 0.0017;
            lhs[mi * k + ki] = bf16_round(x.sin() * 0.75);
        }
    }
    lhs
}

fn expected_q8_1_group(vals: &[f32]) -> (f32, f32, Vec<i8>) {
    let amax = vals
        .iter()
        .fold(0.0f32, |acc, &v| if v.abs() > acc { v.abs() } else { acc });
    let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let sum = vals.iter().sum::<f32>();
    let qs = vals
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (d, sum, qs)
}

fn reference_q8_1_matmul(m: usize, n: usize, k: usize, lhs: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for group in 0..(k / 32) {
                let start = mi * k + group * 32;
                let (d, _, qs) = expected_q8_1_group(&lhs[start..start + 32]);
                for (i, &q) in qs.iter().enumerate() {
                    acc += rows[ni][group * 32 + i] * d * q as f32;
                }
            }
            out[mi * n + ni] = bf16_round(acc);
        }
    }
    out
}

fn run_mmq_q6_matmul_case(ordinal: usize, name: &str, m: usize, n: usize, k: usize) -> Result<()> {
    println!("=== {name}: m={m} n={n} k={k} ===");
    if k % 256 != 0 {
        return Err(anyhow!("Q6_K MMQ case requires k multiple of 256, got {k}"));
    }

    let lhs = make_bench_lhs(m, k);
    let mut rhs = Vec::new();
    let mut rows = Vec::new();
    for ni in 0..n {
        let mut vals = Vec::with_capacity(k);
        for block in 0..ggml_blocks_for(qwen35::weights::LOWBIT_GGML_Q6_K, k)? {
            let (bytes, block_vals) = ggml_q6k_row(ni.wrapping_mul(131).wrapping_add(block));
            rhs.extend_from_slice(&bytes);
            vals.extend_from_slice(&block_vals);
        }
        rows.push(vals);
    }
    let row_bytes = ggml_row_bytes_for(qwen35::weights::LOWBIT_GGML_Q6_K, k)?;
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, row_bytes], &rhs)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;
    let workspace_bytes = mmq_q8_1_workspace_bytes(1, m, k)?;
    let mut q8_gpu = GpuBuffer::alloc(ordinal, ScalarType::U8, &[workspace_bytes])
        .map_err(|e| anyhow!("q8 workspace alloc: {e}"))?;
    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;

    prefill_ffi::quantize_mmq_q8_1(
        ordinal,
        1,
        m,
        k,
        &lhs_gpu,
        qwen35::weights::LOWBIT_GGML_Q6_K,
        &mut q8_gpu,
    )
    .map_err(|e| anyhow!("q8_1 quant: {e}"))?;
    prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, 1, m, n, k, &q8_gpu, &rhs_gpu, &mut out_gpu)
        .map_err(|e| anyhow!("q6 mmq matmul: {e}"))?;

    let out_host = bf16_bytes_to_f32(
        &out_gpu
            .to_host_bytes()
            .map_err(|e| anyhow!("out d2h: {e}"))?,
    );
    let ref_out = reference_q8_1_matmul(m, n, k, &lhs, &rows);

    let mut nbad = 0usize;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut first_bad = None;
    for mi in 0..m {
        for ni in 0..n {
            let idx = mi * n + ni;
            let g = out_host[idx];
            let r = ref_out[idx];
            let abs = (g - r).abs();
            let rel = abs / r.abs().max(1.0e-5);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            if abs > 0.35 && rel > 0.03 {
                nbad += 1;
                if first_bad.is_none() {
                    first_bad = Some((mi, ni, g, r));
                }
            }
        }
    }
    println!(
        "  max_abs={max_abs:.5e} max_rel={max_rel:.5e} bad={nbad}/{}",
        m * n
    );
    if let Some((mi, ni, g, r)) = first_bad {
        println!("  first bad @ [{mi},{ni}]: gpu={g:.6} ref={r:.6}");
    }
    if nbad > 0 {
        return Err(anyhow!("{name} mismatches"));
    }
    Ok(())
}

fn run_mmq_q8_quant_case(ordinal: usize, name: &str, qtype: i32) -> Result<()> {
    let m = 2usize;
    let k = 256usize;
    println!("=== {name} Q8_1 quant layout: m={m} k={k} ===");
    let lhs = make_bench_lhs(m, k);
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let workspace_bytes = mmq_q8_1_workspace_bytes(1, m, k)?;
    let mut q8_gpu = GpuBuffer::alloc(ordinal, ScalarType::U8, &[workspace_bytes])
        .map_err(|e| anyhow!("q8 workspace alloc: {e}"))?;
    prefill_ffi::quantize_mmq_q8_1(ordinal, 1, m, k, &lhs_gpu, qtype, &mut q8_gpu)
        .map_err(|e| anyhow!("q8_1 quant: {e}"))?;
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("q8 sync: {e}"))?;
    let out = q8_gpu.to_host_bytes().map_err(|e| anyhow!("q8 d2h: {e}"))?;

    let blocks_per_row = k / 128;
    let mut nbad = 0usize;
    let mut max_q_diff = 0i32;
    let mut max_meta_abs = 0f32;
    for row in 0..m {
        for block in 0..blocks_per_row {
            // Lucebox/ggml MMQ stores activation blocks as
            // [k_block][row][block_q8_1_mmq], not [row][k_block].
            let block_base = (block * m + row) * 144;
            for group in 0..4 {
                let start = row * k + block * 128 + group * 32;
                let vals = &lhs[start..start + 32];
                let (d, sum, expected_qs) = expected_q8_1_group(vals);
                let qs_base = block_base + 16 + group * 32;
                for (i, &expected) in expected_qs.iter().enumerate() {
                    let got = out[qs_base + i] as i8;
                    let diff = (got as i32 - expected as i32).abs();
                    max_q_diff = max_q_diff.max(diff);
                    if diff != 0 {
                        nbad += 1;
                    }
                }
                if qtype == qwen35::weights::LOWBIT_GGML_Q6_K {
                    let off = block_base + group * 4;
                    let got =
                        f32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
                    max_meta_abs = max_meta_abs.max((got - d).abs());
                    if (got - d).abs() > 1.0e-6 {
                        nbad += 1;
                    }
                } else {
                    let off = block_base + group * 4;
                    let got_d =
                        f16::from_bits(u16::from_le_bytes([out[off], out[off + 1]])).to_f32();
                    let got_sum =
                        f16::from_bits(u16::from_le_bytes([out[off + 2], out[off + 3]])).to_f32();
                    let exp_d = f16::from_f32(d).to_f32();
                    let exp_sum = f16::from_f32(sum).to_f32();
                    max_meta_abs = max_meta_abs.max((got_d - exp_d).abs());
                    max_meta_abs = max_meta_abs.max((got_sum - exp_sum).abs());
                    if (got_d - exp_d).abs() > 0.0 || (got_sum - exp_sum).abs() > 0.0 {
                        nbad += 1;
                    }
                }
            }
        }
    }
    println!("  max_q_diff={max_q_diff} max_meta_abs={max_meta_abs:.5e} bad={nbad}");
    if nbad > 0 {
        return Err(anyhow!("{name} Q8_1 quant layout mismatches"));
    }
    Ok(())
}

fn bench_ggml_hot_shape(
    ordinal: usize,
    name: &str,
    qtype: i32,
    row_fn: fn(usize) -> (Vec<u8>, Vec<f32>),
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
) -> Result<()> {
    let row_bytes = ggml_row_bytes_for(qtype, k)?;
    println!("=== bench {name}: m={m} n={n} k={k} row_bytes={row_bytes} iters={iterations} ===");

    let lhs = make_bench_lhs(m, k);
    let rhs = make_ggml_k_slab(n, k, qtype, row_fn)?;
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, row_bytes], &rhs)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;
    let dummy_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, 1],
        &f32_to_bf16_bytes(&[0.0]),
    )
    .map_err(|e| anyhow!("dummy upload: {e}"))?;
    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;

    for _ in 0..3 {
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            m,
            n,
            k,
            &lhs_gpu,
            &rhs_gpu,
            &dummy_gpu,
            &dummy_gpu,
            None,
            128,
            qtype,
            &mut out_gpu,
        )
        .map_err(|e| anyhow!("warmup ggml matmul: {e}"))?;
    }
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("warmup sync: {e}"))?;

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            m,
            n,
            k,
            &lhs_gpu,
            &rhs_gpu,
            &dummy_gpu,
            &dummy_gpu,
            None,
            128,
            qtype,
            &mut out_gpu,
        )
        .map_err(|e| anyhow!("bench ggml matmul: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench sync: {e}"))?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p90 = samples[((samples.len() * 9) / 10).min(samples.len() - 1)];
    println!("  mean_ms={mean:.4} p50_ms={p50:.4} p90_ms={p90:.4}");
    Ok(())
}

fn bench_ggml_pair_m16_hot_shape(
    ordinal: usize,
    name: &str,
    qtype: i32,
    row_fn: fn(usize) -> (Vec<u8>, Vec<f32>),
    n_each: usize,
    k: usize,
    iterations: usize,
) -> Result<()> {
    let m = 16usize;
    let row_bytes = ggml_row_bytes_for(qtype, k)?;
    println!(
        "=== bench {name} pair m16: m={m} n_each={n_each} k={k} row_bytes={row_bytes} iters={iterations} ==="
    );

    let lhs = make_bench_lhs(m, k);
    let rhs_first = make_ggml_k_slab(n_each, k, qtype, row_fn)?;
    let rhs_second = make_ggml_k_slab(n_each, k, qtype, row_fn)?;
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let rhs_first_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n_each, row_bytes], &rhs_first)
            .map_err(|e| anyhow!("rhs_first upload: {e}"))?;
    let rhs_second_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n_each, row_bytes], &rhs_second)
            .map_err(|e| anyhow!("rhs_second upload: {e}"))?;
    let dummy_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, 1],
        &f32_to_bf16_bytes(&[0.0]),
    )
    .map_err(|e| anyhow!("dummy upload: {e}"))?;
    let mut out_first = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n_each])
        .map_err(|e| anyhow!("out_first alloc: {e}"))?;
    let mut out_second = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n_each])
        .map_err(|e| anyhow!("out_second alloc: {e}"))?;
    let mut out_pair = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n_each * 2])
        .map_err(|e| anyhow!("out_pair alloc: {e}"))?;
    let mut out_swiglu_ref = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n_each])
        .map_err(|e| anyhow!("out_swiglu_ref alloc: {e}"))?;
    let mut out_swiglu_fused = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n_each])
        .map_err(|e| anyhow!("out_swiglu_fused alloc: {e}"))?;

    for _ in 0..3 {
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &dummy_gpu,
            &dummy_gpu,
            None,
            128,
            qtype,
            &mut out_first,
        )
        .map_err(|e| anyhow!("warmup first ggml matmul: {e}"))?;
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_second_gpu,
            &dummy_gpu,
            &dummy_gpu,
            None,
            128,
            qtype,
            &mut out_second,
        )
        .map_err(|e| anyhow!("warmup second ggml matmul: {e}"))?;
        prefill_ffi::matmul_rhs_transposed_ggml_pair(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &rhs_second_gpu,
            qtype,
            &mut out_pair,
        )
        .map_err(|e| anyhow!("warmup pair ggml matmul: {e}"))?;
        prefill_ffi::swiglu_mul_split(
            ordinal,
            ScalarType::BF16,
            m,
            n_each,
            &out_pair,
            &mut out_swiglu_ref,
        )
        .map_err(|e| anyhow!("warmup pair swiglu split: {e}"))?;
        let fused = prefill_ffi::matmul_rhs_transposed_ggml_pair_swiglu(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &rhs_second_gpu,
            qtype,
            &mut out_swiglu_fused,
        )
        .map_err(|e| anyhow!("warmup fused pair swiglu: {e}"))?;
        if !fused {
            return Err(anyhow!("fused pair swiglu unsupported for {name}"));
        }
    }
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("warmup sync: {e}"))?;

    let first_bytes = out_first
        .to_host_bytes()
        .map_err(|e| anyhow!("first d2h: {e}"))?;
    let second_bytes = out_second
        .to_host_bytes()
        .map_err(|e| anyhow!("second d2h: {e}"))?;
    let pair_bytes = out_pair
        .to_host_bytes()
        .map_err(|e| anyhow!("pair d2h: {e}"))?;
    let first_host = bf16_bytes_to_f32(&first_bytes);
    let second_host = bf16_bytes_to_f32(&second_bytes);
    let pair_host = bf16_bytes_to_f32(&pair_bytes);
    let mut byte_mismatch = 0usize;
    let mut max_abs = 0f32;
    for row in 0..m {
        for col in 0..n_each {
            let sep_idx = row * n_each + col;
            let pair_first_idx = row * n_each * 2 + col;
            let pair_second_idx = row * n_each * 2 + n_each + col;
            max_abs = max_abs.max((first_host[sep_idx] - pair_host[pair_first_idx]).abs());
            max_abs = max_abs.max((second_host[sep_idx] - pair_host[pair_second_idx]).abs());
            if first_bytes[2 * sep_idx..2 * sep_idx + 2]
                != pair_bytes[2 * pair_first_idx..2 * pair_first_idx + 2]
            {
                byte_mismatch += 1;
            }
            if second_bytes[2 * sep_idx..2 * sep_idx + 2]
                != pair_bytes[2 * pair_second_idx..2 * pair_second_idx + 2]
            {
                byte_mismatch += 1;
            }
        }
    }
    println!(
        "  pair_vs_two_fixed_m16 max_abs={max_abs:.5e} byte_mismatch={byte_mismatch}/{}",
        m * n_each * 2
    );

    prefill_ffi::swiglu_mul_split(
        ordinal,
        ScalarType::BF16,
        m,
        n_each,
        &out_pair,
        &mut out_swiglu_ref,
    )
    .map_err(|e| anyhow!("reference pair swiglu split: {e}"))?;
    let fused = prefill_ffi::matmul_rhs_transposed_ggml_pair_swiglu(
        ordinal,
        1,
        m,
        n_each,
        k,
        &lhs_gpu,
        &rhs_first_gpu,
        &rhs_second_gpu,
        qtype,
        &mut out_swiglu_fused,
    )
    .map_err(|e| anyhow!("fused pair swiglu: {e}"))?;
    if !fused {
        return Err(anyhow!("fused pair swiglu unsupported for {name}"));
    }
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("fused parity sync: {e}"))?;
    let swiglu_ref_bytes = out_swiglu_ref
        .to_host_bytes()
        .map_err(|e| anyhow!("swiglu ref d2h: {e}"))?;
    let swiglu_fused_bytes = out_swiglu_fused
        .to_host_bytes()
        .map_err(|e| anyhow!("swiglu fused d2h: {e}"))?;
    let swiglu_ref_host = bf16_bytes_to_f32(&swiglu_ref_bytes);
    let swiglu_fused_host = bf16_bytes_to_f32(&swiglu_fused_bytes);
    let mut swiglu_byte_mismatch = 0usize;
    let mut swiglu_max_abs = 0f32;
    for idx in 0..m * n_each {
        swiglu_max_abs = swiglu_max_abs.max((swiglu_ref_host[idx] - swiglu_fused_host[idx]).abs());
        if swiglu_ref_bytes[2 * idx..2 * idx + 2] != swiglu_fused_bytes[2 * idx..2 * idx + 2] {
            swiglu_byte_mismatch += 1;
        }
    }
    println!(
        "  fused_swiglu_vs_pair_split max_abs={swiglu_max_abs:.5e} byte_mismatch={swiglu_byte_mismatch}/{}",
        m * n_each
    );
    if swiglu_byte_mismatch != 0 {
        return Err(anyhow!("fused pair swiglu byte mismatch for {name}"));
    }

    let mut separate_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &dummy_gpu,
            &dummy_gpu,
            None,
            128,
            qtype,
            &mut out_first,
        )
        .map_err(|e| anyhow!("bench first ggml matmul: {e}"))?;
        prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_second_gpu,
            &dummy_gpu,
            &dummy_gpu,
            None,
            128,
            qtype,
            &mut out_second,
        )
        .map_err(|e| anyhow!("bench second ggml matmul: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench separate sync: {e}"))?;
        separate_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut pair_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_rhs_transposed_ggml_pair(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &rhs_second_gpu,
            qtype,
            &mut out_pair,
        )
        .map_err(|e| anyhow!("bench pair ggml matmul: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench pair sync: {e}"))?;
        pair_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut pair_swiglu_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_rhs_transposed_ggml_pair(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &rhs_second_gpu,
            qtype,
            &mut out_pair,
        )
        .map_err(|e| anyhow!("bench pair ggml matmul before swiglu: {e}"))?;
        prefill_ffi::swiglu_mul_split(
            ordinal,
            ScalarType::BF16,
            m,
            n_each,
            &out_pair,
            &mut out_swiglu_ref,
        )
        .map_err(|e| anyhow!("bench pair swiglu split: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench pair swiglu sync: {e}"))?;
        pair_swiglu_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut fused_swiglu_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let fused = prefill_ffi::matmul_rhs_transposed_ggml_pair_swiglu(
            ordinal,
            1,
            m,
            n_each,
            k,
            &lhs_gpu,
            &rhs_first_gpu,
            &rhs_second_gpu,
            qtype,
            &mut out_swiglu_fused,
        )
        .map_err(|e| anyhow!("bench fused pair swiglu: {e}"))?;
        if !fused {
            return Err(anyhow!(
                "fused pair swiglu unsupported during bench for {name}"
            ));
        }
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench fused swiglu sync: {e}"))?;
        fused_swiglu_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    separate_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pair_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pair_swiglu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    fused_swiglu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let separate_mean = separate_samples.iter().sum::<f64>() / separate_samples.len() as f64;
    let pair_mean = pair_samples.iter().sum::<f64>() / pair_samples.len() as f64;
    let pair_swiglu_mean =
        pair_swiglu_samples.iter().sum::<f64>() / pair_swiglu_samples.len() as f64;
    let fused_swiglu_mean =
        fused_swiglu_samples.iter().sum::<f64>() / fused_swiglu_samples.len() as f64;
    let separate_p50 = separate_samples[separate_samples.len() / 2];
    let pair_p50 = pair_samples[pair_samples.len() / 2];
    let pair_swiglu_p50 = pair_swiglu_samples[pair_swiglu_samples.len() / 2];
    let fused_swiglu_p50 = fused_swiglu_samples[fused_swiglu_samples.len() / 2];
    let separate_p90 =
        separate_samples[((separate_samples.len() * 9) / 10).min(separate_samples.len() - 1)];
    let pair_p90 = pair_samples[((pair_samples.len() * 9) / 10).min(pair_samples.len() - 1)];
    let pair_swiglu_p90 = pair_swiglu_samples
        [((pair_swiglu_samples.len() * 9) / 10).min(pair_swiglu_samples.len() - 1)];
    let fused_swiglu_p90 = fused_swiglu_samples
        [((fused_swiglu_samples.len() * 9) / 10).min(fused_swiglu_samples.len() - 1)];
    println!(
        "  two_fixed_m16 mean_ms={separate_mean:.4} p50_ms={separate_p50:.4} p90_ms={separate_p90:.4}"
    );
    println!("  pair_ffi mean_ms={pair_mean:.4} p50_ms={pair_p50:.4} p90_ms={pair_p90:.4}");
    println!(
        "  pair_ffi_vs_two_fixed speedup={:.3}x",
        separate_mean / pair_mean
    );
    println!(
        "  pair_plus_swiglu mean_ms={pair_swiglu_mean:.4} p50_ms={pair_swiglu_p50:.4} p90_ms={pair_swiglu_p90:.4}"
    );
    println!(
        "  fused_pair_swiglu mean_ms={fused_swiglu_mean:.4} p50_ms={fused_swiglu_p50:.4} p90_ms={fused_swiglu_p90:.4}"
    );
    println!(
        "  fused_pair_swiglu_vs_pair_plus_swiglu speedup={:.3}x",
        pair_swiglu_mean / fused_swiglu_mean
    );
    Ok(())
}

fn mmq_q8_1_workspace_bytes(batch: usize, m: usize, k: usize) -> Result<usize> {
    if k % 128 != 0 {
        return Err(anyhow!(
            "MMQ Q8_1 workspace requires k multiple of 128, got {k}"
        ));
    }
    Ok(batch * m * (k / 128) * 144)
}

fn bench_mmq_q8_quant_hot_shape(
    ordinal: usize,
    name: &str,
    qtype: i32,
    m: usize,
    k: usize,
    iterations: usize,
) -> Result<()> {
    let workspace_bytes = mmq_q8_1_workspace_bytes(1, m, k)?;
    println!(
        "=== bench {name} q8_1 quant: m={m} k={k} workspace_bytes={workspace_bytes} iters={iterations} ==="
    );

    let lhs = make_bench_lhs(m, k);
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let mut q8_gpu = GpuBuffer::alloc(ordinal, ScalarType::U8, &[workspace_bytes])
        .map_err(|e| anyhow!("q8 workspace alloc: {e}"))?;

    for _ in 0..3 {
        prefill_ffi::quantize_mmq_q8_1(ordinal, 1, m, k, &lhs_gpu, qtype, &mut q8_gpu)
            .map_err(|e| anyhow!("warmup q8_1 quant: {e}"))?;
    }
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("warmup sync: {e}"))?;

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::quantize_mmq_q8_1(ordinal, 1, m, k, &lhs_gpu, qtype, &mut q8_gpu)
            .map_err(|e| anyhow!("bench q8_1 quant: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench sync: {e}"))?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p90 = samples[((samples.len() * 9) / 10).min(samples.len() - 1)];
    println!("  mean_ms={mean:.4} p50_ms={p50:.4} p90_ms={p90:.4}");
    Ok(())
}

fn bench_mmq_q6_matmul_hot_shape(
    ordinal: usize,
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
) -> Result<()> {
    let row_bytes = ggml_row_bytes_for(qwen35::weights::LOWBIT_GGML_Q6_K, k)?;
    let workspace_bytes = mmq_q8_1_workspace_bytes(1, m, k)?;
    println!(
        "=== bench {name} q6_k mmq: m={m} n={n} k={k} row_bytes={row_bytes} workspace_bytes={workspace_bytes} iters={iterations} ==="
    );

    let lhs = make_bench_lhs(m, k);
    let residual = make_bench_lhs(m, n);
    let rhs = make_ggml_k_slab(n, k, qwen35::weights::LOWBIT_GGML_Q6_K, ggml_q6k_row)?;
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;
    let residual_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[m, n],
        &f32_to_bf16_bytes(&residual),
    )
    .map_err(|e| anyhow!("residual upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, row_bytes], &rhs)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;
    let mut q8_gpu = GpuBuffer::alloc(ordinal, ScalarType::U8, &[workspace_bytes])
        .map_err(|e| anyhow!("q8 workspace alloc: {e}"))?;
    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;
    let mut out_residual_ref = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out residual ref alloc: {e}"))?;
    let mut out_residual_fused = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out residual fused alloc: {e}"))?;

    prefill_ffi::quantize_mmq_q8_1(
        ordinal,
        1,
        m,
        k,
        &lhs_gpu,
        qwen35::weights::LOWBIT_GGML_Q6_K,
        &mut q8_gpu,
    )
    .map_err(|e| anyhow!("q8_1 quant: {e}"))?;
    for _ in 0..3 {
        prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, 1, m, n, k, &q8_gpu, &rhs_gpu, &mut out_gpu)
            .map_err(|e| anyhow!("warmup q6 mmq matmul: {e}"))?;
        prefill_ffi::element_add(
            ordinal,
            ScalarType::BF16,
            m * n,
            &residual_gpu,
            &out_gpu,
            &mut out_residual_ref,
        )
        .map_err(|e| anyhow!("warmup q6 mmq residual ref: {e}"))?;
        prefill_ffi::matmul_mmq_q8_1_q6_k_residual_add(
            ordinal,
            1,
            m,
            n,
            k,
            &q8_gpu,
            &rhs_gpu,
            &residual_gpu,
            &mut out_residual_fused,
        )
        .map_err(|e| anyhow!("warmup q6 mmq residual fused: {e}"))?;
    }
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("warmup sync: {e}"))?;

    prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, 1, m, n, k, &q8_gpu, &rhs_gpu, &mut out_gpu)
        .map_err(|e| anyhow!("parity q6 mmq matmul: {e}"))?;
    prefill_ffi::element_add(
        ordinal,
        ScalarType::BF16,
        m * n,
        &residual_gpu,
        &out_gpu,
        &mut out_residual_ref,
    )
    .map_err(|e| anyhow!("parity q6 mmq residual ref: {e}"))?;
    prefill_ffi::matmul_mmq_q8_1_q6_k_residual_add(
        ordinal,
        1,
        m,
        n,
        k,
        &q8_gpu,
        &rhs_gpu,
        &residual_gpu,
        &mut out_residual_fused,
    )
    .map_err(|e| anyhow!("parity q6 mmq residual fused: {e}"))?;
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("q6 mmq residual parity sync: {e}"))?;
    let ref_bytes = out_residual_ref
        .to_host_bytes()
        .map_err(|e| anyhow!("q6 mmq residual ref d2h: {e}"))?;
    let fused_bytes = out_residual_fused
        .to_host_bytes()
        .map_err(|e| anyhow!("q6 mmq residual fused d2h: {e}"))?;
    let ref_host = bf16_bytes_to_f32(&ref_bytes);
    let fused_host = bf16_bytes_to_f32(&fused_bytes);
    let mut byte_mismatch = 0usize;
    let mut max_abs = 0f32;
    for idx in 0..m * n {
        max_abs = max_abs.max((ref_host[idx] - fused_host[idx]).abs());
        if ref_bytes[2 * idx..2 * idx + 2] != fused_bytes[2 * idx..2 * idx + 2] {
            byte_mismatch += 1;
        }
    }
    println!(
        "  residual_fused_vs_mmq_plus_add max_abs={max_abs:.5e} byte_mismatch={byte_mismatch}/{}",
        m * n
    );
    if byte_mismatch != 0 {
        return Err(anyhow!("{name} residual fused byte mismatch"));
    }

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, 1, m, n, k, &q8_gpu, &rhs_gpu, &mut out_gpu)
            .map_err(|e| anyhow!("bench q6 mmq matmul: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench sync: {e}"))?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut residual_ref_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_mmq_q8_1_q6_k(ordinal, 1, m, n, k, &q8_gpu, &rhs_gpu, &mut out_gpu)
            .map_err(|e| anyhow!("bench q6 mmq residual matmul: {e}"))?;
        prefill_ffi::element_add(
            ordinal,
            ScalarType::BF16,
            m * n,
            &residual_gpu,
            &out_gpu,
            &mut out_residual_ref,
        )
        .map_err(|e| anyhow!("bench q6 mmq residual add: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench residual ref sync: {e}"))?;
        residual_ref_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut residual_fused_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        prefill_ffi::matmul_mmq_q8_1_q6_k_residual_add(
            ordinal,
            1,
            m,
            n,
            k,
            &q8_gpu,
            &rhs_gpu,
            &residual_gpu,
            &mut out_residual_fused,
        )
        .map_err(|e| anyhow!("bench q6 mmq residual fused: {e}"))?;
        gpu_hal::sync(ordinal).map_err(|e| anyhow!("bench residual fused sync: {e}"))?;
        residual_fused_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    residual_ref_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    residual_fused_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p90 = samples[((samples.len() * 9) / 10).min(samples.len() - 1)];
    println!("  mean_ms={mean:.4} p50_ms={p50:.4} p90_ms={p90:.4}");
    let residual_ref_mean =
        residual_ref_samples.iter().sum::<f64>() / residual_ref_samples.len() as f64;
    let residual_fused_mean =
        residual_fused_samples.iter().sum::<f64>() / residual_fused_samples.len() as f64;
    let residual_ref_p50 = residual_ref_samples[residual_ref_samples.len() / 2];
    let residual_fused_p50 = residual_fused_samples[residual_fused_samples.len() / 2];
    let residual_ref_p90 = residual_ref_samples
        [((residual_ref_samples.len() * 9) / 10).min(residual_ref_samples.len() - 1)];
    let residual_fused_p90 = residual_fused_samples
        [((residual_fused_samples.len() * 9) / 10).min(residual_fused_samples.len() - 1)];
    println!(
        "  mmq_plus_add mean_ms={residual_ref_mean:.4} p50_ms={residual_ref_p50:.4} p90_ms={residual_ref_p90:.4}"
    );
    println!(
        "  mmq_residual_fused mean_ms={residual_fused_mean:.4} p50_ms={residual_fused_p50:.4} p90_ms={residual_fused_p90:.4}"
    );
    println!(
        "  mmq_residual_fused_vs_mmq_plus_add speedup={:.3}x",
        residual_ref_mean / residual_fused_mean
    );
    Ok(())
}

fn run_q6_k_m16_argmax_case(ordinal: usize) -> Result<()> {
    let m = 16usize;
    let n = 256usize;
    let k = 512usize;
    let row_bytes = ggml_row_bytes_for(qwen35::weights::LOWBIT_GGML_Q6_K, k)?;
    let tiles = n / 16;
    println!("=== q6_k m16 fused argmax parity: m={m} n={n} k={k} ===");

    let lhs = make_bench_lhs(m, k);
    let rhs = make_ggml_k_slab(n, k, qwen35::weights::LOWBIT_GGML_Q6_K, ggml_q6k_row)?;
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("argmax lhs upload: {e}"))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, row_bytes], &rhs)
        .map_err(|e| anyhow!("argmax rhs upload: {e}"))?;
    let mut logits_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("argmax logits alloc: {e}"))?;
    let mut ref_indices = GpuBuffer::zeros(ordinal, ScalarType::U32, &[m])
        .map_err(|e| anyhow!("argmax ref indices alloc: {e}"))?;
    let mut fused_indices = GpuBuffer::zeros(ordinal, ScalarType::U32, &[m])
        .map_err(|e| anyhow!("argmax fused indices alloc: {e}"))?;
    let mut block_best_vals = GpuBuffer::zeros(ordinal, ScalarType::F32, &[m, tiles])
        .map_err(|e| anyhow!("argmax block vals alloc: {e}"))?;
    let mut block_best_indices = GpuBuffer::zeros(ordinal, ScalarType::U32, &[m, tiles])
        .map_err(|e| anyhow!("argmax block indices alloc: {e}"))?;

    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        &rhs_gpu,
        &rhs_gpu,
        &rhs_gpu,
        None,
        128,
        qwen35::weights::LOWBIT_GGML_Q6_K,
        &mut logits_gpu,
    )
    .map_err(|e| anyhow!("argmax reference q6 lm_head matmul: {e}"))?;
    prefill_ffi::argmax_bf16_rows(ordinal, m, n, &logits_gpu, &mut ref_indices)
        .map_err(|e| anyhow!("argmax reference rows: {e}"))?;
    let fused = prefill_ffi::matmul_q6_k_m16_argmax(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        &rhs_gpu,
        &mut block_best_vals,
        &mut block_best_indices,
        &mut fused_indices,
    )
    .map_err(|e| anyhow!("argmax fused q6: {e}"))?;
    if !fused {
        return Err(anyhow!("q6_k m16 fused argmax unsupported"));
    }
    gpu_hal::sync(ordinal).map_err(|e| anyhow!("argmax fused sync: {e}"))?;

    let ref_bytes = ref_indices
        .to_host_bytes()
        .map_err(|e| anyhow!("argmax ref d2h: {e}"))?;
    let fused_bytes = fused_indices
        .to_host_bytes()
        .map_err(|e| anyhow!("argmax fused d2h: {e}"))?;
    let mut mismatches = 0usize;
    for row in 0..m {
        let start = row * 4;
        let ref_id = u32::from_le_bytes([
            ref_bytes[start],
            ref_bytes[start + 1],
            ref_bytes[start + 2],
            ref_bytes[start + 3],
        ]);
        let fused_id = u32::from_le_bytes([
            fused_bytes[start],
            fused_bytes[start + 1],
            fused_bytes[start + 2],
            fused_bytes[start + 3],
        ]);
        if ref_id != fused_id {
            mismatches += 1;
            println!("  row={row} ref_id={ref_id} fused_id={fused_id}");
        }
    }
    println!("  fused_argmax_mismatches={mismatches}/{m}");
    if mismatches != 0 {
        return Err(anyhow!("q6_k m16 fused argmax mismatch"));
    }
    Ok(())
}

fn maybe_run_ggml_hot_benches(ordinal: usize) -> Result<()> {
    if std::env::var_os("SUPERSONIC_INT4_TEST_BENCH_GGML_HOT").is_none() {
        return Ok(());
    }
    let iterations = std::env::var("SUPERSONIC_INT4_TEST_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(20)
        .max(1);
    if std::env::var_os("SUPERSONIC_INT4_TEST_BENCH_Q6_DOWN_M16_ONLY").is_some() {
        bench_mmq_q6_matmul_hot_shape(ordinal, "Q6_K down hot m16", 16, 5_120, 17_408, iterations)?;
        return Ok(());
    }
    bench_ggml_hot_shape(
        ordinal,
        "Q4_K mlp gate/up hot",
        qwen35::weights::LOWBIT_GGML_Q4_K,
        ggml_q4k_row,
        8,
        17_408,
        5_120,
        iterations,
    )?;
    bench_ggml_hot_shape(
        ordinal,
        "Q6_K down hot",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
        8,
        5_120,
        17_408,
        iterations,
    )?;
    bench_ggml_hot_shape(
        ordinal,
        "Q4_K down hot",
        qwen35::weights::LOWBIT_GGML_Q4_K,
        ggml_q4k_row,
        8,
        5_120,
        17_408,
        iterations,
    )?;
    bench_ggml_hot_shape(
        ordinal,
        "Q5_K linear hot",
        qwen35::weights::LOWBIT_GGML_Q5_K,
        ggml_q5k_row,
        8,
        5_120,
        6_144,
        iterations,
    )?;
    bench_ggml_hot_shape(
        ordinal,
        "Q6_K vocab row-scan hot",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
        8,
        248_320,
        5_120,
        iterations,
    )?;
    bench_ggml_hot_shape(
        ordinal,
        "Q6_K mid linear hot",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
        8,
        10_240,
        5_120,
        iterations,
    )?;
    bench_ggml_hot_shape(
        ordinal,
        "Q6_K small linear hot",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
        8,
        1_024,
        5_120,
        iterations,
    )?;
    if std::env::var_os("SUPERSONIC_INT4_TEST_BENCH_MMQ_Q8").is_some() {
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q4_K mlp gate/up hot",
            qwen35::weights::LOWBIT_GGML_Q4_K,
            8,
            5_120,
            iterations,
        )?;
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q6_K down hot",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            8,
            17_408,
            iterations,
        )?;
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q4_K down hot",
            qwen35::weights::LOWBIT_GGML_Q4_K,
            8,
            17_408,
            iterations,
        )?;
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q5_K linear hot",
            qwen35::weights::LOWBIT_GGML_Q5_K,
            8,
            6_144,
            iterations,
        )?;
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q6_K vocab row-scan hot",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            8,
            5_120,
            iterations,
        )?;
    }
    if std::env::var_os("SUPERSONIC_INT4_TEST_BENCH_MMQ_Q6_MATMUL").is_some() {
        bench_mmq_q6_matmul_hot_shape(ordinal, "Q6_K down hot", 8, 5_120, 17_408, iterations)?;
        bench_mmq_q6_matmul_hot_shape(
            ordinal,
            "Q6_K vocab row-scan hot",
            8,
            248_320,
            5_120,
            iterations,
        )?;
        bench_mmq_q6_matmul_hot_shape(
            ordinal,
            "Q6_K mid linear hot",
            8,
            10_240,
            5_120,
            iterations,
        )?;
        bench_mmq_q6_matmul_hot_shape(
            ordinal,
            "Q6_K small linear hot",
            8,
            1_024,
            5_120,
            iterations,
        )?;
    }
    if std::env::var_os("SUPERSONIC_INT4_TEST_BENCH_M16_HOT").is_some() {
        bench_ggml_hot_shape(
            ordinal,
            "Q4_K mlp gate/up hot m16",
            qwen35::weights::LOWBIT_GGML_Q4_K,
            ggml_q4k_row,
            16,
            17_408,
            5_120,
            iterations,
        )?;
        bench_ggml_hot_shape(
            ordinal,
            "Q4_K down hot m16",
            qwen35::weights::LOWBIT_GGML_Q4_K,
            ggml_q4k_row,
            16,
            5_120,
            17_408,
            iterations,
        )?;
        bench_ggml_hot_shape(
            ordinal,
            "Q5_K linear hot m16",
            qwen35::weights::LOWBIT_GGML_Q5_K,
            ggml_q5k_row,
            16,
            5_120,
            6_144,
            iterations,
        )?;
        bench_ggml_hot_shape(
            ordinal,
            "Q6_K down hot m16",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            ggml_q6k_row,
            16,
            5_120,
            17_408,
            iterations,
        )?;
        bench_ggml_hot_shape(
            ordinal,
            "Q6_K vocab row-scan hot m16",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            ggml_q6k_row,
            16,
            248_320,
            5_120,
            iterations,
        )?;
        bench_ggml_hot_shape(
            ordinal,
            "Q6_K mid linear hot m16",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            ggml_q6k_row,
            16,
            10_240,
            5_120,
            iterations,
        )?;
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q6_K down hot m16",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            16,
            17_408,
            iterations,
        )?;
        bench_mmq_q8_quant_hot_shape(
            ordinal,
            "Q6_K vocab row-scan hot m16",
            qwen35::weights::LOWBIT_GGML_Q6_K,
            16,
            5_120,
            iterations,
        )?;
        bench_mmq_q6_matmul_hot_shape(ordinal, "Q6_K down hot m16", 16, 5_120, 17_408, iterations)?;
        bench_mmq_q6_matmul_hot_shape(
            ordinal,
            "Q6_K vocab row-scan hot m16",
            16,
            248_320,
            5_120,
            iterations,
        )?;
        bench_mmq_q6_matmul_hot_shape(
            ordinal,
            "Q6_K mid linear hot m16",
            16,
            10_240,
            5_120,
            iterations,
        )?;
    }
    if std::env::var_os("SUPERSONIC_INT4_TEST_BENCH_GGML_PAIR_M16").is_some() {
        bench_ggml_pair_m16_hot_shape(
            ordinal,
            "Q4_K mlp gate/up hot",
            qwen35::weights::LOWBIT_GGML_Q4_K,
            ggml_q4k_row,
            17_408,
            5_120,
            iterations,
        )?;
    }
    Ok(())
}

struct TestCase {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    gs: usize,
}

fn run_case(ordinal: usize, c: &TestCase) -> Result<()> {
    let TestCase { name, m, n, k, gs } = *c;
    println!("=== {name}: m={m} n={n} k={k} gs={gs} ===");

    if k % 2 != 0 {
        return Err(anyhow!("k must be even"));
    }
    // Scale layout matches the kernel: [ceil(n/gs), ceil(k/gs)].
    let sr = (n + gs - 1) / gs;
    let sc = (k + gs - 1) / gs;

    // --- Deterministic inputs ---
    // lhs: [m, k] varied small values, bf16-rounded.
    let mut lhs = vec![0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let v = ((mi as f32) * 0.125 + (ki as f32) * 0.03125).sin() * 0.5;
            lhs[mi * k + ki] = bf16_round(v);
        }
    }

    // nibbles: [n, k] deterministic 0..15
    let mut nibbles = vec![0u8; n * k];
    for ni in 0..n {
        for ki in 0..k {
            nibbles[ni * k + ki] = ((ni * 131 + ki * 17) & 0xF) as u8;
        }
    }

    // scales/zeros: [sr, sc] - use small varied positive scales and non-zero zeros.
    let mut scales = vec![0f32; sr * sc];
    let mut zeros = vec![0f32; sr * sc];
    for i in 0..sr {
        for j in 0..sc {
            let s = 0.004 + (i as f32) * 0.001 + (j as f32) * 0.00025;
            let z = 6.0 + ((i + j) as f32) * 0.3;
            scales[i * sc + j] = bf16_round(s);
            zeros[i * sc + j] = bf16_round(z);
        }
    }

    // --- GPU buffers ---
    let lhs_gpu =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
            .map_err(|e| anyhow!("lhs upload: {e}"))?;

    let packed = pack_nibbles(&nibbles, n, k);
    let rhs_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[n, k / 2], &packed)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;

    let scale_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[sr, sc],
        &f32_to_bf16_bytes(&scales),
    )
    .map_err(|e| anyhow!("scale upload: {e}"))?;

    let zero_gpu = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[sr, sc],
        &f32_to_bf16_bytes(&zeros),
    )
    .map_err(|e| anyhow!("zero upload: {e}"))?;

    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;

    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        n,
        k,
        &lhs_gpu,
        &rhs_gpu,
        &scale_gpu,
        &zero_gpu,
        None,
        gs,
        qwen35::weights::LOWBIT_NATIVE_INT4,
        &mut out_gpu,
    )
    .map_err(|e| anyhow!("int4 matmul: {e}"))?;

    let out_host = bf16_bytes_to_f32(
        &out_gpu
            .to_host_bytes()
            .map_err(|e| anyhow!("out d2h: {e}"))?,
    );

    let ref_out = reference_matmul(m, n, k, gs, &lhs, &nibbles, &scales, &zeros);

    // --- Compare ---
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    let mut nbad = 0usize;
    for mi in 0..m {
        for ni in 0..n {
            let g = out_host[mi * n + ni];
            let r = ref_out[mi * n + ni];
            let abs_diff = (g - r).abs();
            let rel = abs_diff / r.abs().max(1e-6);
            max_abs = max_abs.max(abs_diff);
            max_rel = max_rel.max(rel);
            // Tolerance: ~half a bf16 ULP at this magnitude; allow 1e-2 because
            // the kernel does NOT round through bf16 mid-accumulation.
            if abs_diff > 0.05 && rel > 0.02 {
                nbad += 1;
                if first_bad.is_none() {
                    first_bad = Some((mi, ni, g, r));
                }
            }
        }
    }
    println!(
        "  max_abs={max_abs:.5e}  max_rel={max_rel:.5e}  bad={nbad}/{}",
        m * n
    );
    if let Some((mi, ni, g, r)) = first_bad {
        println!("  first bad @ [{mi},{ni}]: gpu={g:.6} ref={r:.6}");
    }

    // Dump a few sample values for sanity
    println!(
        "  samples: gpu[0,0]={:.4}  ref[0,0]={:.4}  gpu[{}, {}]={:.4}  ref={:.4}",
        out_host[0],
        ref_out[0],
        m - 1,
        n - 1,
        out_host[(m - 1) * n + (n - 1)],
        ref_out[(m - 1) * n + (n - 1)],
    );

    if nbad > 0 {
        println!("  FAIL: {nbad} mismatches");
    } else {
        println!("  OK");
    }

    Ok(())
}

fn main() -> Result<()> {
    let ordinal = 0usize;
    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    let cases = [
        TestCase {
            name: "single group, single tile",
            m: 16,
            n: 16,
            k: 128,
            gs: 128,
        },
        TestCase {
            name: "2 groups in k",
            m: 16,
            n: 16,
            k: 256,
            gs: 128,
        },
        TestCase {
            name: "2 groups in n",
            m: 16,
            n: 32,
            k: 128,
            gs: 128,
        },
        TestCase {
            name: "multi-tile, aligned",
            m: 32,
            n: 32,
            k: 256,
            gs: 128,
        },
        TestCase {
            name: "Qwen-size group=128",
            m: 1,
            n: 128,
            k: 256,
            gs: 128,
        },
        TestCase {
            name: "k spans many groups",
            m: 4,
            n: 16,
            k: 1024,
            gs: 128,
        },
        TestCase {
            name: "prefill-like shape",
            m: 8,
            n: 128,
            k: 2560,
            gs: 128,
        },
    ];

    for c in &cases {
        run_case(ordinal, c)?;
    }
    run_ggml_case(
        ordinal,
        "GGML Q8_0",
        qwen35::weights::LOWBIT_GGML_Q8_0,
        ggml_q8_0_row,
    )?;
    run_ggml_case(
        ordinal,
        "GGML Q4_K",
        qwen35::weights::LOWBIT_GGML_Q4_K,
        ggml_q4k_row,
    )?;
    run_ggml_case(
        ordinal,
        "GGML Q5_K",
        qwen35::weights::LOWBIT_GGML_Q5_K,
        ggml_q5k_row,
    )?;
    run_ggml_case(
        ordinal,
        "GGML Q6_K",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q8_0 m8 aligned",
        qwen35::weights::LOWBIT_GGML_Q8_0,
        ggml_q8_0_row,
        8,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q4_K m8 aligned",
        qwen35::weights::LOWBIT_GGML_Q4_K,
        ggml_q4k_row,
        8,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q5_K m8 aligned",
        qwen35::weights::LOWBIT_GGML_Q5_K,
        ggml_q5k_row,
        8,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q6_K m8 aligned",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
        8,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q8_0 m16 aligned",
        qwen35::weights::LOWBIT_GGML_Q8_0,
        ggml_q8_0_row,
        16,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q4_K m16 aligned",
        qwen35::weights::LOWBIT_GGML_Q4_K,
        ggml_q4k_row,
        16,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q5_K m16 aligned",
        qwen35::weights::LOWBIT_GGML_Q5_K,
        ggml_q5k_row,
        16,
        32,
        256,
    )?;
    run_ggml_case_shape(
        ordinal,
        "GGML Q6_K m16 aligned",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
        16,
        32,
        256,
    )?;
    run_ggml_residual_add_case(
        ordinal,
        "GGML Q4_K m16",
        qwen35::weights::LOWBIT_GGML_Q4_K,
        ggml_q4k_row,
    )?;
    run_ggml_residual_add_case(
        ordinal,
        "GGML Q5_K m16",
        qwen35::weights::LOWBIT_GGML_Q5_K,
        ggml_q5k_row,
    )?;
    run_ggml_residual_add_case(
        ordinal,
        "GGML Q6_K m16",
        qwen35::weights::LOWBIT_GGML_Q6_K,
        ggml_q6k_row,
    )?;
    run_mmq_q8_quant_case(ordinal, "GGML Q4_K", qwen35::weights::LOWBIT_GGML_Q4_K)?;
    run_mmq_q8_quant_case(ordinal, "GGML Q5_K", qwen35::weights::LOWBIT_GGML_Q5_K)?;
    run_mmq_q8_quant_case(ordinal, "GGML Q6_K", qwen35::weights::LOWBIT_GGML_Q6_K)?;
    run_mmq_q6_matmul_case(ordinal, "GGML Q6_K MMQ m8", 8, 128, 256)?;
    run_mmq_q6_matmul_case(ordinal, "GGML Q6_K MMQ m16", 16, 128, 512)?;
    run_q6_k_m16_argmax_case(ordinal)?;
    maybe_run_ggml_hot_benches(ordinal)?;

    Ok(())
}
