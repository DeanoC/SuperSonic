use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use half::{bf16, f16};
use kernel_ffi::prefill_ffi;

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

fn f32_to_f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn f16_round(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

fn build_lhs(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let x = ((r as u32).wrapping_mul(0x9E37) ^ (c as u32).wrapping_mul(0x85EB) ^ seed) as f32;
            let v = ((x * 1e-5).sin() * 0.75) + ((c as f32) * 1e-4) - ((r as f32) * 2e-4);
            out.push(bf16_round(v));
        }
    }
    out
}

fn build_rhs(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_dim * in_dim);
    for r in 0..out_dim {
        for c in 0..in_dim {
            let x = r.wrapping_mul(0x517C_C1B7) ^ c.wrapping_mul(0x68BC_21EB);
            let q = ((x ^ (x >> 13) ^ (x >> 7)) % 255) as i32 - 127;
            out.push((q as i8) as u8);
        }
    }
    out
}

fn build_scb(out_dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let t = i as f32 / out_dim.max(1) as f32;
        out.push(0.25 + 1.75 * (0.5 + 0.5 * (t * 5.3).sin()));
    }
    out
}

fn quantize_lhs_bnb(lhs_bf16: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];
    for r in 0..rows {
        let row = &lhs_bf16[r * cols..(r + 1) * cols];
        let row_f16: Vec<f32> = row.iter().map(|&v| f16_round(v)).collect();
        let absmax = row_f16
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));
        scales[r] = absmax;
        if absmax == 0.0 {
            continue;
        }
        let inv = 127.0 / absmax;
        for c in 0..cols {
            let rounded = (row_f16[c] * inv + row_f16[c].signum() * 0.49999)
                .trunc()
                .clamp(-127.0, 127.0);
            q[r * cols + c] = rounded as i8;
        }
    }
    (q, scales)
}

fn reference_matmul(lhs_bf16: &[f32], rhs_i8_bytes: &[u8], scb: &[f32], rows: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let (lhs_q, lhs_scales) = quantize_lhs_bnb(lhs_bf16, rows, in_dim);
    let mut out = vec![0f32; rows * out_dim];
    let inv_127_sq = 1.0 / (127.0 * 127.0);
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0i32;
            for c in 0..in_dim {
                let a = lhs_q[r * in_dim + c] as i32;
                let b = rhs_i8_bytes[o * in_dim + c] as i8 as i32;
                acc += a * b;
            }
            let val = (acc as f32) * lhs_scales[r] * scb[o] * inv_127_sq;
            out[r * out_dim + o] = bf16_round(f16_round(val));
        }
    }
    out
}

struct Shape {
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    label: &'static str,
}

fn run_shape(dev: usize, shape: &Shape) -> Result<()> {
    let lhs = build_lhs(shape.rows, shape.in_dim, 0xC0FFEE);
    let rhs = build_rhs(shape.out_dim, shape.in_dim);
    let scb = build_scb(shape.out_dim);
    let expected = reference_matmul(&lhs, &rhs, &scb, shape.rows, shape.out_dim, shape.in_dim);

    let lhs_gpu = GpuBuffer::from_host_bytes(dev, ScalarType::BF16, &[shape.rows, shape.in_dim], &f32_to_bf16_bytes(&lhs))?;
    let rhs_gpu = GpuBuffer::from_host_bytes(dev, ScalarType::U8, &[shape.out_dim, shape.in_dim], &rhs)?;
    let scb_gpu = GpuBuffer::from_host_bytes(dev, ScalarType::F32, &[shape.out_dim], &f32_to_f32_bytes(&scb))?;
    let mut out_gpu = GpuBuffer::zeros(dev, ScalarType::BF16, &[shape.rows, shape.out_dim])?;

    prefill_ffi::matmul_rhs_transposed_int8(
        dev,
        1,
        shape.rows,
        shape.out_dim,
        shape.in_dim,
        &lhs_gpu,
        &rhs_gpu,
        &scb_gpu,
        &mut out_gpu,
    )?;

    let got = bf16_bytes_to_f32(&out_gpu.to_host_bytes()?);
    let mut mismatches = 0usize;
    let mut worst = 0.0f32;
    let mut worst_idx = 0usize;
    for i in 0..got.len() {
        let delta = (got[i] - expected[i]).abs();
        if delta > worst {
            worst = delta;
            worst_idx = i;
        }
        if delta != 0.0 {
            mismatches += 1;
        }
    }
    eprintln!(
        "[{}] rows={} in={} out={} max_abs_delta={:.6} idx={} got={:.6} expected={:.6} mismatches={}",
        shape.label,
        shape.rows,
        shape.in_dim,
        shape.out_dim,
        worst,
        worst_idx,
        got[worst_idx],
        expected[worst_idx],
        mismatches,
    );
    if mismatches > 0 {
        bail!("INT8 matmul mismatch on {}", shape.label);
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = 0usize;
    gpu_hal::set_device(dev).map_err(|e| anyhow!("set_device: {e}"))?;

    let shapes = [
        Shape { rows: 1, in_dim: 4096, out_dim: 4096, label: "llama-qproj-step" },
        Shape { rows: 1, in_dim: 4096, out_dim: 1024, label: "llama-kv-step" },
        Shape { rows: 3, in_dim: 4096, out_dim: 4096, label: "llama-qproj-prefill" },
        Shape { rows: 3, in_dim: 4096, out_dim: 14336, label: "llama-mlp-prefill" },
    ];

    for shape in &shapes {
        run_shape(dev, shape)?;
    }

    eprintln!("[PASS] llama31 INT8 matmul primitive matches CPU reference");
    Ok(())
}
