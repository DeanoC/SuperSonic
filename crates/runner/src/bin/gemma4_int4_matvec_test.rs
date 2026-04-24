//! Known-value correctness test for Gemma 4's INT4 matvec primitives.
//!
//! Builds a synthetic (packed, scale, zero) INT4 weight matrix plus a BF16
//! activation, computes the GPU result via `g4::matvec_int4` /
//! `g4::matvec_batched_int4`, and compares against a CPU reference that
//! reproduces the kernel's `bf16(nibble*s - zero*s)` dequant + F32 accumulate
//! + BF16 output rounding exactly.
//!
//! Shapes exercised:
//!   - (in=1536, out=2048) — E2B SWA q_proj
//!   - (in=1536, out=512)  — E2B FULL k/v_proj
//!   - (in=1536, out=12288) — E2B double-wide MLP gate/up
//!   - (in=12288, out=1536) — E2B double-wide MLP down
//!   - (in=2560, out=4096) — E4B FULL q_proj
//! Plus a batched version across all shapes with seq_len=3.

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::gemma4 as g4;

const GROUP_SIZE: usize = 128;

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
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

/// Mirror the kernel's `g4_int4_dequant_scalar` exactly.
fn dequant_scalar(nibble: u8, scale_bf16: f32, zero_bf16: f32) -> f32 {
    bf16_round(nibble as f32 * scale_bf16 - zero_bf16 * scale_bf16)
}

/// Deterministic pseudo-random byte sequence keyed by (row, col). Used to
/// fill nibbles so each shape gets a reproducible pattern without pulling in
/// `rand`.
fn nibble_at(row: usize, col: usize) -> u8 {
    let mut x = row.wrapping_mul(0x9E3779B1) ^ col.wrapping_mul(0xBF58476D);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB352D);
    x ^= x >> 15;
    (x & 0xF) as u8
}

/// Pack `[out_dim, in_dim]` nibbles into `[out_dim, in_dim/2]` bytes,
/// low nibble = even col, high nibble = odd col. Matches both the Qwen and
/// Gemma 4 bake formats.
fn pack_nibbles(nibbles: &[u8], out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % 2, 0);
    let mut out = Vec::with_capacity(out_dim * in_dim / 2);
    for r in 0..out_dim {
        for c in (0..in_dim).step_by(2) {
            let lo = nibbles[r * in_dim + c] & 0xF;
            let hi = nibbles[r * in_dim + c + 1] & 0xF;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Build synthetic scale[out_dim/gs, in_dim/gs] and zero[out_dim/gs, in_dim/gs]
/// in BF16-rounded F32. Scale in (0.004, 0.02), zero in [0, 15] — ranges that
/// resemble real GPTQ output on projection weights.
fn build_scales(out_dim: usize, in_dim: usize, gs: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(out_dim % gs, 0);
    assert_eq!(in_dim % gs, 0);
    let rows = out_dim / gs;
    let cols = in_dim / gs;
    let n = rows * cols;
    let mut scale = Vec::with_capacity(n);
    let mut zero = Vec::with_capacity(n);
    for i in 0..n {
        // Simple deterministic sequence, then round through BF16 so the host
        // and kernel see identical values.
        let t = (i as f32) / (n.max(1) as f32);
        let s = 0.004 + 0.016 * (0.5 + 0.5 * (t * 7.1).sin());
        let z = 7.5 + 7.0 * (t * 3.3).cos();
        scale.push(bf16_round(s));
        zero.push(bf16_round(z));
    }
    (scale, zero)
}

/// Build a BF16-rounded activation `[in_dim]` using a deterministic seed.
fn build_activation(in_dim: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(in_dim);
    for i in 0..in_dim {
        let x = ((i as u32).wrapping_mul(0x5851F42D).wrapping_add(seed)) as f32;
        let v = ((x * 1e-7).sin() * 0.5) + ((i as f32) * 0.0001);
        out.push(bf16_round(v));
    }
    out
}

fn reference_matvec(
    in_dim: usize,
    out_dim: usize,
    gs: usize,
    x: &[f32],
    nibbles: &[u8],
    scales: &[f32],
    zeros: &[f32],
) -> Vec<f32> {
    let scale_cols = in_dim / gs;
    let mut out = vec![0f32; out_dim];
    for r in 0..out_dim {
        let mut acc = 0f32;
        let sr = r / gs;
        for c in 0..in_dim {
            let sc = c / gs;
            let si = sr * scale_cols + sc;
            let w = dequant_scalar(nibbles[r * in_dim + c], scales[si], zeros[si]);
            acc += w * x[c];
        }
        out[r] = bf16_round(acc);
    }
    out
}

struct Shape {
    in_dim: usize,
    out_dim: usize,
    label: &'static str,
}

fn test_single(dev: usize, shape: &Shape) -> Result<()> {
    let in_dim = shape.in_dim;
    let out_dim = shape.out_dim;
    let gs = GROUP_SIZE;

    // Build weights.
    let mut nibbles = vec![0u8; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            nibbles[r * in_dim + c] = nibble_at(r, c);
        }
    }
    let (scales, zeros) = build_scales(out_dim, in_dim, gs);
    let x_f32 = build_activation(in_dim, 0xC0FFEE);

    // Reference (CPU).
    let expected = reference_matvec(in_dim, out_dim, gs, &x_f32, &nibbles, &scales, &zeros);

    // Upload to GPU.
    let packed_bytes = pack_nibbles(&nibbles, out_dim, in_dim);
    let scale_bytes = f32_to_bf16_bytes(&scales);
    let zero_bytes = f32_to_bf16_bytes(&zeros);
    let x_bytes = f32_to_bf16_bytes(&x_f32);

    let w_packed =
        GpuBuffer::from_host_bytes(dev, ScalarType::U8, &[out_dim, in_dim / 2], &packed_bytes)?;
    let w_scale = GpuBuffer::from_host_bytes(
        dev,
        ScalarType::BF16,
        &[out_dim / gs, in_dim / gs],
        &scale_bytes,
    )?;
    let w_zero = GpuBuffer::from_host_bytes(
        dev,
        ScalarType::BF16,
        &[out_dim / gs, in_dim / gs],
        &zero_bytes,
    )?;
    let x = GpuBuffer::from_host_bytes(dev, ScalarType::BF16, &[in_dim], &x_bytes)?;
    let mut out_buf = GpuBuffer::zeros(dev, ScalarType::BF16, &[out_dim])?;
    let mut counter = GpuBuffer::zeros(dev, ScalarType::U32, &[1])?;

    g4::matvec_int4(
        dev,
        ScalarType::BF16,
        &mut out_buf,
        &x,
        &w_packed,
        &w_scale,
        &w_zero,
        in_dim,
        out_dim,
        gs,
        &mut counter,
    )?;

    let got = bf16_bytes_to_f32(&out_buf.to_host_bytes()?);
    let mut mismatches = 0usize;
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for i in 0..out_dim {
        let d = (got[i] - expected[i]).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
        // BF16 output has ~1-2 ULP of rounding variance between kernel's
        // distributed F32 reduction and the CPU reference's serial accumulate.
        // On large-out_dim low-magnitude E4B shapes (e.g. SWA o_proj at
        // in=2048 out=2560) a single value can round 1 ULP away from the
        // reference; a relative tolerance of 1% absorbs that without hiding
        // real kernel bugs (which produce huge deltas — we saw 16-20 on
        // Qwen's INT4 kernel bug; see project_int4_kernel_bug).
        let tol = (expected[i].abs() * 0.02).max(0.15);
        if d > tol {
            mismatches += 1;
        }
    }
    eprintln!(
        "[{}] in={} out={} max_abs_delta={:.4} (row {}, got={:.4} expected={:.4}) mismatches(>tol)={}",
        shape.label, in_dim, out_dim, worst, worst_at, got[worst_at], expected[worst_at], mismatches,
    );
    if mismatches > 0 {
        bail!("INT4 matvec mismatch on {}", shape.label);
    }
    Ok(())
}

fn test_batched(dev: usize, shape: &Shape, seq_len: usize) -> Result<()> {
    let in_dim = shape.in_dim;
    let out_dim = shape.out_dim;
    let gs = GROUP_SIZE;

    let mut nibbles = vec![0u8; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            nibbles[r * in_dim + c] = nibble_at(r, c);
        }
    }
    let (scales, zeros) = build_scales(out_dim, in_dim, gs);

    // Per-sequence activations + references.
    let mut x_flat = Vec::with_capacity(seq_len * in_dim);
    let mut expected_flat = Vec::with_capacity(seq_len * out_dim);
    for s in 0..seq_len {
        let xs = build_activation(in_dim, 0xCAFE ^ (s as u32 * 17));
        let refs = reference_matvec(in_dim, out_dim, gs, &xs, &nibbles, &scales, &zeros);
        x_flat.extend_from_slice(&xs);
        expected_flat.extend_from_slice(&refs);
    }

    let packed_bytes = pack_nibbles(&nibbles, out_dim, in_dim);
    let scale_bytes = f32_to_bf16_bytes(&scales);
    let zero_bytes = f32_to_bf16_bytes(&zeros);
    let x_bytes = f32_to_bf16_bytes(&x_flat);

    let w_packed =
        GpuBuffer::from_host_bytes(dev, ScalarType::U8, &[out_dim, in_dim / 2], &packed_bytes)?;
    let w_scale = GpuBuffer::from_host_bytes(
        dev,
        ScalarType::BF16,
        &[out_dim / gs, in_dim / gs],
        &scale_bytes,
    )?;
    let w_zero = GpuBuffer::from_host_bytes(
        dev,
        ScalarType::BF16,
        &[out_dim / gs, in_dim / gs],
        &zero_bytes,
    )?;
    let x = GpuBuffer::from_host_bytes(dev, ScalarType::BF16, &[seq_len, in_dim], &x_bytes)?;
    let mut out_buf = GpuBuffer::zeros(dev, ScalarType::BF16, &[seq_len, out_dim])?;
    let mut counter = GpuBuffer::zeros(dev, ScalarType::U32, &[1])?;

    g4::matvec_batched_int4(
        dev,
        ScalarType::BF16,
        &mut out_buf,
        &x,
        &w_packed,
        &w_scale,
        &w_zero,
        seq_len,
        in_dim,
        out_dim,
        gs,
        &mut counter,
    )?;

    let got = bf16_bytes_to_f32(&out_buf.to_host_bytes()?);
    let mut mismatches = 0usize;
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for i in 0..(seq_len * out_dim) {
        let d = (got[i] - expected_flat[i]).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
        let tol = (expected_flat[i].abs() * 0.02).max(0.15);
        if d > tol {
            mismatches += 1;
        }
    }
    eprintln!(
        "[{} batched seq_len={}] in={} out={} max_abs_delta={:.4} (idx {}, got={:.4} expected={:.4}) mismatches(>tol)={}",
        shape.label, seq_len, in_dim, out_dim, worst, worst_at,
        got[worst_at], expected_flat[worst_at], mismatches,
    );
    if mismatches > 0 {
        bail!("INT4 batched matvec mismatch on {}", shape.label);
    }
    Ok(())
}

fn main() -> Result<()> {
    let dev = 0usize;
    gpu_hal::set_device(dev).map_err(|e| anyhow!("set_device: {e}"))?;

    let shapes = [
        Shape {
            in_dim: 1536,
            out_dim: 2048,
            label: "E2B-SWA-q_proj",
        },
        Shape {
            in_dim: 1536,
            out_dim: 512,
            label: "E2B-FULL-k/v_proj",
        },
        Shape {
            in_dim: 1536,
            out_dim: 12288,
            label: "E2B-wide-gate/up",
        },
        Shape {
            in_dim: 12288,
            out_dim: 1536,
            label: "E2B-wide-down",
        },
        Shape {
            in_dim: 2560,
            out_dim: 4096,
            label: "E4B-FULL-q_proj",
        },
        // E4B-specific shapes (added 2026-04-19 during INT4 drift investigation).
        // in_dim=2560 is critical: it's not a multiple of 2048 (block_stride),
        // so it exercises the multi-round non-aligned path which earlier E2B
        // tests never hit (E2B has in_dim=1536 which fits in one round).
        Shape {
            in_dim: 2560,
            out_dim: 2048,
            label: "E4B-SWA-q_proj",
        },
        Shape {
            in_dim: 2560,
            out_dim: 1024,
            label: "E4B-FULL-k/v_proj",
        },
        Shape {
            in_dim: 2560,
            out_dim: 512,
            label: "E4B-SWA-k/v_proj",
        },
        Shape {
            in_dim: 2560,
            out_dim: 10240,
            label: "E4B-gate/up",
        },
        Shape {
            in_dim: 10240,
            out_dim: 2560,
            label: "E4B-down",
        },
        Shape {
            in_dim: 4096,
            out_dim: 2560,
            label: "E4B-FULL-o_proj",
        },
        Shape {
            in_dim: 2048,
            out_dim: 2560,
            label: "E4B-SWA-o_proj",
        },
    ];

    for shape in &shapes {
        test_single(dev, shape)?;
    }
    for shape in &shapes {
        test_batched(dev, shape, 3)?;
    }

    eprintln!(
        "[PASS] all {} single + {} batched INT4 matvec tests",
        shapes.len(),
        shapes.len()
    );
    Ok(())
}
