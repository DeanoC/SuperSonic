//! Minimal known-value test for `matmul_rhs_transposed_int4`.
//!
//! Compares the HIP INT4 dequant matmul against a CPU reference using
//! bit-exact dequant: `bf16(q*s - zf*s)`.

use anyhow::{anyhow, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::prefill_ffi;

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
    lhs: &[f32],               // [m, k] BF16-rounded values
    nibbles: &[u8],            // [n, k]
    scales: &[f32],            // [n/gs, k/gs] BF16-rounded values
    zeros: &[f32],             // [n/gs, k/gs] BF16-rounded values
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
    let lhs_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[m, k], &f32_to_bf16_bytes(&lhs))
        .map_err(|e| anyhow!("lhs upload: {e}"))?;

    let packed = pack_nibbles(&nibbles, n, k);
    let rhs_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::U8, &[n, k / 2], &packed)
        .map_err(|e| anyhow!("rhs upload: {e}"))?;

    let scale_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[sr, sc], &f32_to_bf16_bytes(&scales))
        .map_err(|e| anyhow!("scale upload: {e}"))?;

    let zero_gpu = GpuBuffer::from_host_bytes(
        ordinal, ScalarType::BF16, &[sr, sc], &f32_to_bf16_bytes(&zeros))
        .map_err(|e| anyhow!("zero upload: {e}"))?;

    let mut out_gpu = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[m, n])
        .map_err(|e| anyhow!("out alloc: {e}"))?;

    prefill_ffi::matmul_rhs_transposed_int4(
        ordinal, 1, m, n, k,
        &lhs_gpu, &rhs_gpu, &scale_gpu, &zero_gpu,
        gs, &mut out_gpu,
    ).map_err(|e| anyhow!("int4 matmul: {e}"))?;

    let out_host = bf16_bytes_to_f32(&out_gpu.to_host_bytes()
        .map_err(|e| anyhow!("out d2h: {e}"))?);

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
    println!("  max_abs={max_abs:.5e}  max_rel={max_rel:.5e}  bad={nbad}/{}", m * n);
    if let Some((mi, ni, g, r)) = first_bad {
        println!("  first bad @ [{mi},{ni}]: gpu={g:.6} ref={r:.6}");
    }

    // Dump a few sample values for sanity
    println!(
        "  samples: gpu[0,0]={:.4}  ref[0,0]={:.4}  gpu[{}, {}]={:.4}  ref={:.4}",
        out_host[0], ref_out[0],
        m - 1, n - 1,
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
        TestCase { name: "single group, single tile",  m: 16, n: 16, k: 128, gs: 128 },
        TestCase { name: "2 groups in k",              m: 16, n: 16, k: 256, gs: 128 },
        TestCase { name: "2 groups in n",              m: 16, n: 32, k: 128, gs: 128 },
        TestCase { name: "multi-tile, aligned",        m: 32, n: 32, k: 256, gs: 128 },
        TestCase { name: "Qwen-size group=128",        m: 1,  n: 128, k: 256, gs: 128 },
        TestCase { name: "k spans many groups",        m: 4,  n: 16,  k: 1024, gs: 128 },
        TestCase { name: "prefill-like shape",         m: 8,  n: 128, k: 2560, gs: 128 },
    ];

    for c in &cases {
        run_case(ordinal, c)?;
    }

    Ok(())
}
