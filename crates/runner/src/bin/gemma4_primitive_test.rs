//! Numerical correctness tests for the Gemma 4 decode primitives.
//!
//! Each primitive is fired against a tiny synthetic input with a hand-computed
//! reference value, so we can catch kernel bugs (wrong axis indexing, miscounted
//! reductions, missed mask branches) without needing to wire up safetensors
//! loading or the full single-layer forward pass. Runs on any GPU — no
//! Gemma-4-specific data needed.
//!
//! Usage:
//!   cargo run --release --bin gemma4_primitive_test
//!
//! Exits with status 0 on success and prints per-primitive pass/fail.

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use half::bf16;
use kernel_ffi::gemma4;

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

fn upload_bf16(shape: &[usize], host: &[f32]) -> Result<GpuBuffer> {
    let bytes = f32_to_bf16_bytes(host);
    Ok(GpuBuffer::from_host_bytes(0, ScalarType::BF16, shape, &bytes)?)
}

fn download_bf16(buf: &GpuBuffer) -> Result<Vec<f32>> {
    Ok(bf16_bytes_to_f32(&buf.to_host_bytes()?))
}

/// Relative-error check with a fixed absolute tolerance floor (to accommodate
/// tiny values). BF16 has ~3 decimal digits of precision, so 1% rel + 1e-3
/// abs is generous for a single-op check but tight enough to catch bugs.
fn nearly_eq(a: f32, b: f32, rtol: f32, atol: f32) -> bool {
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1.0);
    diff <= atol || diff / denom <= rtol
}

fn assert_vec(label: &str, got: &[f32], want: &[f32], rtol: f32, atol: f32) -> Result<()> {
    if got.len() != want.len() {
        bail!("{label}: length mismatch got={} want={}", got.len(), want.len());
    }
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if !nearly_eq(g, w, rtol, atol) {
            bail!(
                "{label}: element [{i}] got={g:.6} want={w:.6} (rtol={rtol} atol={atol})"
            );
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Test 1: RMSNorm (Gemma variant — no `w+1` offset)
// -----------------------------------------------------------------------------

fn test_rms_norm() -> Result<()> {
    // Input: 8 values; weight: simple ramp. Reference computed in F32 on host.
    let x: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let w: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let eps = 1e-6_f32;
    let n = x.len();

    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = (mean_sq + eps).sqrt().recip();
    let want: Vec<f32> = x
        .iter()
        .zip(w.iter())
        .map(|(&xv, &wv)| (xv * inv_rms) * wv)
        .collect();

    let xs = upload_bf16(&[n], &x)?;
    let weight = upload_bf16(&[n], &w)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[n])?;
    gemma4::rms_norm(0, ScalarType::BF16, &mut out, &xs, Some(&weight), eps, n)?;

    let got = download_bf16(&out)?;
    // BF16 has ~3 digits of precision; allow 2% relative error.
    assert_vec("rms_norm", &got, &want, 2e-2, 5e-3)
}

fn test_rms_norm_no_weight() -> Result<()> {
    // with_scale=False path: null weight ⇒ output is pure normalization.
    let x: Vec<f32> = vec![1.0, 2.0, -3.0, -4.0];
    let eps = 1e-6_f32;
    let n = x.len();
    let inv_rms = ((x.iter().map(|v| v * v).sum::<f32>() / n as f32) + eps).sqrt().recip();
    let want: Vec<f32> = x.iter().map(|&xv| xv * inv_rms).collect();

    let xs = upload_bf16(&[n], &x)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[n])?;
    gemma4::rms_norm(0, ScalarType::BF16, &mut out, &xs, None, eps, n)?;

    let got = download_bf16(&out)?;
    assert_vec("rms_norm_no_weight", &got, &want, 2e-2, 5e-3)
}

// -----------------------------------------------------------------------------
// Test 2: matvec — W [out, in] @ x[in]
// -----------------------------------------------------------------------------

fn test_matvec() -> Result<()> {
    let in_dim = 4;
    let out_dim = 3;
    let x: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5];
    // W[out, in] row-major
    let w: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // row 0 = x[0]
        0.0, 2.0, 0.0, 0.0, // row 1 = 2 * x[1]
        1.0, 1.0, 1.0, 1.0, // row 2 = sum(x)
    ];
    let want: Vec<f32> = vec![1.0, 4.0, 1.0 + 2.0 - 1.0 + 0.5];

    let xs = upload_bf16(&[in_dim], &x)?;
    let ws = upload_bf16(&[out_dim, in_dim], &w)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[out_dim])?;
    let mut counter = GpuBuffer::zeros(0, ScalarType::U32, &[1])?;
    gemma4::matvec(0, ScalarType::BF16, &mut out, &xs, &ws, in_dim, out_dim, &mut counter)?;
    let got = download_bf16(&out)?;
    assert_vec("matvec", &got, &want, 2e-2, 5e-3)
}

// -----------------------------------------------------------------------------
// Test 3: GeLU-tanh gated multiply
// -----------------------------------------------------------------------------

fn gelu_tanh_ref(x: f32) -> f32 {
    let k_sqrt_2_over_pi = 0.7978845608028654_f32;
    let coef = 0.044715_f32;
    let inner = k_sqrt_2_over_pi * (x + coef * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn test_gelu_tanh_gate_mul() -> Result<()> {
    let gate = vec![0.0_f32, 1.0, -1.0, 2.5, -3.0];
    let up = vec![1.0_f32, 2.0, 3.0, 4.0, 0.5];
    let want: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| gelu_tanh_ref(g) * u)
        .collect();

    let gate_b = upload_bf16(&[gate.len()], &gate)?;
    let up_b = upload_bf16(&[up.len()], &up)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[gate.len()])?;
    gemma4::gelu_tanh_gate_mul(0, ScalarType::BF16, &mut out, &gate_b, &up_b, gate.len())?;
    let got = download_bf16(&out)?;
    assert_vec("gelu_tanh_gate_mul", &got, &want, 2e-2, 5e-3)
}

// -----------------------------------------------------------------------------
// Test 4: RoPE (split-half, Gemma style)
// -----------------------------------------------------------------------------

fn test_rope_decode() -> Result<()> {
    // 2 heads, head_dim=4 (so half=2), rotary_dim=4 (full rotation).
    // Position 1, with cos/sin = cos(1 * inv_freq), sin(...) for inv_freq[0..2].
    // We synthesize cos/sin tables directly.
    let num_heads = 2;
    let head_dim = 4;
    let rotary_dim = 4;
    let half = rotary_dim / 2;
    let max_pos = 4;
    let position = 1;

    let c0 = 0.8_f32;
    let s0 = 0.6_f32; // c0^2 + s0^2 == 1
    let c1 = 0.5_f32;
    let s1 = (1.0_f32 - 0.25).sqrt(); // sqrt(0.75) ≈ 0.866

    // cos/sin tables shape [max_pos, rotary_dim=half*2]; kernel reads
    // cos_table[position * rotary_dim + i] for i in [0, half).
    let mut cos_host = vec![0.0_f32; max_pos * rotary_dim];
    let mut sin_host = vec![0.0_f32; max_pos * rotary_dim];
    let row_start = position * rotary_dim;
    cos_host[row_start] = c0;
    cos_host[row_start + 1] = c1;
    cos_host[row_start + half] = c0; // Gemma's split-half pattern has cos repeated; we only read first half.
    cos_host[row_start + half + 1] = c1;
    sin_host[row_start] = s0;
    sin_host[row_start + 1] = s1;
    sin_host[row_start + half] = s0;
    sin_host[row_start + half + 1] = s1;

    // Head 0: [1, 0, 0, 0] — first element only
    // Head 1: [0, 1, 0, 0]
    let x_host = vec![
        1.0_f32, 0.0, 0.0, 0.0, // head 0
        0.0, 1.0, 0.0, 0.0, // head 1
    ];

    // Apply rotation by hand:
    // y[i] = x[i]*c - x[i+half]*s          (i in [0, half))
    // y[i+half] = x[i+half]*c + x[i]*s
    let mut want = x_host.clone();
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let (c, s) = if i == 0 { (c0, s0) } else { (c1, s1) };
            let x0 = x_host[base + i];
            let x1 = x_host[base + i + half];
            want[base + i] = x0 * c - x1 * s;
            want[base + i + half] = x1 * c + x0 * s;
        }
    }

    let cos_b = upload_bf16(&[max_pos, rotary_dim], &cos_host)?;
    let sin_b = upload_bf16(&[max_pos, rotary_dim], &sin_host)?;
    let mut x_b = upload_bf16(&[num_heads, head_dim], &x_host)?;

    gemma4::rope_decode(
        0, ScalarType::BF16, &mut x_b, &cos_b, &sin_b,
        num_heads, head_dim, rotary_dim, position,
    )?;

    let got = download_bf16(&x_b)?;
    assert_vec("rope_decode", &got, &want, 2e-2, 5e-3)
}

// -----------------------------------------------------------------------------
// Test 5: KV append — copy one token's K/V into a cache slot
// -----------------------------------------------------------------------------

fn test_kv_append() -> Result<()> {
    let num_kv = 2;
    let head_dim = 3;
    let max_t = 4;
    let pos = 2;

    let k_in: Vec<f32> = vec![
        1.0, 2.0, 3.0, // head 0
        4.0, 5.0, 6.0, // head 1
    ];
    let v_in: Vec<f32> = vec![
        -1.0, -2.0, -3.0, // head 0
        -4.0, -5.0, -6.0, // head 1
    ];

    let k_in_b = upload_bf16(&[num_kv, head_dim], &k_in)?;
    let v_in_b = upload_bf16(&[num_kv, head_dim], &v_in)?;
    let mut k_cache = GpuBuffer::zeros(0, ScalarType::BF16, &[num_kv, max_t, head_dim])?;
    let mut v_cache = GpuBuffer::zeros(0, ScalarType::BF16, &[num_kv, max_t, head_dim])?;

    gemma4::kv_append(
        0, ScalarType::BF16, &k_in_b, &v_in_b, &mut k_cache, &mut v_cache,
        num_kv, head_dim, pos, max_t,
    )?;

    let k_host = download_bf16(&k_cache)?;
    let v_host = download_bf16(&v_cache)?;

    for h in 0..num_kv {
        for t in 0..max_t {
            for d in 0..head_dim {
                let off = (h * max_t + t) * head_dim + d;
                let expected_k = if t == pos { k_in[h * head_dim + d] } else { 0.0 };
                let expected_v = if t == pos { v_in[h * head_dim + d] } else { 0.0 };
                if !nearly_eq(k_host[off], expected_k, 2e-2, 5e-3) {
                    bail!(
                        "kv_append K mismatch at h={h} t={t} d={d}: got {} want {}",
                        k_host[off], expected_k
                    );
                }
                if !nearly_eq(v_host[off], expected_v, 2e-2, 5e-3) {
                    bail!(
                        "kv_append V mismatch at h={h} t={t} d={d}: got {} want {}",
                        v_host[off], expected_v
                    );
                }
            }
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Test 6: SWA attention — tiny known case (2 tokens cached, uniform softmax)
// -----------------------------------------------------------------------------

fn test_swa_attn_uniform() -> Result<()> {
    // Single q head, single kv head, head_dim = 4, kv_len = 2, sliding_window large.
    // Q = zero vector ⇒ scores both zero ⇒ softmax is uniform [0.5, 0.5] ⇒
    // output = 0.5 * (V[0] + V[1]).
    let num_q = 1;
    let num_kv = 1;
    let hd = 4;
    let kv_len = 2;
    let max_t = 4;

    let q: Vec<f32> = vec![0.0; num_q * hd];
    let k: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,  // t=0
        5.0, 6.0, 7.0, 8.0,  // t=1
        0.0, 0.0, 0.0, 0.0,  // t=2 (unused)
        0.0, 0.0, 0.0, 0.0,  // t=3
    ];
    let v: Vec<f32> = vec![
        2.0, 4.0, 6.0, 8.0,
        10.0, 12.0, 14.0, 16.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let want: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0]; // avg of V[0] and V[1]

    let q_b = upload_bf16(&[num_q, hd], &q)?;
    let k_b = upload_bf16(&[num_kv, max_t, hd], &k)?;
    let v_b = upload_bf16(&[num_kv, max_t, hd], &v)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[num_q, hd])?;
    let mut scores = GpuBuffer::zeros(0, ScalarType::F32, &[num_q, max_t])?;

    gemma4::swa_attn_decode(
        0, ScalarType::BF16, &q_b, &k_b, &v_b, &mut scores, &mut out,
        num_q, num_kv, hd, kv_len, max_t, 0 /* no window */, 1.0,
    )?;

    let got = download_bf16(&out)?;
    assert_vec("swa_attn_uniform", &got, &want, 2e-2, 5e-3)
}

fn test_swa_attn_window_masks_oldest() -> Result<()> {
    // 3 tokens cached, sliding_window=2. Window should mask out t=0 (-inf),
    // so softmax sees only t=1, t=2. Q is zero so softmax on [t=1, t=2] is
    // uniform and output = 0.5 * (V[1] + V[2]).
    let num_q = 1;
    let num_kv = 1;
    let hd = 2;
    let kv_len = 3;
    let max_t = 4;

    let q: Vec<f32> = vec![0.0; hd];
    let k: Vec<f32> = vec![
        100.0, 100.0,  // t=0 — would dominate if not masked
        1.0, 1.0,
        2.0, 2.0,
        0.0, 0.0,
    ];
    let v: Vec<f32> = vec![
        10.0, 20.0,
        3.0, 5.0,
        5.0, 9.0,
        0.0, 0.0,
    ];
    // Expected: avg(V[1], V[2]) = (4.0, 7.0)
    let want: Vec<f32> = vec![4.0, 7.0];

    let q_b = upload_bf16(&[num_q, hd], &q)?;
    let k_b = upload_bf16(&[num_kv, max_t, hd], &k)?;
    let v_b = upload_bf16(&[num_kv, max_t, hd], &v)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[num_q, hd])?;
    let mut scores = GpuBuffer::zeros(0, ScalarType::F32, &[num_q, max_t])?;

    gemma4::swa_attn_decode(
        0, ScalarType::BF16, &q_b, &k_b, &v_b, &mut scores, &mut out,
        num_q, num_kv, hd, kv_len, max_t, 2 /* window=2 */, 1.0,
    )?;

    let got = download_bf16(&out)?;
    assert_vec("swa_attn_window_masks_oldest", &got, &want, 2e-2, 5e-3)
}

fn test_swa_attn_gqa() -> Result<()> {
    // 2 Q heads, 1 KV head (ratio 2:1), head_dim = 2. Q head 0 = [1, 0]
    // and Q head 1 = [0, 1] both attend to the single KV head's cache.
    // kv_len = 2 with large scores dominated by K[1] at position 1.
    let num_q = 2;
    let num_kv = 1;
    let hd = 2;
    let kv_len = 2;
    let max_t = 2;

    // Pick K so Q_h0 dots strongly with K[1] and Q_h1 also dots strongly with K[1].
    // K[0] = [0.1, 0.1]; K[1] = [5.0, 5.0]. Q_h0 = [1, 0] → scores [0.1, 5.0].
    // Q_h1 = [0, 1] → scores [0.1, 5.0]. Both softmax heavily onto t=1 ⇒ out≈V[1].
    let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let k: Vec<f32> = vec![0.1, 0.1, 5.0, 5.0];
    let v: Vec<f32> = vec![0.0, 0.0, 7.0, 9.0]; // V[1] = [7.0, 9.0]
    let want: Vec<f32> = vec![7.0, 9.0, 7.0, 9.0];

    let q_b = upload_bf16(&[num_q, hd], &q)?;
    let k_b = upload_bf16(&[num_kv, max_t, hd], &k)?;
    let v_b = upload_bf16(&[num_kv, max_t, hd], &v)?;
    let mut out = GpuBuffer::zeros(0, ScalarType::BF16, &[num_q, hd])?;
    let mut scores = GpuBuffer::zeros(0, ScalarType::F32, &[num_q, max_t])?;

    gemma4::swa_attn_decode(
        0, ScalarType::BF16, &q_b, &k_b, &v_b, &mut scores, &mut out,
        num_q, num_kv, hd, kv_len, max_t, 0, 1.0,
    )?;

    let got = download_bf16(&out)?;
    // Softmax(5.0 vs 0.1) is not perfectly 1.0; tolerance loose.
    assert_vec("swa_attn_gqa", &got, &want, 5e-2, 1e-1)
}

fn main() -> Result<()> {
    gpu_hal::set_device(0).map_err(|e| anyhow!("set_device: {e}"))?;

    let tests: Vec<(&str, fn() -> Result<()>)> = vec![
        ("rms_norm", test_rms_norm),
        ("rms_norm_no_weight", test_rms_norm_no_weight),
        ("matvec", test_matvec),
        ("gelu_tanh_gate_mul", test_gelu_tanh_gate_mul),
        ("rope_decode", test_rope_decode),
        ("kv_append", test_kv_append),
        ("swa_attn_uniform", test_swa_attn_uniform),
        ("swa_attn_window_masks_oldest", test_swa_attn_window_masks_oldest),
        ("swa_attn_gqa", test_swa_attn_gqa),
    ];

    let mut failures = 0usize;
    for (name, f) in tests {
        match f() {
            Ok(()) => println!("PASS  {name}"),
            Err(e) => {
                failures += 1;
                println!("FAIL  {name}: {e:#}");
            }
        }
    }

    if failures == 0 {
        println!("\nAll Gemma 4 primitive tests passed.");
        Ok(())
    } else {
        bail!("{failures} Gemma 4 primitive test(s) failed")
    }
}
