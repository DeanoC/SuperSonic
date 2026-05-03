//! Parity test for `lookahead_attention_scores` (SpecPrefill — Phase C).
//!
//! Asserts the GPU kernel produces softmax-of-(Q·Kᵀ) within 1e-3 of a
//! straight CPU reference, AND that each (q_head, q_row) row sums to 1.0
//! (the softmax invariant). Skipped silently when HIP is not compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi::lookahead_attention_scores;

fn make_bf16_buf(ordinal: usize, shape: &[usize], seed_offset: usize) -> GpuBuffer {
    let total: usize = shape.iter().product();
    let host: Vec<half::bf16> = (0..total)
        .map(|i| {
            let v = ((i + seed_offset) as f32 * 0.0019).sin() * 0.4 + 0.1;
            half::bf16::from_f32(v)
        })
        .collect();
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, shape).expect("alloc bf16");
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )
    .expect("h2d bf16");
    buf
}

fn d2h_f32(buf: &GpuBuffer) -> Vec<f32> {
    let elem_count = buf.elem_count();
    let mut bytes = vec![0u8; elem_count * 4];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )
    .expect("d2h f32");
    let mut out = vec![0.0f32; elem_count];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    out
}

fn host_bf16(shape: &[usize], seed_offset: usize) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| {
            // BF16 round-trip mirroring make_bf16_buf.
            let v = ((i + seed_offset) as f32 * 0.0019).sin() * 0.4 + 0.1;
            half::bf16::from_f32(v).to_f32()
        })
        .collect()
}

#[test]
fn lookahead_attention_matches_cpu_softmax() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let ordinal = 0usize;
    let q_heads = 8usize;
    let kv_heads = 2usize;
    let head_dim = 128usize;
    let lookahead_count = 5usize;
    let kv_len = 65usize;
    let num_kv_groups = q_heads / kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = make_bf16_buf(ordinal, &[lookahead_count, q_heads, head_dim], 0);
    let k = make_bf16_buf(ordinal, &[kv_heads, kv_len, head_dim], 1000);
    let mut scores = GpuBuffer::zeros(
        ordinal,
        ScalarType::F32,
        &[q_heads, lookahead_count, kv_len],
    )
    .expect("alloc scores");

    lookahead_attention_scores(
        ordinal,
        ScalarType::BF16,
        q_heads,
        kv_heads,
        lookahead_count,
        kv_len,
        head_dim,
        scale,
        &q,
        &k,
        &mut scores,
    )
    .expect("kernel launch");

    // CPU reference.
    let q_host = host_bf16(&[lookahead_count, q_heads, head_dim], 0);
    let k_host = host_bf16(&[kv_heads, kv_len, head_dim], 1000);
    let mut want = vec![0.0_f32; q_heads * lookahead_count * kv_len];
    for q_head in 0..q_heads {
        let kv_head = q_head / num_kv_groups;
        for q_row in 0..lookahead_count {
            // dot products
            let mut s = vec![0.0_f32; kv_len];
            for k_pos in 0..kv_len {
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    let qv = q_host[(q_row * q_heads + q_head) * head_dim + d];
                    let kv = k_host[(kv_head * kv_len + k_pos) * head_dim + d];
                    acc += qv * kv;
                }
                s[k_pos] = acc * scale;
            }
            // softmax
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0_f32;
            for v in s.iter_mut() {
                *v = (*v - m).exp();
                denom += *v;
            }
            for v in s.iter_mut() {
                *v /= denom;
            }
            let base = (q_head * lookahead_count + q_row) * kv_len;
            want[base..base + kv_len].copy_from_slice(&s);
        }
    }

    let got = d2h_f32(&scores);
    assert_eq!(got.len(), want.len());

    // Per-element max abs error. 1e-3 is a conservative tolerance: BF16
    // accumulation noise is ~1e-3 in the raw dot products, but softmax
    // normalisation compresses this significantly in practice — the
    // observed max_abs is typically an order of magnitude smaller.
    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 1e-3,
        "GPU softmax differs from CPU reference: max_abs={max_abs}"
    );

    // Softmax invariant: each (q_head, q_row) row sums to 1.0.
    for q_head in 0..q_heads {
        for q_row in 0..lookahead_count {
            let base = (q_head * lookahead_count + q_row) * kv_len;
            let s: f32 = got[base..base + kv_len].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-4,
                "row sum {s} != 1.0 at q_head={q_head} q_row={q_row}"
            );
        }
    }
}
