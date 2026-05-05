//! Kernel-direct parity sweep for the M3 batched-Q full-attention prefill
//! kernel (`qwen36_moe::batched_prefill_attn_full_launch`).
//!
//! Compares the kernel's BF16-input / F32-output against a CPU fp32
//! reference across (q_len, kv_len, head_dim) shapes representative of
//! Qwen 3.6 MoE. Production shape is `(q_heads=16, kv_heads=2, hd=256)`;
//! the sweep also covers small head_dim regimes for portability.
//!
//! q_len values include non-multiples of BM=4 so the inactive-warp
//! tail-block code path gets exercised (PR #219 caught a related bug
//! with this regime in the Qwen 3.5 tiled kernel).
//!
//! Skipped silently when HIP backend isn't compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::qwen36_moe::batched_prefill_attn_full_launch;

fn upload_bf16(ordinal: usize, host: &[half::bf16], shape: &[usize]) -> GpuBuffer {
    assert_eq!(host.len(), shape.iter().product::<usize>());
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, shape).expect("alloc bf16");
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    gpu_hal::copy_h2d(ordinal, buf.as_mut_ptr(), bytes.as_ptr() as *const _, bytes.len())
        .expect("h2d bf16");
    buf
}

fn download_f32(buf: &GpuBuffer) -> Vec<f32> {
    let n = buf.elem_count();
    let mut bytes = vec![0u8; n * 4];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )
    .expect("d2h f32");
    let mut out = vec![0.0f32; n];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    out
}

fn make_host_bf16(n: usize, seed: usize) -> (Vec<half::bf16>, Vec<f32>) {
    let bf: Vec<half::bf16> = (0..n)
        .map(|i| {
            let v = ((i + seed) as f32 * 0.0017).sin() * 0.4 + 0.05;
            half::bf16::from_f32(v)
        })
        .collect();
    let f32_round = bf.iter().map(|x| x.to_f32()).collect();
    (bf, f32_round)
}

fn cpu_attention_fp32(
    batch: usize, q_heads: usize, kv_heads: usize,
    q_len: usize, kv_len: usize, head_dim: usize,
    scale: f32, seqlen_offset: usize,
    q: &[f32], k: &[f32], v: &[f32],
) -> Vec<f32> {
    let groups = q_heads / kv_heads;
    let mut out = vec![0.0f32; batch * q_heads * q_len * head_dim];
    for b in 0..batch {
        for hq in 0..q_heads {
            let hk = hq / groups;
            for qi in 0..q_len {
                let limit = (seqlen_offset + qi + 1).min(kv_len);
                let q_off = ((b * q_heads + hq) * q_len + qi) * head_dim;
                let k_head = (b * kv_heads + hk) * kv_len * head_dim;
                let v_head = (b * kv_heads + hk) * kv_len * head_dim;
                let mut scores = vec![0f32; limit];
                for kp in 0..limit {
                    let k_row = k_head + kp * head_dim;
                    let mut s = 0.0f32;
                    for d in 0..head_dim { s += q[q_off + d] * k[k_row + d]; }
                    scores[kp] = s * scale;
                }
                let mut m = f32::NEG_INFINITY;
                for &s in &scores { if s > m { m = s; } }
                let mut denom = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - m).exp(); denom += *s; }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let out_off = ((b * q_heads + hq) * q_len + qi) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for kp in 0..limit {
                        let v_row = v_head + kp * head_dim;
                        acc += scores[kp] * v[v_row + d];
                    }
                    out[out_off + d] = acc * inv;
                }
            }
        }
    }
    out
}

fn run_one_shape(
    ordinal: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    seqlen_offset: usize,
    seed: usize,
) {
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_elems = batch * q_heads * q_len * head_dim;
    let k_elems = batch * kv_heads * kv_len * head_dim;
    let v_elems = batch * kv_heads * kv_len * head_dim;

    let (q_bf, q_f32) = make_host_bf16(q_elems, seed);
    let (k_bf, k_f32) = make_host_bf16(k_elems, seed + 1000);
    let (v_bf, v_f32) = make_host_bf16(v_elems, seed + 2000);

    let q_buf = upload_bf16(ordinal, &q_bf, &[batch, q_heads, q_len, head_dim]);
    let k_buf = upload_bf16(ordinal, &k_bf, &[batch, kv_heads, kv_len, head_dim]);
    let v_buf = upload_bf16(ordinal, &v_bf, &[batch, kv_heads, kv_len, head_dim]);
    let mut out_buf = GpuBuffer::zeros(
        ordinal, ScalarType::F32, &[batch, q_heads, q_len, head_dim],
    ).expect("alloc out");

    batched_prefill_attn_full_launch(
        ordinal, batch, q_heads, kv_heads, q_len, kv_len, head_dim,
        scale, seqlen_offset, &q_buf, &k_buf, &v_buf, &mut out_buf,
    ).expect("ffi");
    gpu_hal::sync(ordinal).expect("sync");

    let got = download_f32(&out_buf);
    let want = cpu_attention_fp32(
        batch, q_heads, kv_heads, q_len, kv_len, head_dim,
        scale, seqlen_offset, &q_f32, &k_f32, &v_f32,
    );

    // Tolerances: max_abs guards correctness for normal-magnitude outputs
    // (BF16 accumulated dot products at hd=256 land in ~1e-3 noise floor);
    // max_rel guards correctness for non-tiny outputs only — clamping the
    // denominator at 1e-3 stops near-zero attention outputs (where the
    // softmax weights cancel) from tripping a relative bar that has no
    // physical meaning at that scale. The Qwen 3.5 Tier 1 sweep used 1e-6
    // as the floor and got away with hd=128; at hd=256 the accumulator's
    // rounding noise makes 1e-6 fragile.
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..got.len() {
        let a = (got[i] - want[i]).abs();
        let r = a / want[i].abs().max(1e-3);
        if a > max_abs { max_abs = a; }
        if r > max_rel { max_rel = r; }
    }
    assert!(
        max_abs < 2e-2 && max_rel < 5e-2,
        "shape q_h={} kv_h={} q_len={} kv_len={} hd={}: max_abs={} max_rel={}",
        q_heads, kv_heads, q_len, kv_len, head_dim, max_abs, max_rel,
    );
}

#[test]
fn batched_prefill_attn_full_parity_sweep() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let ordinal = 0usize;

    // Production shape (qwen3.6-35b-a3b): q_heads=16, kv_heads=2, hd=256.
    // Tail q_len values (15, 17, 33) exercise the inactive-warp code path.
    for &(q_heads, kv_heads, q_len, kv_len, head_dim) in &[
        // Small portability shapes
        (4usize, 2usize, 16usize, 16usize, 64usize),
        (4, 2, 16, 16, 128),
        // Production shape
        (16, 2, 16, 16, 256),
        (16, 2, 15, 15, 256),  // tail: q_len % BM == 3
        (16, 2, 17, 17, 256),  // tail: q_len % BM == 1
        (16, 2, 33, 64, 256),  // tail with kv_len > q_len (chunk after past tokens)
        (16, 2, 64, 64, 256),
        (16, 2, 256, 256, 256),
        (16, 2, 512, 512, 256),
        // Past-context regime: q_len < kv_len, seqlen_offset > 0
        (16, 2, 16, 256, 256),
    ] {
        let seqlen_offset = if kv_len > q_len { kv_len - q_len } else { 0 };
        run_one_shape(ordinal, q_heads, kv_heads, q_len, kv_len, head_dim, seqlen_offset, 0xC36E);
    }
}
