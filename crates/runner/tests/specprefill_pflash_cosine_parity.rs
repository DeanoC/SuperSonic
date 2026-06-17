//! Parity test for `pflash_cosine_score` (SpecPrefill — Phase D).
//!
//! Asserts the GPU kernel produces per-block cosine values within 1e-3
//! of a straight CPU reference, AND that each cosine sits in [-1, 1].
//! Skipped silently when HIP is not compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi::pflash_cosine_score;

fn make_bf16_kv_cache(
    ordinal: usize,
    kv_heads: usize,
    cap: usize,
    head_dim: usize,
    seed_offset: usize,
) -> (GpuBuffer, Vec<f32>) {
    // Layout [1, kv_heads, cap, head_dim] flattened.
    let total = kv_heads * cap * head_dim;
    let host: Vec<half::bf16> = (0..total)
        .map(|i| {
            let v = ((i + seed_offset) as f32 * 0.0021).sin() * 0.4 + 0.1;
            half::bf16::from_f32(v)
        })
        .collect();
    let host_f32: Vec<f32> = host.iter().map(|b| b.to_f32()).collect();
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_heads, cap, head_dim])
        .expect("alloc kv");
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )
    .expect("h2d kv");
    (buf, host_f32)
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

#[test]
fn pflash_cosine_score_matches_cpu_reference() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let ordinal = 0usize;
    let kv_heads = 4usize;
    let cap = 200usize; // > n_pos so the kernel exercises the cap-stride path
    let head_dim = 128usize; // realistic Qwen3.5 head_dim
    let n_pos = 137usize; // non-power-of-two prompt length, < cap
    let block_size = 32usize;
    let n_blocks = (n_pos + block_size - 1) / block_size; // = 5
    let last_pos = n_pos - 1;
    let kv_dim = kv_heads * head_dim;

    let (k_buf, k_host) = make_bf16_kv_cache(ordinal, kv_heads, cap, head_dim, 0);
    let mut scores = GpuBuffer::zeros(ordinal, ScalarType::F32, &[n_blocks]).expect("alloc scores");

    pflash_cosine_score(
        ordinal,
        ScalarType::BF16,
        n_pos,
        kv_heads,
        cap,
        head_dim,
        block_size,
        last_pos,
        &k_buf,
        &mut scores,
    )
    .expect("kernel launch");

    // CPU reference. Per-position K[d]: `k_host[(h * cap + pos) * head_dim + dim_in_head]`.
    let kv_at = |pos: usize, d: usize| -> f32 {
        let h = d / head_dim;
        let dim_in_head = d - h * head_dim;
        k_host[(h * cap + pos) * head_dim + dim_in_head]
    };
    let mut want = vec![0.0_f32; n_blocks];
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = ((b + 1) * block_size).min(n_pos);
        let block_len = end - start;
        let mut dot = 0.0_f32;
        let mut nb = 0.0_f32;
        let mut nl = 0.0_f32;
        for d in 0..kv_dim {
            let last_v = kv_at(last_pos, d);
            let mut sum = 0.0_f32;
            for pos in start..end {
                sum += kv_at(pos, d);
            }
            let mean = sum / block_len as f32;
            dot += mean * last_v;
            nb += mean * mean;
            nl += last_v * last_v;
        }
        let mut denom = nb.sqrt() * nl.sqrt();
        if denom < 1e-12 {
            denom = 1e-12;
        }
        want[b] = dot / denom;
    }

    let got = d2h_f32(&scores);
    assert_eq!(got.len(), want.len());

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 1e-3,
        "GPU cosine differs from CPU reference: max_abs={max_abs}"
    );

    for (b, &s) in got.iter().enumerate() {
        assert!(
            s >= -1.0001 && s <= 1.0001,
            "score[{b}] = {s} outside [-1, 1]"
        );
    }
}
