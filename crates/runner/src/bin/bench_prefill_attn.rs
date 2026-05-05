//! Micro-benchmark for `prefill_ffi::full_attention_prefill`.
//!
//! Times the prefill attention kernel in isolation across a sweep of
//! (q_len, kv_len, head_dim) shapes, printing ms/call. Used as the perf
//! gate for Stage 2 (online softmax) and Stage 3 (K-tiled) rewrites.
//!
//! Run: `cargo run --release --bin bench_prefill_attn`

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi;
use std::time::Instant;

fn upload_zeros_bf16(ordinal: usize, shape: &[usize]) -> GpuBuffer {
    // Zeros are fine for timing — kernel work is shape-driven, not data-driven.
    GpuBuffer::zeros(ordinal, ScalarType::BF16, shape).expect("alloc bf16")
}

fn bench_shape(
    ordinal: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_len: usize,
    kv_len: usize,
    iters: usize,
) -> f64 {
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_buf = upload_zeros_bf16(ordinal, &[batch, q_heads, q_len, head_dim]);
    let k_buf = upload_zeros_bf16(ordinal, &[batch, kv_heads, kv_len, head_dim]);
    let v_buf = upload_zeros_bf16(ordinal, &[batch, kv_heads, kv_len, head_dim]);
    let mut out_buf = GpuBuffer::zeros(
        ordinal, ScalarType::F32, &[batch, q_heads, q_len, head_dim],
    ).expect("alloc out");

    let seqlen_offset = if kv_len > q_len { kv_len - q_len } else { 0 };

    let mut call = || {
        prefill_ffi::full_attention_prefill(
            ordinal, ScalarType::BF16,
            batch, q_heads, kv_heads, q_len, kv_len, head_dim,
            scale, seqlen_offset, &q_buf, &k_buf, &v_buf, &mut out_buf,
        ).expect("ffi");
    };

    for _ in 0..3 { call(); }
    gpu_hal::sync(ordinal).expect("sync");

    let t0 = Instant::now();
    for _ in 0..iters { call(); }
    gpu_hal::sync(ordinal).expect("sync");
    let dt = t0.elapsed().as_secs_f64();
    dt * 1000.0 / iters as f64
}

fn main() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("HIP backend not compiled");
        return;
    }
    let ordinal = 0usize;
    println!("ordinal: {ordinal}");

    // (q_heads, kv_heads, head_dim) configs reflect the real shipping configs.
    // All flagship models (Qwen 3.5 *, Qwen 3.6 35B-A3B, Gemma 4 sliding) use
    // head_dim=256 — that's the row that matters for production. The hd=64/128
    // rows are kept synthetic so future small-head-dim models (e.g. Phi-4-mini)
    // are covered without a third bench edit.
    //   - (8,  2, 256)   — qwen3.5-0.8b / 2b
    //   - (16, 4, 256)   — qwen3.5-4b / 9b   and qwen3.6-35b-a3b
    //   - (16, 8,  64)   — synthetic (small head_dim regime)
    //   - (16, 8, 128)   — synthetic (medium head_dim regime)
    let configs = [
        (8usize, 2usize, 256usize),
        (16, 4, 256),
        (16, 8, 64),
        (16, 8, 128),
    ];

    for (q_heads, kv_heads, head_dim) in configs {
        println!("\n=== q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim} ===");
        println!("{:>8} {:>8} {:>10}  {:>10}", "q_len", "kv_len", "ms/call", "tok/s");
        for &n in &[256usize, 1024, 4096, 16384] {
            // Bulk prefill: q_len == kv_len mimics the first chunk of a long prompt.
            let iters = if n >= 4096 { 5 } else { 20 };
            let ms = bench_shape(ordinal, q_heads, kv_heads, head_dim, n, n, iters);
            let tok_per_s = (n as f64 / ms) * 1000.0;
            println!("{:>8} {:>8} {:>10.3}  {:>10.1}", n, n, ms, tok_per_s);
        }
    }
}
