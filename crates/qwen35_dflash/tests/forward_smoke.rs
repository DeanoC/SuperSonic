//! M2 smoke test for the DFlash draft forward pass.
//!
//! Gates:
//!   - All outputs finite (no NaN / no Inf).
//!   - Runtime under a generous 2 s sanity ceiling (hang / explosion catch).
//!
//! The plan's <2 ms target cannot be met by the M2 vanilla per-layer-GEMM
//! design on gfx1150 — 15 launches/layer × 5 layers dominates wall time at
//! q_len=16. Megakernel-ifying the draft is explicitly M4 scope; this smoke
//! just reports timing so M4 has a baseline to beat.
//!
//! The test loads the real draft checkpoint from `$QWEN35_DFLASH_DIR`
//! (default `~/models/qwen35-9b-dflash`) and fabricates tiny placeholder
//! `embed_tokens` / `lm_head` Arcs — those tensors exist in the target,
//! not in the draft (see docs/dflash.md §7), and the forward does not read
//! them. The random `noise_embedding` / `target_hidden_raw` fed in make
//! this a smoke test for the arithmetic pipeline, not for token quality —
//! token quality is validated end-to-end in M3 greedy-equivalence.
//!
//! The test is `#[ignore]` by default because it requires the 2 GiB
//! checkpoint and a functional HIP runtime. Run with:
//!
//!   cargo test --release -p qwen35_dflash -- --ignored forward_smoke

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gpu_hal::{GpuBuffer, ScalarType};
use qwen35_dflash::{
    config::load_config, forward::forward, rotary::RotaryTables, state::DFlashScratch,
    weights::DFlashWeights, ForwardParams,
};

fn checkpoint_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("QWEN35_DFLASH_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").expect("$HOME must be set");
    PathBuf::from(home).join("models/qwen35-9b-dflash")
}

fn random_bf16_bytes(count: usize, seed: u64) -> Vec<u8> {
    // Tiny LCG — deterministic, no rand crate dependency. Values in [-0.05, 0.05).
    let mut s: u64 = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(count * 2);
    for _ in 0..count {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) as f32 / (u32::MAX as f32);
        let v = (u - 0.5) * 0.1;
        let bf = half::bf16::from_f32(v);
        out.extend_from_slice(&bf.to_le_bytes());
    }
    out
}

#[test]
#[ignore = "requires the 2 GiB z-lab/Qwen3.5-9B-DFlash checkpoint + HIP runtime"]
fn forward_smoke() {
    let dir = checkpoint_dir();
    assert!(
        dir.join("config.json").exists(),
        "config.json missing at {}",
        dir.display()
    );

    let config = load_config(&dir).expect("load DFlash config");
    let ordinal = 0_usize;

    // Fake embed_tokens / lm_head. Shape doesn't matter — forward never reads them.
    let dummy = Arc::new(
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1])
            .expect("alloc dummy embed placeholder"),
    );
    let weights = DFlashWeights::load(&dir, &config, ordinal, dummy.clone(), dummy.clone())
        .expect("load DFlash weights");

    // Smoke uses a small RoPE table to keep VRAM modest. pos_offset + seq < 256.
    let rotary = RotaryTables::build(&config, ordinal, 256).expect("build RoPE tables");

    let mut scratch = DFlashScratch::new(ordinal, &config).expect("alloc scratch");

    let q_len = config.block_size;
    let ctx_len = 1;
    let hidden = config.hidden_size;
    let num_taps_hidden = config.fuser_in_dim();

    let noise_bytes = random_bf16_bytes(q_len * hidden, 0xDF1A5_u64);
    let noise_embedding =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[1, q_len, hidden], &noise_bytes)
            .expect("upload noise_embedding");

    let target_bytes = random_bf16_bytes(ctx_len * num_taps_hidden, 0xB1007_u64);
    let target_hidden_raw = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, ctx_len, num_taps_hidden],
        &target_bytes,
    )
    .expect("upload target_hidden_raw");

    let params = ForwardParams { past_len: 0, ctx_len, q_len, pos_offset: 0 };

    // ---- Warmup ----
    for _ in 0..5 {
        let _ = forward(
            &weights,
            &mut scratch,
            &rotary,
            &noise_embedding,
            &target_hidden_raw,
            ForwardParams { ..params },
        )
        .expect("warmup forward");
        gpu_hal::sync(ordinal).expect("warmup sync");
    }

    // ---- Timed trials ----
    let trials = 50_usize;
    let start = Instant::now();
    for _ in 0..trials {
        let _ = forward(
            &weights,
            &mut scratch,
            &rotary,
            &noise_embedding,
            &target_hidden_raw,
            ForwardParams { ..params },
        )
        .expect("timed forward");
    }
    gpu_hal::sync(ordinal).expect("final sync");
    let elapsed = start.elapsed();
    let per_call_ms = elapsed.as_secs_f64() * 1000.0 / trials as f64;

    // ---- Correctness: read back final_hidden and assert finite ----
    let final_hidden = forward(
        &weights,
        &mut scratch,
        &rotary,
        &noise_embedding,
        &target_hidden_raw,
        ForwardParams { ..params },
    )
    .expect("correctness forward");
    gpu_hal::sync(ordinal).expect("correctness sync");

    let host_bytes = {
        let mut out = vec![0u8; final_hidden.len_bytes()];
        gpu_hal::copy_d2h(
            ordinal,
            out.as_mut_ptr() as *mut std::ffi::c_void,
            final_hidden.as_ptr(),
            out.len(),
        )
        .expect("d2h final_hidden");
        out
    };

    let mut bad = 0usize;
    let mut total = 0usize;
    for chunk in host_bytes.chunks_exact(2) {
        let bf = half::bf16::from_le_bytes([chunk[0], chunk[1]]);
        let v = bf.to_f32();
        if !v.is_finite() {
            bad += 1;
        }
        total += 1;
    }
    assert_eq!(bad, 0, "final_hidden has {bad}/{total} non-finite values");

    println!(
        "forward_smoke: q_len={q_len} ctx_len={ctx_len} past_len=0 → {per_call_ms:.3} ms/call \
         over {trials} trials (M2 baseline; M4 megakernel-ification target < 2 ms)"
    );
    assert!(
        per_call_ms < 2000.0,
        "forward runtime {per_call_ms:.3} ms exceeds 2000 ms sanity ceiling (hang / explosion?)"
    );
}
