//! M2/M3 smoke test for the DFlash draft forward pass.
//!
//! M2 gates (round 1, `past_len = 0`):
//!   - All outputs finite (no NaN / no Inf).
//!   - Runtime under a generous 2 s sanity ceiling (hang / explosion catch).
//!
//! M3.1 gates (round 2+):
//!   - The Lucebox-compatible stateless draft path produces finite outputs.
//!   - Context windows larger than the 16-token draft block are accepted.
//!   - Repeated calls do not leak draft KV state across rounds.
//!
//! The plan's <2 ms target cannot be met by the vanilla per-layer-GEMM
//! design on gfx1150 — ~75 launches dominates wall time at q_len=16.
//! Megakernel-ifying is M4 scope; this smoke reports timing as a baseline.
//!
//! The test loads the real draft checkpoint from `$QWEN35_DFLASH_DIR`
//! (default `~/models/qwen35-9b-dflash`) and fabricates tiny placeholder
//! `embed_tokens` / `lm_head` Arcs — those tensors exist in the target,
//! not in the draft (see docs/dflash.md §7), and forward does not read
//! them. Random `noise_embedding` / `target_hidden_raw` make this a smoke
//! test for the arithmetic pipeline, not for token quality — quality is
//! validated end-to-end in M3 greedy-equivalence once the engine lands.
//!
//! `#[ignore]` by default because it requires the 2 GiB checkpoint and a
//! functional HIP runtime. Run with:
//!
//!   cargo test --release -p qwen35_dflash -- --ignored forward_smoke

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gpu_hal::{GpuBuffer, ScalarType};
use qwen35_dflash::{
    config::load_config,
    forward::forward,
    rotary::RotaryTables,
    state::{DFlashScratch, DFlashState},
    weights::DFlashWeights,
    ForwardParams,
};

fn checkpoint_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("QWEN35_DFLASH_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").expect("$HOME must be set");
    PathBuf::from(home).join("models/qwen35-9b-dflash")
}

fn random_bf16_bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut s: u64 = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(count * 2);
    for _ in 0..count {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 33) as u32) as f32 / (u32::MAX as f32);
        let v = (u - 0.5) * 0.1;
        let bf = half::bf16::from_f32(v);
        out.extend_from_slice(&bf.to_le_bytes());
    }
    out
}

fn read_final_hidden_to_host(ordinal: usize, buf: &GpuBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.len_bytes()];
    gpu_hal::copy_d2h(
        ordinal,
        out.as_mut_ptr() as *mut std::ffi::c_void,
        buf.as_ptr(),
        out.len(),
    )
    .expect("d2h final_hidden");
    out
}

fn count_non_finite(bytes: &[u8]) -> (usize, usize) {
    let mut bad = 0usize;
    let mut total = 0usize;
    for chunk in bytes.chunks_exact(2) {
        let bf = half::bf16::from_le_bytes([chunk[0], chunk[1]]);
        if !bf.to_f32().is_finite() {
            bad += 1;
        }
        total += 1;
    }
    (bad, total)
}

fn upload_fuser_input(ordinal: usize, scratch: &mut DFlashScratch, bytes: &[u8]) {
    assert!(
        bytes.len() <= scratch.fuser_input.len_bytes(),
        "fuser input too small: {} < {}",
        scratch.fuser_input.len_bytes(),
        bytes.len()
    );
    gpu_hal::copy_h2d(
        ordinal,
        scratch.fuser_input.as_mut_ptr(),
        bytes.as_ptr() as *const std::ffi::c_void,
        bytes.len(),
    )
    .expect("upload fuser_input");
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

    let dummy = Arc::new(
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1]).expect("alloc dummy placeholder"),
    );
    let weights = DFlashWeights::load(&dir, &config, ordinal, dummy.clone(), dummy.clone())
        .expect("load DFlash weights");

    // Keep the RoPE table modest — smoke only needs a few dozen positions.
    let max_ctx = 256_usize;
    let rotary = RotaryTables::build(&config, ordinal, max_ctx).expect("build RoPE tables");

    let ctx_capacity = 64_usize;
    let mut scratch = DFlashScratch::new_with_ctx_capacity(ordinal, &config, ctx_capacity)
        .expect("alloc scratch");
    let mut state = DFlashState::new(ordinal, &config, max_ctx).expect("alloc state");

    let q_len = config.block_size;
    let ctx_len = 1;
    let hidden = config.hidden_size;
    let num_taps_hidden = config.fuser_in_dim();

    let noise_bytes = random_bf16_bytes(q_len * hidden, 0xDF1A5_u64);
    let noise_embedding =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[1, q_len, hidden], &noise_bytes)
            .expect("upload noise_embedding");

    let target_bytes = random_bf16_bytes(ctx_len * num_taps_hidden, 0xB1007_u64);
    upload_fuser_input(ordinal, &mut scratch, &target_bytes);

    // ---- Warmup (5 rounds, resetting cache between to avoid overflow) ----
    for _ in 0..5 {
        state.reset();
        upload_fuser_input(ordinal, &mut scratch, &target_bytes);
        let _ = forward(
            &weights,
            &mut state,
            &mut scratch,
            &rotary,
            &noise_embedding,
            ForwardParams {
                ctx_len,
                q_len,
                pos_offset: 0,
            },
        )
        .expect("warmup forward");
        gpu_hal::sync(ordinal).expect("warmup sync");
    }

    // ---- Timed trials (round-1 path, reset every iteration) ----
    let trials = 50_usize;
    let start = Instant::now();
    for _ in 0..trials {
        state.reset();
        let _ = forward(
            &weights,
            &mut state,
            &mut scratch,
            &rotary,
            &noise_embedding,
            ForwardParams {
                ctx_len,
                q_len,
                pos_offset: 0,
            },
        )
        .expect("timed forward");
    }
    gpu_hal::sync(ordinal).expect("timed sync");
    let per_call_ms = start.elapsed().as_secs_f64() * 1000.0 / trials as f64;

    // ---- Round-1 correctness ----
    state.reset();
    upload_fuser_input(ordinal, &mut scratch, &target_bytes);
    let r1 = forward(
        &weights,
        &mut state,
        &mut scratch,
        &rotary,
        &noise_embedding,
        ForwardParams {
            ctx_len,
            q_len,
            pos_offset: 0,
        },
    )
    .expect("round-1 forward");
    gpu_hal::sync(ordinal).expect("round-1 sync");
    assert_eq!(
        state.kv_filled, 0,
        "stateless draft forward must not carry KV rows between rounds"
    );
    let r1_bytes = read_final_hidden_to_host(ordinal, r1);
    let (bad, total) = count_non_finite(&r1_bytes);
    assert_eq!(
        bad, 0,
        "round 1 final_hidden has {bad}/{total} non-finite values"
    );

    // Round 2: stateless replay with a larger target-history window.
    let ctx_len_r2 = 10;
    let target_bytes_r2 = random_bf16_bytes(ctx_len_r2 * num_taps_hidden, 0xB1008_u64);
    upload_fuser_input(ordinal, &mut scratch, &target_bytes_r2);
    let noise_bytes_r2 = random_bf16_bytes(q_len * hidden, 0xDF1A6_u64);
    let noise_embedding_r2 = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[1, q_len, hidden],
        &noise_bytes_r2,
    )
    .expect("upload round-2 noise_embedding");

    let r2 = forward(
        &weights,
        &mut state,
        &mut scratch,
        &rotary,
        &noise_embedding_r2,
        ForwardParams {
            ctx_len: ctx_len_r2,
            q_len,
            pos_offset: 0,
        },
    )
    .expect("round-2 forward");
    gpu_hal::sync(ordinal).expect("round-2 sync");
    assert_eq!(
        state.kv_filled, 0,
        "round 2 must still leave no persistent draft KV state"
    );
    let r2_bytes = read_final_hidden_to_host(ordinal, r2);
    let (bad, total) = count_non_finite(&r2_bytes);
    assert_eq!(
        bad, 0,
        "round 2 final_hidden has {bad}/{total} non-finite values"
    );

    // ---- Round 3 with ctx_len > block_size (full target-history window) ----
    let ctx_len_r3 = q_len + 8;
    let target_bytes_r3 = random_bf16_bytes(ctx_len_r3 * num_taps_hidden, 0xB1009_u64);
    upload_fuser_input(ordinal, &mut scratch, &target_bytes_r3);
    let r3 = forward(
        &weights,
        &mut state,
        &mut scratch,
        &rotary,
        &noise_embedding_r2,
        ForwardParams {
            ctx_len: ctx_len_r3,
            q_len,
            pos_offset: 0,
        },
    )
    .expect("round-3 forward");
    gpu_hal::sync(ordinal).expect("round-3 sync");
    let r3_bytes = read_final_hidden_to_host(ordinal, r3);
    let (bad, total) = count_non_finite(&r3_bytes);
    assert_eq!(
        bad, 0,
        "round 3 final_hidden has {bad}/{total} non-finite values"
    );

    println!(
        "forward_smoke: q_len={q_len} ctx_len={ctx_len} past_len=0 → {per_call_ms:.3} ms/call \
         over {trials} trials (M2 baseline; M4 megakernel-ification target < 2 ms)"
    );
    println!(
        "multi-round stateless: round-2 ctx={ctx_len_r2}, \
         round-3 ctx={ctx_len_r3}; both finite"
    );
    assert!(
        per_call_ms < 2000.0,
        "forward runtime {per_call_ms:.3} ms exceeds 2000 ms sanity ceiling (hang / explosion?)"
    );
}
