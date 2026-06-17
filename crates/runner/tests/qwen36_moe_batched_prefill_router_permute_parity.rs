//! Kernel-direct parity sweep for the M9 router permutation kernel
//! (`qwen36_moe::batched_prefill_router_permute_launch`).
//!
//! Generates synthetic per-token top-K assignments at multiple shapes
//! (production: N=64, top_k=8, num_experts=256, plus smaller edge-case
//! shapes), uploads them, runs the kernel, downloads the four outputs,
//! and compares against a pure-CPU counting-sort reference.
//!
//! **Order stability:** the GPU's atomicAdd-based scatter cursor is not
//! a stable sort — the relative order of entries within an expert's
//! segment can differ from the CPU reference. We compare each segment as
//! a multiset of (token_idx, kpos, weight_bits) triples instead of
//! positionally. The downstream grouped GEMM (M10) is permutation-
//! invariant within a segment so this is the right invariant.
//!
//! Skipped silently when HIP backend isn't compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::qwen36_moe::batched_prefill_router_permute_launch;

// gpu-hal's ScalarType has no I32 variant — we use U32 as the 4-byte
// storage tag for buffers that semantically hold i32 values. The kernel
// reads them as `int*` so the bit pattern is what matters; the tag only
// affects elem_count() rounding.
fn upload_i32(ordinal: usize, host: &[i32], shape: &[usize]) -> GpuBuffer {
    assert_eq!(host.len(), shape.iter().product::<usize>());
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::U32, shape).expect("alloc i32");
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )
    .expect("h2d i32");
    buf
}

fn upload_bf16(ordinal: usize, host: &[half::bf16], shape: &[usize]) -> GpuBuffer {
    assert_eq!(host.len(), shape.iter().product::<usize>());
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

fn download_i32(buf: &GpuBuffer) -> Vec<i32> {
    let n = buf.elem_count();
    let mut bytes = vec![0u8; n * 4];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )
    .expect("d2h i32");
    let mut out = vec![0i32; n];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    out
}

fn download_bf16(buf: &GpuBuffer) -> Vec<half::bf16> {
    let n = buf.elem_count();
    let mut bytes = vec![0u8; n * 2];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )
    .expect("d2h bf16");
    let mut out = vec![half::bf16::ZERO; n];
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        out[i] = half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

/// Tiny deterministic LCG so the test doesn't pull a `rand` dependency.
/// Returns (next_state, value).
fn lcg(state: u64) -> (u64, u64) {
    let next = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (next, next >> 32)
}

fn make_topk_idx(n_tokens: usize, top_k: usize, num_experts: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_tokens * top_k);
    for _t in 0..n_tokens {
        // Sample top_k distinct expert ids per token by the simplest method:
        // pull `top_k` candidates, dedupe by replacement on collision. With
        // top_k <= num_experts this terminates quickly and exercises the
        // kernel's path of "many tokens map to overlapping expert subsets".
        let mut chosen: Vec<i32> = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let (next, raw) = lcg(state);
            state = next;
            let cand = (raw as usize % num_experts) as i32;
            if !chosen.contains(&cand) {
                chosen.push(cand);
            }
        }
        out.extend_from_slice(&chosen);
    }
    out
}

fn make_topk_weight(n_tokens: usize, top_k: usize, seed: u64) -> Vec<half::bf16> {
    let total = n_tokens * top_k;
    let mut out = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        let (next, raw) = lcg(state);
        state = next;
        // Map to a bounded BF16-friendly range; specific values don't matter
        // (we only check exact bitwise copy-through).
        let v = ((raw % 10_000) as f32) / 10_000.0;
        out.push(half::bf16::from_f32(v * 0.5 + 0.01));
    }
    out
}

fn cpu_reference(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    topk_idx: &[i32],
    topk_weight: &[half::bf16],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<half::bf16>) {
    let total = n_tokens * top_k;
    let mut counts = vec![0i32; num_experts];
    for &e in topk_idx {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0i32; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut p_token = vec![0i32; total];
    let mut p_kpos = vec![0i32; total];
    let mut p_w = vec![half::bf16::ZERO; total];
    let mut cursors = vec![0i32; num_experts];
    for entry in 0..total {
        let e = topk_idx[entry] as usize;
        let dst = (offsets[e] + cursors[e]) as usize;
        p_token[dst] = (entry / top_k) as i32;
        p_kpos[dst] = (entry % top_k) as i32;
        p_w[dst] = topk_weight[entry];
        cursors[e] += 1;
    }
    (offsets, p_token, p_kpos, p_w)
}

fn run_one_shape(ordinal: usize, n_tokens: usize, top_k: usize, num_experts: usize, seed: u64) {
    assert!(top_k <= num_experts);
    let total = n_tokens * top_k;

    let topk_idx_host = make_topk_idx(n_tokens, top_k, num_experts, seed);
    let topk_w_host = make_topk_weight(n_tokens, top_k, seed.wrapping_add(0xA5A5));

    let topk_idx_buf = upload_i32(ordinal, &topk_idx_host, &[n_tokens, top_k]);
    let topk_w_buf = upload_bf16(ordinal, &topk_w_host, &[n_tokens, top_k]);

    let mut offsets_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[num_experts + 1]).expect("alloc offsets");
    let mut p_token_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[total]).expect("alloc p_token");
    let mut p_kpos_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[total]).expect("alloc p_kpos");
    let mut p_w_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[total]).expect("alloc p_w");

    batched_prefill_router_permute_launch(
        ordinal,
        n_tokens,
        top_k,
        num_experts,
        &topk_idx_buf,
        &topk_w_buf,
        &mut offsets_buf,
        &mut p_token_buf,
        &mut p_kpos_buf,
        &mut p_w_buf,
    )
    .expect("ffi");
    gpu_hal::sync(ordinal).expect("sync");

    let got_offsets = download_i32(&offsets_buf);
    let got_token = download_i32(&p_token_buf);
    let got_kpos = download_i32(&p_kpos_buf);
    let got_w = download_bf16(&p_w_buf);

    let (want_offsets, want_token, want_kpos, want_w) =
        cpu_reference(n_tokens, top_k, num_experts, &topk_idx_host, &topk_w_host);

    // expert_offsets must match exactly — it's deterministic.
    assert_eq!(
        got_offsets, want_offsets,
        "expert_offsets mismatch at shape (N={n_tokens}, top_k={top_k}, num_experts={num_experts})",
    );
    assert_eq!(
        got_offsets[num_experts] as usize, total,
        "expert_offsets[num_experts] should equal n_tokens * top_k",
    );

    // For each expert, compare the (token_idx, kpos, weight_bits) triples
    // as a multiset — the GPU scatter cursor is unstable inside a segment.
    for e in 0..num_experts {
        let lo = got_offsets[e] as usize;
        let hi = got_offsets[e + 1] as usize;
        let mut got_set: Vec<(i32, i32, u16)> = (lo..hi)
            .map(|i| (got_token[i], got_kpos[i], got_w[i].to_bits()))
            .collect();
        let mut want_set: Vec<(i32, i32, u16)> = (lo..hi)
            .map(|i| (want_token[i], want_kpos[i], want_w[i].to_bits()))
            .collect();
        got_set.sort();
        want_set.sort();
        assert_eq!(
            got_set, want_set,
            "expert {e} segment mismatch at shape (N={n_tokens}, top_k={top_k}, num_experts={num_experts})",
        );

        // Sanity-check that the (token, kpos) pair really is one of the
        // top_k entries the original input mapped to this expert.
        for i in lo..hi {
            let token = got_token[i] as usize;
            let kpos = got_kpos[i] as usize;
            assert!(token < n_tokens && kpos < top_k);
            assert_eq!(
                topk_idx_host[token * top_k + kpos] as usize,
                e,
                "permuted entry e={e} idx={i} token={token} kpos={kpos} \
                 doesn't reverse-map to expert e",
            );
        }
    }

    // Total entry count across all segments must equal n_tokens * top_k.
    let total_in_segments: usize = (0..num_experts)
        .map(|e| (got_offsets[e + 1] - got_offsets[e]) as usize)
        .sum();
    assert_eq!(total_in_segments, total);
}

#[test]
fn router_permute_matches_cpu_reference() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let ordinal = 0usize;

    // Sweep production shape (N=64, top_k=8, num_experts=256) plus smaller
    // shapes to exercise edge cases:
    // - single token (verifies count phase doesn't OOB on tiny totals)
    // - tail entries < BLOCK threads
    // - degenerate top_k=1 (every entry is its own expert)
    // - smaller num_experts (verifies LDS init loop strides correctly)
    // - 128 tokens (max chunk size; total = 1024 entries)
    for &(n_tokens, top_k, num_experts, seed) in &[
        (1usize, 8usize, 256usize, 0x1u64),
        (16, 8, 256, 0x2),
        (64, 8, 256, 0x3), // production
        (128, 8, 256, 0x4),
        (1, 1, 4, 0x5),
        (8, 4, 32, 0x6),
        (32, 2, 16, 0x7),
        (64, 8, 64, 0x8), // top_k == num_experts/8: heavy collisions
    ] {
        run_one_shape(ordinal, n_tokens, top_k, num_experts, seed);
    }
}
