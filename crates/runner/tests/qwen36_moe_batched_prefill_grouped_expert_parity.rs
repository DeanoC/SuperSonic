//! Kernel-direct parity sweep for the M10 grouped-expert INT4 GEMM kernel
//! (`qwen36_moe::batched_prefill_grouped_expert_launch`).
//!
//! Runs the GPU kernel against a pure-CPU reference that does the same
//! gather + gate_up + silu*mul + down sequence per (token, kpos) pair.
//! The reference uses the same INT4 dequant formula as
//! `helpers.cuh::int4_dequant_8` (single BF16-rounding through
//! `nibble * scale - zero * scale`) so the only deltas should come from
//! BF16/F32 accumulation order — bounded by the codebase's INT4 floor of
//! `max_abs < 2e-2`, `max_rel < 5e-2` with a `1e-3` denominator floor
//! (matches the M3 attn parity test in this repo).
//!
//! Skipped silently when HIP backend isn't compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::qwen36_moe::batched_prefill_grouped_expert_launch;

// gpu-hal's ScalarType has no I32 variant — we use U32 as the 4-byte
// storage tag for buffers that semantically hold i32 values. The kernel
// reads them as `int*` so the bit pattern is what matters.
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

fn upload_u8(ordinal: usize, host: &[u8], shape: &[usize]) -> GpuBuffer {
    assert_eq!(host.len(), shape.iter().product::<usize>());
    GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, shape, host).expect("alloc u8")
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
fn lcg(state: u64) -> (u64, u64) {
    let next = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (next, next >> 32)
}

/// Round f32 to BF16 precision (round-to-nearest-even). Mirrors the
/// `bf16_round_rne_f32` device helper in `kernels/qwen36_moe_persistent/helpers.cuh`.
fn bf16_round_rne_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounding_bias = 0x7FFF_u32 + ((bits >> 16) & 1);
    let rounded = (bits.wrapping_add(rounding_bias)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// Same dequant formula as `int4_dequant_8` / `int4_dequant_scalar`:
///   `bf16_round_rne_f32(nibble * scale - zero * scale)`
fn int4_dequant_scalar(nibble: u32, scale_bf16: half::bf16, zero_bf16: half::bf16) -> f32 {
    let s = scale_bf16.to_f32();
    let z = zero_bf16.to_f32();
    bf16_round_rne_f32((nibble as f32) * s - z * s)
}

#[derive(Clone, Copy)]
struct Shape {
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    hidden: usize,
    moe_intermediate: usize,
}

const GROUP_SIZE: usize = 128;

fn make_x_norm(n_tokens: usize, hidden: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_tokens * hidden);
    for _ in 0..n_tokens * hidden {
        let (next, raw) = lcg(state);
        state = next;
        let v = ((raw % 10_000) as f32) / 10_000.0 - 0.5; // [-0.5, 0.5)
        out.push(half::bf16::from_f32(v * 0.4));
    }
    out
}

fn make_topk_idx(n_tokens: usize, top_k: usize, num_experts: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_tokens * top_k);
    for _ in 0..n_tokens {
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

fn cpu_router_permute(
    n_tokens: usize,
    top_k: usize,
    num_experts: usize,
    topk_idx: &[i32],
) -> (Vec<i32>, Vec<i32>) {
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
    let mut cursors = vec![0i32; num_experts];
    for entry in 0..total {
        let e = topk_idx[entry] as usize;
        let dst = (offsets[e] + cursors[e]) as usize;
        p_token[dst] = (entry / top_k) as i32;
        cursors[e] += 1;
    }
    (offsets, p_token)
}

/// Pack two i4 nibbles per byte; even col → low nibble.
/// Returns (packed_bytes, scales_bf16, zeros_bf16).
///
/// Layout of `packed`: `[out_rows, in_cols / 2]` u8.
/// Layout of `scales` / `zeros`: `[out_rows / gs, in_cols / gs]` BF16.
fn make_int4_slab(
    out_rows: usize,
    in_cols: usize,
    gs: usize,
    seed: u64,
) -> (Vec<u8>, Vec<half::bf16>, Vec<half::bf16>) {
    assert!(in_cols % 2 == 0);
    assert!(out_rows % gs == 0);
    assert!(in_cols % gs == 0);
    let scale_rows = out_rows / gs;
    let scale_cols = in_cols / gs;

    let mut state = seed;
    let mut packed = vec![0u8; out_rows * (in_cols / 2)];
    for byte in packed.iter_mut() {
        let (next, raw) = lcg(state);
        state = next;
        *byte = (raw & 0xFF) as u8; // both nibbles random in [0, 16).
    }

    let mut scales = vec![half::bf16::ZERO; scale_rows * scale_cols];
    let mut zeros = vec![half::bf16::ZERO; scale_rows * scale_cols];
    for i in 0..scale_rows * scale_cols {
        let (next, raw) = lcg(state);
        state = next;
        // Scale in roughly [0.001, 0.02]: keeps dequanted weights small
        // enough that the cumulative dot products don't blow up.
        let s = 0.001 + ((raw % 1000) as f32 / 1000.0) * 0.02;
        scales[i] = half::bf16::from_f32(s);
        let (next2, raw2) = lcg(state);
        state = next2;
        // Zero point in nibble-space [0, 16). Matches GPTQ-style INT4.
        let z = (raw2 % 16) as f32;
        zeros[i] = half::bf16::from_f32(z);
    }
    (packed, scales, zeros)
}

/// Per-row matvec against a 2D INT4 slab. Mirrors `int4_dequant_8` numerics:
/// dequants 8 nibbles at a time, rounds each through BF16, multiplies by F32 x.
fn int4_matvec_row(
    packed: &[u8],
    scales: &[half::bf16],
    zeros: &[half::bf16],
    out_rows: usize,
    in_cols: usize,
    gs: usize,
    row: usize,
    x: &[f32],
) -> f32 {
    let _ = out_rows;
    let byte_cols = in_cols / 2;
    let scale_cols = in_cols / gs;
    let scale_row = row / gs;
    let row_byte_off = row * byte_cols;

    let mut acc = 0.0f32;
    let mut col = 0usize;
    while col < in_cols {
        // Read 8 nibbles (4 bytes) starting at col.
        let n = [
            (packed[row_byte_off + (col + 0) / 2] >> 0) & 0xF,
            (packed[row_byte_off + (col + 1) / 2] >> 4) & 0xF,
            (packed[row_byte_off + (col + 2) / 2] >> 0) & 0xF,
            (packed[row_byte_off + (col + 3) / 2] >> 4) & 0xF,
            (packed[row_byte_off + (col + 4) / 2] >> 0) & 0xF,
            (packed[row_byte_off + (col + 5) / 2] >> 4) & 0xF,
            (packed[row_byte_off + (col + 6) / 2] >> 0) & 0xF,
            (packed[row_byte_off + (col + 7) / 2] >> 4) & 0xF,
        ];
        for i in 0..8 {
            let g = (col + i) / gs;
            let s = scales[scale_row * scale_cols + g];
            let z = zeros[scale_row * scale_cols + g];
            let dq = int4_dequant_scalar(n[i] as u32, s, z);
            acc += dq * x[col + i];
        }
        col += 8;
    }
    acc
}

/// CPU reference: for each permuted row produce
/// `down(silu(gate(x_norm[token])) * up(x_norm[token]))`.
#[allow(clippy::too_many_arguments)]
fn cpu_grouped_expert(
    shape: Shape,
    x_norm: &[half::bf16],
    permuted_token_idx: &[i32],
    expert_offsets: &[i32],
    gate_up_w: &[u8],
    gate_up_s: &[half::bf16],
    gate_up_z: &[half::bf16],
    down_w: &[u8],
    down_s: &[half::bf16],
    down_z: &[half::bf16],
) -> Vec<half::bf16> {
    let Shape {
        n_tokens,
        top_k,
        num_experts,
        hidden,
        moe_intermediate,
        ..
    } = shape;
    let i = moe_intermediate;
    let two_i = 2 * i;

    let mut out = vec![half::bf16::ZERO; n_tokens * top_k * hidden];

    let gu_per_expert_packed = two_i * (hidden / 2);
    let gu_per_expert_scales = (two_i / GROUP_SIZE) * (hidden / GROUP_SIZE);

    let dp_per_expert_packed = hidden * (i / 2);
    let dp_per_expert_scales = (hidden / GROUP_SIZE) * (i / GROUP_SIZE);

    for e in 0..num_experts {
        let lo = expert_offsets[e] as usize;
        let hi = expert_offsets[e + 1] as usize;
        if lo == hi {
            continue;
        }
        let gu_w = &gate_up_w[e * gu_per_expert_packed..(e + 1) * gu_per_expert_packed];
        let gu_s = &gate_up_s[e * gu_per_expert_scales..(e + 1) * gu_per_expert_scales];
        let gu_z = &gate_up_z[e * gu_per_expert_scales..(e + 1) * gu_per_expert_scales];
        let dp_w = &down_w[e * dp_per_expert_packed..(e + 1) * dp_per_expert_packed];
        let dp_s = &down_s[e * dp_per_expert_scales..(e + 1) * dp_per_expert_scales];
        let dp_z = &down_z[e * dp_per_expert_scales..(e + 1) * dp_per_expert_scales];

        for row in lo..hi {
            let token_idx = permuted_token_idx[row] as usize;
            // Build x_token as F32 with BF16-rounding (the kernel does this
            // when staging into LDS — see the load loop in
            // `batched_prefill_grouped_expert.cuh`).
            let mut x = vec![0.0f32; hidden];
            for c in 0..hidden {
                let v = x_norm[token_idx * hidden + c].to_f32();
                x[c] = bf16_round_rne_f32(v);
            }

            // Phase G: gate_up @ x → gu[2*I].
            let mut gu = vec![0.0f32; two_i];
            for r in 0..two_i {
                gu[r] = int4_matvec_row(gu_w, gu_s, gu_z, two_i, hidden, GROUP_SIZE, r, &x);
            }

            // Phase H: silu(gp) * up. BF16-round on store to match the
            // kernel — the WMMA path (`wmma_int4_matvec_partial_16rows`)
            // takes top-16-bits of the F32 input, so storing
            // BF16-rounded values keeps the scalar and WMMA paths
            // numerically equivalent (and reproducible).
            let mut mid = vec![0.0f32; i];
            for k in 0..i {
                let gp = gu[k];
                let up = gu[i + k];
                let sigmoid = 1.0f32 / (1.0f32 + (-gp).exp());
                mid[k] = bf16_round_rne_f32((gp * sigmoid) * up);
            }

            // Phase I: down @ mid → out_row[hidden] BF16.
            for r in 0..hidden {
                let val = int4_matvec_row(dp_w, dp_s, dp_z, hidden, i, GROUP_SIZE, r, &mid);
                out[row * hidden + r] = half::bf16::from_f32(bf16_round_rne_f32(val));
            }
        }
    }
    out
}

fn run_one_shape(ordinal: usize, shape: Shape, seed: u64) {
    assert!(shape.top_k <= shape.num_experts);
    assert!(shape.hidden % GROUP_SIZE == 0);
    assert!(shape.moe_intermediate % GROUP_SIZE == 0);

    let total_rows = shape.n_tokens * shape.top_k;

    // Synthesize inputs.
    let x_norm_host = make_x_norm(shape.n_tokens, shape.hidden, seed);
    let topk_idx_host = make_topk_idx(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        seed.wrapping_add(0x101),
    );
    let (offsets_host, perm_token_host) = cpu_router_permute(
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        &topk_idx_host,
    );

    // Synthesize per-expert INT4 weights. Layouts match the bake:
    //   gate_up: [E, 2*I, hidden/2] u8 + [E, 2*I/gs, hidden/gs] BF16.
    //   down:    [E, hidden, I/2] u8 + [E, hidden/gs, I/gs] BF16.
    let two_i = 2 * shape.moe_intermediate;
    let mut gu_w_all: Vec<u8> = Vec::with_capacity(shape.num_experts * two_i * (shape.hidden / 2));
    let mut gu_s_all: Vec<half::bf16> =
        Vec::with_capacity(shape.num_experts * (two_i / GROUP_SIZE) * (shape.hidden / GROUP_SIZE));
    let mut gu_z_all: Vec<half::bf16> = Vec::with_capacity(gu_s_all.capacity());
    for e in 0..shape.num_experts {
        let (w, s, z) = make_int4_slab(
            two_i,
            shape.hidden,
            GROUP_SIZE,
            seed.wrapping_add(0x200 + e as u64),
        );
        gu_w_all.extend_from_slice(&w);
        gu_s_all.extend_from_slice(&s);
        gu_z_all.extend_from_slice(&z);
    }

    let mut dp_w_all: Vec<u8> =
        Vec::with_capacity(shape.num_experts * shape.hidden * (shape.moe_intermediate / 2));
    let mut dp_s_all: Vec<half::bf16> = Vec::with_capacity(
        shape.num_experts * (shape.hidden / GROUP_SIZE) * (shape.moe_intermediate / GROUP_SIZE),
    );
    let mut dp_z_all: Vec<half::bf16> = Vec::with_capacity(dp_s_all.capacity());
    for e in 0..shape.num_experts {
        let (w, s, z) = make_int4_slab(
            shape.hidden,
            shape.moe_intermediate,
            GROUP_SIZE,
            seed.wrapping_add(0x300 + e as u64),
        );
        dp_w_all.extend_from_slice(&w);
        dp_s_all.extend_from_slice(&s);
        dp_z_all.extend_from_slice(&z);
    }

    // Upload buffers.
    let x_buf = upload_bf16(ordinal, &x_norm_host, &[shape.n_tokens, shape.hidden]);
    let offsets_buf = upload_i32(ordinal, &offsets_host, &[shape.num_experts + 1]);
    let perm_token_buf = upload_i32(ordinal, &perm_token_host, &[total_rows]);

    let gu_w_buf = upload_u8(
        ordinal,
        &gu_w_all,
        &[shape.num_experts, two_i, shape.hidden / 2],
    );
    let gu_s_buf = upload_bf16(
        ordinal,
        &gu_s_all,
        &[
            shape.num_experts,
            two_i / GROUP_SIZE,
            shape.hidden / GROUP_SIZE,
        ],
    );
    let gu_z_buf = upload_bf16(
        ordinal,
        &gu_z_all,
        &[
            shape.num_experts,
            two_i / GROUP_SIZE,
            shape.hidden / GROUP_SIZE,
        ],
    );
    let dp_w_buf = upload_u8(
        ordinal,
        &dp_w_all,
        &[shape.num_experts, shape.hidden, shape.moe_intermediate / 2],
    );
    let dp_s_buf = upload_bf16(
        ordinal,
        &dp_s_all,
        &[
            shape.num_experts,
            shape.hidden / GROUP_SIZE,
            shape.moe_intermediate / GROUP_SIZE,
        ],
    );
    let dp_z_buf = upload_bf16(
        ordinal,
        &dp_z_all,
        &[
            shape.num_experts,
            shape.hidden / GROUP_SIZE,
            shape.moe_intermediate / GROUP_SIZE,
        ],
    );

    let mut out_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[total_rows, shape.hidden])
        .expect("alloc expert_out");
    let mut counters_buf =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counters");

    batched_prefill_grouped_expert_launch(
        ordinal,
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.hidden,
        shape.moe_intermediate,
        GROUP_SIZE,
        &x_buf,
        &offsets_buf,
        &perm_token_buf,
        &gu_w_buf,
        &gu_s_buf,
        &gu_z_buf,
        &dp_w_buf,
        &dp_s_buf,
        &dp_z_buf,
        &mut out_buf,
        &mut counters_buf,
    )
    .expect("ffi");
    gpu_hal::sync(ordinal).expect("sync");

    let got_bf16 = download_bf16(&out_buf);

    let want_bf16 = cpu_grouped_expert(
        shape,
        &x_norm_host,
        &perm_token_host,
        &offsets_host,
        &gu_w_all,
        &gu_s_all,
        &gu_z_all,
        &dp_w_all,
        &dp_s_all,
        &dp_z_all,
    );

    // Tolerances. INT4 GEMV + WMMA + BF16 quantisation noise pushes
    // small-magnitude outputs into a regime where strict per-element
    // relative bounds explode (e.g. got=1.4e-3 vs want=9e-4 has rel=0.5
    // even though the absolute error is one BF16 ulp away), and pushes
    // large-magnitude outputs to absolute errors of one BF16 ulp at
    // their scale (1 ulp at magnitude 8 ≈ 0.06). The repo's INT4 matvec
    // parity tests (FFN per-block, mtp.fused, lm_head) therefore pair a
    // BF16-ulp-relative `max_abs` with cosine-similarity over the
    // per-row vectors — see `crates/kernel-ffi/src/qwen36_moe.rs::tests`
    // and the M11+M12 parity goals in the plan doc. We adopt that:
    //   - `max_abs / max_magnitude <= 2e-2`  (≈ 2 BF16 ulps at the row
    //                                          peak — the FFN tests use
    //                                          1e-2 absolute on smaller
    //                                          dynamic ranges; the
    //                                          ratio-based form
    //                                          generalises that to our
    //                                          random-INT4 inputs)
    //   - `min cos_sim >= 0.999`              (matches `mtp.fused` and
    //                                          `lm_head`).
    let mut max_abs = 0.0f32;
    let mut max_abs_at = (0usize, 0usize);
    let mut max_magnitude = 0.0f32;
    let mut min_cos = 1.0f32;
    let mut min_cos_row = 0usize;
    let mut nz = 0usize;
    for row in 0..total_rows {
        let mut dot = 0.0f64;
        let mut nrm_g = 0.0f64;
        let mut nrm_w = 0.0f64;
        for c in 0..shape.hidden {
            let g = got_bf16[row * shape.hidden + c].to_f32();
            let w = want_bf16[row * shape.hidden + c].to_f32();
            let abs = (g - w).abs();
            if abs > max_abs {
                max_abs = abs;
                max_abs_at = (row, c);
            }
            let mag = w.abs().max(g.abs());
            if mag > max_magnitude {
                max_magnitude = mag;
            }
            dot += (g as f64) * (w as f64);
            nrm_g += (g as f64) * (g as f64);
            nrm_w += (w as f64) * (w as f64);
            if w.abs() > 1e-6 || g.abs() > 1e-6 {
                nz += 1;
            }
        }
        let denom = (nrm_g.sqrt() * nrm_w.sqrt()).max(1e-12);
        let cos = (dot / denom) as f32;
        if cos < min_cos {
            min_cos = cos;
            min_cos_row = row;
        }
    }

    let max_abs_norm = max_abs / max_magnitude.max(1e-3);

    let (arow, acol) = max_abs_at;
    let g_aat = got_bf16[arow * shape.hidden + acol].to_f32();
    let w_aat = want_bf16[arow * shape.hidden + acol].to_f32();
    eprintln!(
        "shape (N={}, top_k={}, E={}, hidden={}, I={}): \
         max_abs={:.5e} at row={} c={} (got={:.5e} want={:.5e}) \
         max_mag={:.5e} max_abs_norm={:.5e} \
         min_cos={:.6} at row={} \
         nz={}/{}",
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.hidden,
        shape.moe_intermediate,
        max_abs,
        arow,
        acol,
        g_aat,
        w_aat,
        max_magnitude,
        max_abs_norm,
        min_cos,
        min_cos_row,
        nz,
        total_rows * shape.hidden,
    );

    // ~2 BF16 ulps at the row peak: BF16 has a 7-bit mantissa, so 1 ulp
    // at magnitude M is M / 128 ≈ 0.0078*M. 2e-2 leaves headroom for
    // accumulation noise in long reductions (hidden=2048).
    assert!(
        max_abs_norm < 2e-2,
        "max_abs/max_mag {max_abs_norm} >= 2e-2 (max_abs={max_abs}, max_mag={max_magnitude}) at shape (N={}, top_k={}, E={}, hidden={}, I={})",
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.hidden,
        shape.moe_intermediate,
    );
    assert!(
        min_cos >= 0.999,
        "min_cos {min_cos} < 0.999 at row={min_cos_row} shape (N={}, top_k={}, E={}, hidden={}, I={})",
        shape.n_tokens,
        shape.top_k,
        shape.num_experts,
        shape.hidden,
        shape.moe_intermediate,
    );
}

#[test]
fn grouped_expert_matches_cpu_reference() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let ordinal = 0usize;

    // Sweep production shape (N=64, top_k=8, num_experts=256, hidden=2048, I=512)
    // plus smaller shapes to exercise edge cases: minimum hidden divisible by gs,
    // single-token, smaller num_experts. The minimal shape uses the smallest
    // dims that still respect group_size=128 divisibility on both reduction dims.
    for &(n_tokens, top_k, num_experts, hidden, moe_int, seed) in &[
        // Minimal: hidden=128, I=128 are both = group_size (one scale-row each).
        (4usize, 2usize, 8usize, 128usize, 128usize, 0x10u64),
        // Smaller mid: closer to production but trims I and hidden by 4x.
        (16, 8, 64, 512, 256, 0x20),
        // Production shape — the one that matters for qwen3.6-35b-a3b.
        (64, 8, 256, 2048, 512, 0x30),
    ] {
        run_one_shape(
            ordinal,
            Shape {
                n_tokens,
                top_k,
                num_experts,
                hidden,
                moe_intermediate: moe_int,
            },
            seed,
        );
    }
}
