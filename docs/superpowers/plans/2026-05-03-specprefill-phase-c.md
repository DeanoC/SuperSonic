# SpecPrefill Phase C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end SpecPrefill (arXiv 2502.02789) on Qwen3.5-9B target + Qwen3.5-0.8B draft, single GPU, single batch, greedy decode, HIP backend, `--specprefill-draft-dir` flag.

**Architecture:** Three independently reviewable PRs. PR C1 lands a standalone `lookahead_attention_scores` HIP kernel (no impact on the dense prefill path). PR C2 plumbs `kept_positions: Option<&[u32]>` through `prefill_inner` and adds a Q-row capture hook to the decode path. PR C3 adds a new orchestrator engine `specprefill_engine.rs` (patterned on `qwen35_dflash_engine.rs`), CLI flags, a `LAST_LOGITS` shim on the Qwen3.5 dense path, and an integration test.

**Tech Stack:** Rust 2021, HIP (gfx1100), `gpu-hal`, `kernel-ffi`, `qwen35` (weights/state/scratch), `clap` (CLI), `anyhow`, `half::bf16`.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-03-specprefill-phase-c-design.md`
- Phase A2 numbers: `docs/research/2026-05-03-specprefill-phase-a2-cross-target.md`
- Existing host selection (landed): `crates/runner/src/specprefill.rs`
- Existing RoPE-indirect kernel (landed): `kernels/prefill_helpers.hip:122` + `crates/kernel-ffi/src/prefill_ffi.rs:2861`
- Phase B parity test (template for kernel parity tests): `crates/runner/tests/specprefill_rope_indirect_parity.rs`
- Engine template: `crates/runner/src/qwen35_dflash_engine.rs`

**Hardware budget:** RX 7900 XTX 24 GiB. Target Qwen3.5-9B BF16 ≈ 18 GiB; draft Qwen3.5-0.8B BF16 ≈ 1.6 GiB; KV at keep=0.50 long-prompt ≈ 70 MiB. Tight peak ≈ 21 GiB; `--specprefill-unload-draft` (default off) is the safety valve.

---

# PR C1 — Look-ahead attention kernel + FFI + parity test

This PR is fully self-contained — it adds one HIP kernel, an FFI binding, and a unit test. The dense prefill path is byte-identical to `main` afterwards.

## Task 1: Add the HIP kernel

**Files:**
- Modify: `kernels/prefill_helpers.hip` (add a new templated kernel after the existing `pfx_apply_rope_prefill_indirect_kernel` block at line ~152)

The kernel computes per-row softmax of `Q · Kᵀ` for the last `lookahead_count` query rows of a single full-attention layer, against the prompt-only K cache. No causality (every query row attends to all `kv_len` keys). No V — V is consumed entirely host-side via aggregation.

Layout convention:
- `q`: BF16, contiguous `[lookahead_count, q_heads, head_dim]` — the captured post-RoPE query rows. Row `r` ∈ `[0, lookahead_count)`, head `h` ∈ `[0, q_heads)`.
- `k`: BF16, `[kv_heads, kv_len, head_dim]` (the prefill K cache layout for full-attention layers in `qwen35::state`). Each KV head is shared by `num_kv_groups = q_heads / kv_heads` query heads.
- `scores`: F32, `[q_heads, lookahead_count, kv_len]` — output, post-softmax weights.

One block computes one `(q_head, q_row)` pair with one warp, mirroring the streaming-softmax structure already in `kernels/full_attention.hip:246`.

- [ ] **Step 1: Add the kernel definition**

Append this template after `pfx_apply_rope_prefill_indirect_kernel` (around line 153 of `kernels/prefill_helpers.hip`):

```cpp
// ---- Kernel 2c: per-row softmax(Q · Kᵀ) for SpecPrefill look-ahead ----
// Computes the post-softmax attention weights for the last `lookahead_count`
// query rows of a single full-attention layer against the prompt-only K
// cache. No V, no causality. Used by SpecPrefill (arXiv 2502.02789) to
// harvest the importance signal for token selection.
//
// Grid: (q_heads * lookahead_count) blocks, each one warp (warpSize threads).
// Shared memory: kv_len floats for the per-row exponentials. Caller
// must size dynamic shared mem at launch.
//
// q:      [lookahead_count, q_heads, head_dim]  BF16, post-RoPE
// k:      [kv_heads, kv_len, head_dim]          BF16
// scores: [q_heads, lookahead_count, kv_len]    F32
template <typename T>
__global__ void pfx_lookahead_attention_scores_kernel(
    int q_heads,
    int kv_heads,
    int lookahead_count,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    const T* __restrict__ q,
    const T* __restrict__ k,
    float* __restrict__ scores
) {
    extern __shared__ float lds[]; // kv_len floats
    const int lane = threadIdx.x;
    if (lane >= warpSize) return;

    const int row_idx = blockIdx.x;                  // 0 .. q_heads * lookahead_count
    const int q_row = row_idx % lookahead_count;
    const int q_head = row_idx / lookahead_count;
    if (q_head >= q_heads) return;
    const int kv_head = q_head / num_kv_groups;

    // q layout: [lookahead_count, q_heads, head_dim]
    const T* q_ptr = q + (static_cast<size_t>(q_row) * q_heads + q_head) * head_dim;
    // k layout: [kv_heads, kv_len, head_dim]
    const T* k_head = k + static_cast<size_t>(kv_head) * kv_len * head_dim;
    // scores layout: [q_heads, lookahead_count, kv_len]
    float* out_row = scores +
        (static_cast<size_t>(q_head) * lookahead_count + q_row) * kv_len;

    // Pass 1: compute scaled dot-products into LDS, track running max.
    __shared__ float shared_max;
    if (lane == 0) shared_max = -INFINITY;
    __syncthreads();

    for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
        const T* k_row = k_head + k_pos * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += warpSize) {
            partial += pfx_to_float(q_ptr[d]) * pfx_to_float(k_row[d]);
        }
        // Warp-reduction (existing supersonic_qwen35_wave_sum lives in
        // full_attention.hip; we keep prefill_helpers.hip self-contained
        // by inlining the reduction here.)
        partial += __shfl_xor(partial, 16);
        partial += __shfl_xor(partial, 8);
        partial += __shfl_xor(partial, 4);
        partial += __shfl_xor(partial, 2);
        partial += __shfl_xor(partial, 1);
        const float score = partial * scale;
        if (lane == 0) {
            lds[k_pos] = score;
            if (score > shared_max) shared_max = score;
        }
        __syncthreads();
    }

    // Pass 2: exponentiate (centred on shared_max), accumulate denom.
    __shared__ float shared_denom;
    if (lane == 0) shared_denom = 0.0f;
    __syncthreads();
    for (int k_pos = lane; k_pos < kv_len; k_pos += warpSize) {
        const float e = expf(lds[k_pos] - shared_max);
        lds[k_pos] = e;
        atomicAdd(&shared_denom, e);
    }
    __syncthreads();

    // Pass 3: normalise and store.
    const float inv_denom = shared_denom > 0.0f ? 1.0f / shared_denom : 0.0f;
    for (int k_pos = lane; k_pos < kv_len; k_pos += warpSize) {
        out_row[k_pos] = lds[k_pos] * inv_denom;
    }
}
```

- [ ] **Step 2: Verify the file still compiles syntactically**

Run: `grep -n "pfx_lookahead_attention_scores_kernel" kernels/prefill_helpers.hip`
Expected: one hit in the new kernel block.

- [ ] **Step 3: Commit**

```bash
git add kernels/prefill_helpers.hip
git commit -m "specprefill: add lookahead_attention_scores HIP kernel"
```

## Task 2: Add the bridge function and extern "C" entry

**Files:**
- Modify: `kernels/prefill_helpers_bridge.cpp` (template `lookahead_attention_scores_device` after `apply_rope_prefill_indirect_device` at line ~85, then extern "C" wrapper after `supersonic_qwen35_hip_apply_rope_prefill_indirect` at line ~360)

- [ ] **Step 1: Add the templated launcher**

Append after the closing brace of `apply_rope_prefill_indirect_device` (line ~85 in the namespace block):

```cpp
// ---- lookahead_attention_scores (SpecPrefill — arXiv 2502.02789) ----

template <typename T>
int lookahead_attention_scores_device(int device_ordinal,
                                      int q_heads, int kv_heads,
                                      int lookahead_count, int kv_len, int head_dim,
                                      float scale,
                                      const void* q, const void* k, void* scores) {
    if (q_heads <= 0 || kv_heads <= 0 || lookahead_count <= 0 || kv_len <= 0 || head_dim <= 0) {
        return 318; // invalid shape
    }
    if (q_heads % kv_heads != 0) {
        return 319; // q_heads must be a multiple of kv_heads
    }
    ScopedHipDevice scoped(device_ordinal);
    const int num_kv_groups = q_heads / kv_heads;
    const dim3 grid(static_cast<unsigned int>(q_heads * lookahead_count));
    // gfx1100 HIP defaults to wave32. Match the existing prefill
    // kernels (full_attention.hip's `if (lane >= warpSize) return;`)
    // by launching one warp per block. `warpSize` is a HIP-defined
    // constant available inside .hip TUs.
    const dim3 block(warpSize);
    const size_t shared_bytes = static_cast<size_t>(kv_len) * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pfx_lookahead_attention_scores_kernel<T>),
        grid, block, shared_bytes, 0,
        q_heads, kv_heads, lookahead_count, kv_len, head_dim, num_kv_groups,
        scale,
        static_cast<const T*>(q),
        static_cast<const T*>(k),
        static_cast<float*>(scores));
    if (hipGetLastError() != hipSuccess) return 316;
    if (hipDeviceSynchronize() != hipSuccess) return 317;
    return 0;
}
```

- [ ] **Step 2: Add the extern "C" entry**

Append after the closing brace of `supersonic_qwen35_hip_apply_rope_prefill_indirect` (line ~360 in the extern "C" block):

```cpp
extern "C" int supersonic_qwen35_hip_lookahead_attention_scores(
    int dtype, size_t device_ordinal,
    size_t q_heads, size_t kv_heads,
    size_t lookahead_count, size_t kv_len, size_t head_dim,
    float scale,
    const void* q, const void* k, void* scores
) {
    switch (dtype) {
    case 0: return lookahead_attention_scores_device<half>(
                static_cast<int>(device_ordinal),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(lookahead_count), static_cast<int>(kv_len),
                static_cast<int>(head_dim), scale, q, k, scores);
    case 2: return lookahead_attention_scores_device<hip_bfloat16>(
                static_cast<int>(device_ordinal),
                static_cast<int>(q_heads), static_cast<int>(kv_heads),
                static_cast<int>(lookahead_count), static_cast<int>(kv_len),
                static_cast<int>(head_dim), scale, q, k, scores);
    default: return 320;
    }
}
```

(F32 not exposed — the prefill K cache is BF16 in the runtime; F16 retained because all other prefill_helpers entries support it.)

- [ ] **Step 3: Add a CUDA stub**

Modify `kernels/prefill_helpers_bridge_cuda.cu`. Find the `supersonic_qwen35_hip_apply_rope_prefill_indirect` extern at line 642 and append after the closing brace of its definition:

```cpp
extern "C" int supersonic_qwen35_hip_lookahead_attention_scores(
    int /*dtype*/, size_t /*device_ordinal*/,
    size_t /*q_heads*/, size_t /*kv_heads*/,
    size_t /*lookahead_count*/, size_t /*kv_len*/, size_t /*head_dim*/,
    float /*scale*/,
    const void* /*q*/, const void* /*k*/, void* /*scores*/
) {
    return 99; // not implemented for CUDA in Phase C
}
```

- [ ] **Step 4: Add a Metal stub**

Modify `crates/kernel-ffi/src/metal_link_stubs.cc`. Find the existing `apply_rope_prefill_indirect` Metal stub and add an analogous one for `lookahead_attention_scores`:

```cpp
extern "C" int supersonic_qwen35_hip_lookahead_attention_scores(
    int, size_t, size_t, size_t, size_t, size_t, size_t, float,
    const void*, const void*, void*
) {
    return 99;
}
```

(Locate the file with `grep -n "apply_rope_prefill_indirect" crates/kernel-ffi/src/metal_link_stubs.cc`. Insert in the same style as that stub.)

- [ ] **Step 5: Commit**

```bash
git add kernels/prefill_helpers_bridge.cpp kernels/prefill_helpers_bridge_cuda.cu crates/kernel-ffi/src/metal_link_stubs.cc
git commit -m "specprefill: HIP bridge + CUDA/Metal stubs for lookahead_attention_scores"
```

## Task 3: Add the Rust FFI binding

**Files:**
- Modify: `crates/kernel-ffi/src/prefill_ffi.rs` (extern declaration near line 682; pub fn after `apply_rope_prefill_indirect` at line ~2912)

- [ ] **Step 1: Add the extern declaration**

Find the `extern "C" { ... }` block that contains `supersonic_qwen35_hip_apply_rope_prefill_indirect` (declaration at line ~682) and append:

```rust
fn supersonic_qwen35_hip_lookahead_attention_scores(
    dtype: c_int,
    device_ordinal: usize,
    q_heads: usize,
    kv_heads: usize,
    lookahead_count: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    q: *const c_void,
    k: *const c_void,
    scores: *mut c_void,
) -> c_int;
```

- [ ] **Step 2: Add the safe wrapper**

After `apply_rope_prefill_indirect` at line ~2912, add:

```rust
/// SpecPrefill (arXiv 2502.02789): per-row softmax(Q · Kᵀ) for the last
/// `lookahead_count` query rows of a single full-attention layer.
/// Computes the importance signal the host-side selection consumes.
///
/// Layouts:
/// - `q`: BF16 `[lookahead_count, q_heads, head_dim]` (post-RoPE)
/// - `k`: BF16 `[kv_heads, kv_len, head_dim]`
/// - `scores`: F32 `[q_heads, lookahead_count, kv_len]` (output)
///
/// `q_heads` must be a multiple of `kv_heads` (GQA broadcasting handled
/// inside the kernel via `num_kv_groups = q_heads / kv_heads`).
pub fn lookahead_attention_scores(
    ordinal: usize,
    dtype: ScalarType,
    q_heads: usize,
    kv_heads: usize,
    lookahead_count: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    scores: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if q_heads == 0 || kv_heads == 0 || lookahead_count == 0 || kv_len == 0 || head_dim == 0 {
        return Err(ffi_error(
            "lookahead_attention_scores: all dimensions must be > 0".into(),
        ));
    }
    if q_heads % kv_heads != 0 {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: q_heads ({q_heads}) must be a multiple of kv_heads ({kv_heads})"
        )));
    }
    let expected_q = lookahead_count * q_heads * head_dim;
    if q.elem_count() < expected_q {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: q has {} elems, expected >= {}",
            q.elem_count(),
            expected_q
        )));
    }
    let expected_k = kv_heads * kv_len * head_dim;
    if k.elem_count() < expected_k {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: k has {} elems, expected >= {}",
            k.elem_count(),
            expected_k
        )));
    }
    if scores.dtype() != ScalarType::F32 {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: scores must be ScalarType::F32, got {:?}",
            scores.dtype()
        )));
    }
    let expected_scores = q_heads * lookahead_count * kv_len;
    if scores.elem_count() < expected_scores {
        return Err(ffi_error(format!(
            "lookahead_attention_scores: scores has {} elems, expected >= {}",
            scores.elem_count(),
            expected_scores
        )));
    }
    let status = unsafe {
        supersonic_qwen35_hip_lookahead_attention_scores(
            dtype.kernel_dtype_code(),
            ordinal,
            q_heads,
            kv_heads,
            lookahead_count,
            kv_len,
            head_dim,
            scale,
            q.as_ptr(),
            k.as_ptr(),
            scores.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(ffi_error(format!(
            "lookahead_attention_scores failed: {status}"
        )));
    }
    Ok(())
}
```

- [ ] **Step 3: Run the lib build**

Run: `cargo check -p kernel-ffi --release`
Expected: clean build (or unrelated warnings).

- [ ] **Step 4: Commit**

```bash
git add crates/kernel-ffi/src/prefill_ffi.rs
git commit -m "specprefill: Rust FFI for lookahead_attention_scores"
```

## Task 4: Parity test against a CPU softmax reference

**Files:**
- Create: `crates/runner/tests/specprefill_lookahead_attention_parity.rs`

The test mirrors `specprefill_rope_indirect_parity.rs` (skipped when HIP isn't compiled). It builds deterministic Q/K tensors on the GPU, runs the kernel, and compares against a CPU softmax reference.

- [ ] **Step 1: Write the failing test**

```rust
//! Parity test for `lookahead_attention_scores` (SpecPrefill — Phase C).
//!
//! Asserts the GPU kernel produces softmax-of-(Q·Kᵀ) within 1e-5 of a
//! straight CPU reference, AND that each (q_head, q_row) row sums to 1.0
//! (the softmax invariant). Skipped silently when HIP is not compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi::lookahead_attention_scores;

fn make_bf16_buf(
    ordinal: usize,
    shape: &[usize],
    seed_offset: usize,
) -> GpuBuffer {
    let total: usize = shape.iter().product();
    let host: Vec<half::bf16> = (0..total)
        .map(|i| {
            let v = ((i + seed_offset) as f32 * 0.0019).sin() * 0.4 + 0.1;
            half::bf16::from_f32(v)
        })
        .collect();
    let mut buf =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, shape).expect("alloc bf16");
    let bytes = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2)
    };
    gpu_hal::copy_h2d(ordinal, buf.as_mut_ptr(), bytes.as_ptr() as *const _, bytes.len())
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
    let head_dim = 32usize;
    let lookahead_count = 5usize;
    let kv_len = 17usize;
    let num_kv_groups = q_heads / kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = make_bf16_buf(
        ordinal,
        &[lookahead_count, q_heads, head_dim],
        0,
    );
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

    // Per-element max abs error (BF16 inputs imply ~1e-3 of dot-product
    // noise; we expect << 1e-3 after the softmax averages it out).
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
```

- [ ] **Step 2: Run the test (it should pass on HIP)**

Run: `cargo test -p runner --release --test specprefill_lookahead_attention_parity -- --nocapture`
Expected on HIP: `test result: ok. 1 passed`. Expected without HIP: `skipped: HIP backend not compiled`, test passes.

If it fails, the kernel math is wrong — debug by reducing `kv_len` to 1 or 2 and printing the LDS values inline.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/tests/specprefill_lookahead_attention_parity.rs
git commit -m "specprefill: parity test for lookahead_attention_scores kernel"
```

**End of PR C1.** Open the PR, await Codex review (per `~/.claude/projects/-home-deano-projects-SuperSonic/memory/reference_codex_review_bot.md`), address P1 issues before merge.

---

# PR C2 — `prefill_with_lookahead_attention` + `prefill_kept` plumbing

This PR adds two public wrappers on `crates/runner/src/prefill_engine.rs` and the layer-level Q-row capture hook on `DecodeEngine`. The dense prefill path remains byte-identical (we route a new optional parameter through `prefill_inner`; when it's `None` the code path is unchanged).

## Task 5: Extend `prefill_inner` with a `kept_positions` parameter

**Files:**
- Modify: `crates/runner/src/prefill_engine.rs`

The change:
1. Add a new optional `kept_positions: Option<&[u32]>` parameter to `prefill_inner` (line 975).
2. Thread it down through `prefill_full_attention_layer` (line 1914) so the per-chunk `apply_rope_prefill` calls swap to `apply_rope_prefill_indirect` when kept positions are present.
3. The chunked-prefill loop already iterates `chunk_start..chunk_start+chunk_len` over the *compacted* sequence when `kept_positions.is_some()` — the prompt-token uploads in line 1098 use `&prompt_ids[chunk_start..]` directly, so we must instead upload `prompt_ids[kept_positions[chunk_start..chunk_start+chunk_len]]`.

- [ ] **Step 1: Add the parameter to `prefill_inner`**

Modify the signature at line 975:

```rust
fn prefill_inner(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
    trace_layers: bool,
    debug_linear_layer: Option<usize>,
    tap_layers: Option<&[usize]>,
    target_nll: Option<(usize, &[u32])>,
    kept_positions: Option<&[u32]>,  // NEW
) -> Result<PrefillResult> {
```

Update all three existing callers (`prefill`, `prefill_with_taps`, `prefill_with_target_nll`) to pass `None` for the new parameter — keep their public signatures unchanged.

- [ ] **Step 2: Validate `kept_positions` early**

Right after the `let seq_len = prompt_ids.len();` line, add:

```rust
let seq_len = if let Some(kept) = kept_positions {
    if kept.is_empty() {
        return Err(anyhow::anyhow!(
            "prefill_inner: kept_positions is empty"
        ));
    }
    let max_pos = kept.iter().copied().max().unwrap() as usize;
    if max_pos >= prompt_ids.len() {
        return Err(anyhow::anyhow!(
            "prefill_inner: kept_positions[{}]={} out of range (prompt_len={})",
            kept.iter().position(|&p| p as usize == max_pos).unwrap(),
            max_pos,
            prompt_ids.len()
        ));
    }
    // Strict ascending uniqueness — the selection layer already
    // guarantees this; we re-check defensively.
    for w in kept.windows(2) {
        if w[0] >= w[1] {
            return Err(anyhow::anyhow!(
                "prefill_inner: kept_positions must be strictly ascending"
            ));
        }
    }
    kept.len()
} else {
    prompt_ids.len()
};
```

(The existing `let seq_len = prompt_ids.len();` becomes the `else` branch above; remove the original line.)

- [ ] **Step 3: Compact the chunk's token IDs**

In the chunk loop at line 1098 (`let chunk_ids = &prompt_ids[chunk_start..chunk_start + chunk_len];`), replace with:

```rust
let chunk_ids_storage: Vec<u32>;
let chunk_ids: &[u32] = if let Some(kept) = kept_positions {
    chunk_ids_storage = kept[chunk_start..chunk_start + chunk_len]
        .iter()
        .map(|&p| prompt_ids[p as usize])
        .collect();
    &chunk_ids_storage
} else {
    &prompt_ids[chunk_start..chunk_start + chunk_len]
};
```

- [ ] **Step 4: Pass kept_positions slice to `prefill_full_attention_layer`**

At line 1131 (the `prefill_full_attention_layer(...)` call site inside the layer loop), add an extra argument:

```rust
prefill_full_attention_layer(
    weights,
    state,
    rotary,
    &mut scratch,
    config,
    idx,
    chunk_len,
    chunk_start,
    ordinal,
    kv_chunk_size,
    /* commit_kv_filled */ true,
    kept_positions.map(|k| &k[chunk_start..chunk_start + chunk_len]),  // NEW
)?;
```

Same change at the second call site (line ~1688) — just add `None` if the second site is in a non-spec path; check by reading the surrounding context. If the second site is also in the chunk loop, pass the same slice.

- [ ] **Step 5: Commit (intermediate, no behaviour change yet)**

```bash
git add crates/runner/src/prefill_engine.rs
git commit -m "specprefill: thread kept_positions through prefill_inner (no behaviour change)"
```

## Task 6: Route `apply_rope_prefill_indirect` through `prefill_full_attention_layer`

**Files:**
- Modify: `crates/runner/src/prefill_engine.rs:1914` (the function signature and the two RoPE call sites at lines 2074 and 2087)

- [ ] **Step 1: Update the signature**

```rust
fn prefill_full_attention_layer(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    chunk_len: usize,
    chunk_start: usize,
    ordinal: usize,
    kv_chunk_size: usize,
    commit_kv_filled: bool,
    kept_positions_chunk: Option<&[u32]>,  // NEW
) -> Result<()> {
```

- [ ] **Step 2: Build the GPU pos_ids buffer once at the top of the function (only when kept_positions_chunk is Some)**

After the existing local declarations (around line 1951), add:

```rust
let pos_ids_buf: Option<GpuBuffer> = if let Some(kept) = kept_positions_chunk {
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[kept.len()])
        .map_err(|e| anyhow::anyhow!("layer {idx} pos_ids alloc: {e}"))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(kept.as_ptr() as *const u8, kept.len() * 4)
    };
    gpu_hal::copy_h2d(ordinal, buf.as_mut_ptr(), bytes.as_ptr() as *const _, bytes.len())
        .map_err(|e| anyhow::anyhow!("layer {idx} pos_ids upload: {e}"))?;
    Some(buf)
} else {
    None
};
```

- [ ] **Step 3: Replace the Q RoPE call at line 2074**

Replace:

```rust
prefill_ffi::apply_rope_prefill(
    ordinal,
    ScalarType::BF16,
    chunk_len,
    num_q_heads,
    head_dim,
    rotary_dim,
    &rotary.cos,
    &rotary.sin,
    chunk_start,
    &mut query_buf,
)
.map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE: {e}"))?;
```

with:

```rust
if let Some(pos_ids) = pos_ids_buf.as_ref() {
    prefill_ffi::apply_rope_prefill_indirect(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        num_q_heads,
        head_dim,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        pos_ids,
        &mut query_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE (indirect): {e}"))?;
} else {
    prefill_ffi::apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        chunk_len,
        num_q_heads,
        head_dim,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        chunk_start,
        &mut query_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE: {e}"))?;
}
```

- [ ] **Step 4: Same swap at the K RoPE call (line 2087)**

Replace the second `apply_rope_prefill` call analogously, using `num_kv_heads` and `&mut scratch.proj_buf2`.

- [ ] **Step 5: Run the existing dense parity check**

Run: `cargo test -p runner --release --test specprefill_rope_indirect_parity`
Expected: `2 passed; 0 failed` (the existing tests pin the indirect-vs-dense identity behaviour and don't change here, but we want to confirm we didn't break the build).

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/prefill_engine.rs
git commit -m "specprefill: prefill_full_attention_layer dispatches RoPE-indirect on kept_positions"
```

## Task 7: Add the `prefill_kept` public wrapper

**Files:**
- Modify: `crates/runner/src/prefill_engine.rs` (add after `prefill_with_target_nll` at line 973)

- [ ] **Step 1: Add the wrapper**

```rust
/// SpecPrefill (arXiv 2502.02789) target sparse prefill: like `prefill`,
/// but consumes a sorted ascending `kept_positions` slice. The compacted
/// embedding sequence is `prompt_ids[kept_positions[i]] for i in 0..len`,
/// each token rotates by its ORIGINAL prompt position via the
/// RoPE-indirect kernel (Phase B), and the lower-triangular causal mask
/// over the compacted sequence is exactly the right semantics — Phase B
/// parity tests pin this.
///
/// Post-condition: `kv_filled` on every full-attention layer equals
/// `kept_positions.len()`. The caller's decode-position cursor must
/// nonetheless start at `prompt_ids.len()` (the original prompt's last
/// position + 1), NOT `kept_positions.len()`.
pub fn prefill_kept(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    kept_positions: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    kv_fp8: bool,
    use_4b_kernel: bool,
) -> Result<PrefillResult> {
    prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        kv_fp8,
        use_4b_kernel,
        false,
        None,
        None,
        None,
        Some(kept_positions),
    )
}
```

- [ ] **Step 2: Build**

Run: `cargo check -p runner --release`
Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/src/prefill_engine.rs
git commit -m "specprefill: prefill_kept public wrapper"
```

## Task 8: Add `decode_step_with_query_capture` to `DecodeEngine`

**Files:**
- Modify: `crates/runner/src/decode_engine.rs`

The hook captures the post-RoPE Q row at each requested full-attention layer during a single decode step. We don't need K (the speculator's K cache after this decode step is irrelevant — the look-ahead kernel only attends against the prompt-only K).

- [ ] **Step 1: Locate the existing decode-step entrypoint**

Run: `grep -n "fn decode_step_with_taps_kernel\|fn decode_step\b" crates/runner/src/decode_engine.rs`

The reference implementation is `decode_step_with_taps_kernel` (used by DFlash) — it runs a single decode step and exports per-layer hidden states. We add a sibling that exports per-layer post-RoPE Q rows.

- [ ] **Step 2: Sketch (read first, before writing)**

Read 200 lines starting at the line `decode_step_with_taps_kernel` returns. Identify:
- Where the Q projection and RoPE happen for each layer (look for `apply_rope_decode` or similar).
- The per-layer `query_buf` in scratch.

- [ ] **Step 3: Write the new method**

Add after `decode_step_with_taps_kernel`:

```rust
/// Run one decode step on `token_id`. After RoPE for each requested
/// full-attention layer in `layers`, copy the post-RoPE Q row
/// (`[num_q_heads, head_dim]` BF16) to a host buffer. Used by
/// SpecPrefill (arXiv 2502.02789) to harvest the importance signal.
///
/// Returns `(logits, query_rows)` where `query_rows[i]` is the captured
/// row for `layers[i]`, length `num_q_heads * head_dim` BF16 bytes.
pub fn decode_step_with_query_capture(
    &mut self,
    token_id: u32,
    seqlen_offset: usize,
    layers: &[usize],
) -> Result<(Vec<f32>, Vec<Vec<u8>>)> {
    // [reuse decode_step_with_taps_kernel's structure — the only
    // difference is what is captured. Copy that function and replace
    // the tap-row capture with a Q-row capture immediately after the
    // post-RoPE query_buf is finalised in the layer.]
    ...
}
```

(The actual implementation reuses the body of `decode_step_with_taps_kernel` and just changes the capture site. Confirm by reading the existing function — the new method's diff against it should be ~30 lines.)

- [ ] **Step 4: Build**

Run: `cargo check -p runner --release`
Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/decode_engine.rs
git commit -m "specprefill: decode_step_with_query_capture for speculator look-ahead Q capture"
```

## Task 9: Add `prefill_with_lookahead_attention`

**Files:**
- Modify: `crates/runner/src/prefill_engine.rs`

This is the speculator-side wrapper. It runs dense prefill on the full prompt, then runs `lookahead_count - 1` decode steps with Q capture, then launches the `lookahead_attention_scores` kernel per full-attention layer.

- [ ] **Step 1: Add the result struct**

Near the existing `PrefillResult` definition (search for `pub struct PrefillResult`):

```rust
/// Result of `prefill_with_lookahead_attention`.
pub struct PrefillWithLookaheadResult {
    pub base: PrefillResult,
    /// Per full-attention layer (in layer-index order, NOT layer-list
    /// order): F32 `[num_q_heads, lookahead_count, kv_len]` flattened.
    /// `kv_len` equals the prompt length (the K cache is captured at
    /// the prompt boundary; look-ahead decode K writes are ignored).
    pub layer_scores: Vec<Vec<f32>>,
    /// The lookahead_count value the kernel was launched with — equals
    /// `lookahead + 1` (the last prompt token + N look-ahead tokens).
    pub lookahead_count: usize,
}
```

- [ ] **Step 2: Add the wrapper**

```rust
/// SpecPrefill (arXiv 2502.02789) speculator-side prefill: dense prefill
/// on the full prompt, plus `lookahead_count - 1` decode steps with Q
/// capture, plus per-full-attention-layer `lookahead_attention_scores`
/// launches that produce the `[num_q_heads, lookahead_count, kv_len]`
/// score tensors the host-side aggregator consumes.
///
/// `lookahead_count` is the number of query rows to harvest, which
/// equals `lookahead + 1` (the last prompt token's row + `lookahead`
/// teacher-free decode rows).
pub fn prefill_with_lookahead_attention(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
    prefill_chunk_size: usize,
    use_4b_kernel: bool,
    lookahead_count: usize,
    decode_engine: &mut crate::decode_engine::DecodeEngine,
) -> Result<PrefillWithLookaheadResult> {
    if lookahead_count < 1 {
        return Err(anyhow::anyhow!(
            "prefill_with_lookahead_attention: lookahead_count must be >= 1"
        ));
    }
    let prompt_len = prompt_ids.len();
    let config = &weights.config;

    // 1. Dense prefill on the full prompt.
    let base = prefill_inner(
        weights,
        state,
        rotary,
        prompt_ids,
        ordinal,
        kv_chunk_size,
        prefill_chunk_size,
        false,
        use_4b_kernel,
        false,
        None,
        None,
        None,
        None,
    )?;

    // 2. Identify full-attention layer indices in source order.
    let full_layer_idxs: Vec<usize> = (0..config.num_hidden_layers)
        .filter(|&i| config.is_full_attention(i))
        .collect();
    if full_layer_idxs.is_empty() {
        return Err(anyhow::anyhow!(
            "prefill_with_lookahead_attention: no full-attention layers in the speculator"
        ));
    }

    // 3. Capture the last prompt position's Q row from the dense prefill
    //    workspace. Implementation note: prefill_inner currently does not
    //    expose post-RoPE Q rows. We re-derive them by running one extra
    //    decode_step_with_query_capture at seqlen_offset = prompt_len - 1
    //    *but* that would write a duplicate K row. Cleaner: borrow Q rows
    //    from `state.layers[idx].full.query_last_row` IF such a field
    //    exists; otherwise add it to ModelState and have prefill capture
    //    the last chunk's last query row before computing attention.
    //
    // For Phase C we add a post-RoPE Q-row capture to the last-chunk path
    // of prefill_full_attention_layer when `last_q_capture: bool` is set.
    //
    // [Sub-task: extend ModelState with `last_full_attn_query: Option<Vec<u8>>` per
    // full-attention layer. Capture in prefill_full_attention_layer when
    // chunk_start + chunk_len == prompt_len AND a flag is set.]

    // 4. Run lookahead_count - 1 decode steps. Greedy-sample from the
    //    previous logits, capture per-full-attention-layer Q rows.
    let mut next_id: u32 = greedy_argmax(&base.logits);
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    let q_row_bytes = config.num_attention_heads * config.head_dim * elem_bytes;
    let mut per_layer_q_rows: Vec<Vec<Vec<u8>>> =
        vec![Vec::with_capacity(lookahead_count); full_layer_idxs.len()];
    // Push the prompt's last-row Q (captured in step 3).
    for (slot, &li) in full_layer_idxs.iter().enumerate() {
        per_layer_q_rows[slot].push(state.take_last_full_attn_query(li)?);
    }
    for step in 0..(lookahead_count - 1) {
        let (logits, q_rows) = decode_engine.decode_step_with_query_capture(
            next_id,
            prompt_len + step,
            &full_layer_idxs,
        )?;
        next_id = greedy_argmax(&logits);
        for (slot, row) in q_rows.into_iter().enumerate() {
            assert_eq!(row.len(), q_row_bytes);
            per_layer_q_rows[slot].push(row);
        }
    }

    // 5. Per-layer lookahead_attention_scores launch.
    let q_heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut layer_scores: Vec<Vec<f32>> = Vec::with_capacity(full_layer_idxs.len());

    for (slot, &li) in full_layer_idxs.iter().enumerate() {
        // Build a single contiguous Q buffer [lookahead_count, q_heads, head_dim] BF16
        let mut q_host = Vec::with_capacity(lookahead_count * q_row_bytes);
        for row in &per_layer_q_rows[slot] {
            q_host.extend_from_slice(row);
        }
        let mut q_buf = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[lookahead_count, q_heads, head_dim],
        )
        .map_err(|e| anyhow::anyhow!("q_buf alloc: {e}"))?;
        gpu_hal::copy_h2d(ordinal, q_buf.as_mut_ptr(), q_host.as_ptr() as *const _, q_host.len())
            .map_err(|e| anyhow::anyhow!("q_buf h2d: {e}"))?;

        // K is the layer's full-attention K cache, prompt-only (kv_len = prompt_len).
        let k_buf = state.layers[li]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {li}: missing full-attention state"))?
            .k_cache
            .clone(); // GpuBuffer is reference-shared; clone is cheap.

        let mut scores = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[q_heads, lookahead_count, prompt_len],
        )
        .map_err(|e| anyhow::anyhow!("scores alloc: {e}"))?;

        prefill_ffi::lookahead_attention_scores(
            ordinal,
            ScalarType::BF16,
            q_heads,
            kv_heads,
            lookahead_count,
            prompt_len,
            head_dim,
            scale,
            &q_buf,
            &k_buf,
            &mut scores,
        )?;

        let mut host = vec![0.0_f32; q_heads * lookahead_count * prompt_len];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, host.len() * 4)
        };
        gpu_hal::copy_d2h(ordinal, bytes.as_mut_ptr() as *mut _, scores.as_ptr(), bytes.len())
            .map_err(|e| anyhow::anyhow!("scores d2h: {e}"))?;
        layer_scores.push(host);
    }

    Ok(PrefillWithLookaheadResult {
        base,
        layer_scores,
        lookahead_count,
    })
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
            if v > acc.1 { (i, v) } else { acc }
        })
        .0 as u32
}
```

(Adjust field names — `state.layers[li].full.k_cache` may live under a different field; read `qwen35::state` to confirm. The structure is unchanged from existing tap-capture call sites.)

- [ ] **Step 4: Add `take_last_full_attn_query` to `ModelState` and capture in `prefill_full_attention_layer`**

Search for `pub struct ModelState` (probably in `crates/qwen35/src/state.rs`). Add:
- A field `last_full_attn_query: Vec<Option<Vec<u8>>>` (one slot per layer, populated only for full-attention layers).
- A method `pub fn take_last_full_attn_query(&mut self, layer: usize) -> Result<Vec<u8>>` that returns the captured row or errors.

In `prefill_full_attention_layer`, after the post-RoPE Q is finalised (immediately after step 6 in the existing code, line ~2099), add:

```rust
// Last-chunk last-row Q capture for SpecPrefill.
if chunk_start + chunk_len == /* total */ {
    let last_row_offset = (chunk_len - 1) * q_dim * elem_bytes;
    let mut host = vec![0u8; q_dim * elem_bytes];
    gpu_hal::copy_d2h(
        ordinal,
        host.as_mut_ptr() as *mut _,
        unsafe { query_buf.as_ptr().add(last_row_offset) },
        host.len(),
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} last-Q d2h: {e}"))?;
    state.last_full_attn_query[idx] = Some(host);
}
```

(`/* total */` here means the prefill's `seq_len` — pass it through as a function parameter or read from `state.kv_filled` after the K-write step. Add a `total_seq_len: usize` parameter to `prefill_full_attention_layer`.)

The capture is unconditional (always populated on the dense prefill path) — the cost is one BF16 copy per full-attention layer per prefill, ~16 KiB total on the 0.8B speculator. Negligible.

- [ ] **Step 5: Build**

Run: `cargo check -p runner --release`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/prefill_engine.rs crates/runner/src/decode_engine.rs crates/qwen35/src/state.rs
git commit -m "specprefill: prefill_with_lookahead_attention + ModelState last-Q capture"
```

## Task 10: Unit test — `kept_positions = None` is bit-identical to current

**Files:**
- Create: `crates/runner/tests/specprefill_prefill_kept_identity.rs`

This test runs `prefill` and `prefill_kept` with `kept_positions = [0..T]` on a tiny synthetic checkpoint and asserts the last-step logits are bit-identical. It catches any drift introduced by the new code path.

- [ ] **Step 1: Write the test**

```rust
//! Identity test: prefill_kept with kept_positions = [0..T] must match
//! the dense prefill bit-for-bit. Skipped without HIP and without a
//! local Qwen3.5-0.8B model dir.

use std::path::PathBuf;

use gpu_hal::Backend;

#[test]
fn prefill_kept_identity_matches_dense_on_qwen35_0_8b() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let dir = match std::env::var("SUPERSONIC_QWEN35_0_8B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => PathBuf::from(d),
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_0_8B_DIR unset or missing");
            return;
        }
    };

    // Bring up the speculator's weights + state via the same setup helpers
    // the runtime uses. The two-line setup that loads weights is in
    // qwen35_dflash_engine; copy that pattern.
    use runner::prefill_engine::{prefill, prefill_kept};
    let (weights, mut state, rotary) = runner::test_helpers::load_qwen35_for_test(&dir, 0)
        .expect("load qwen35 for test");
    let mut state2 = state.clone_for_test();
    let prompt_ids: Vec<u32> = (10..42).collect(); // 32 deterministic ids

    let dense = prefill(
        &weights,
        &mut state,
        &rotary,
        &prompt_ids,
        0,
        128,
        0,
        false,
        false,
        false,
        None,
    )
    .expect("dense");

    let kept: Vec<u32> = (0..prompt_ids.len() as u32).collect();
    let kept_pf = prefill_kept(
        &weights,
        &mut state2,
        &rotary,
        &prompt_ids,
        &kept,
        0,
        128,
        0,
        false,
        false,
    )
    .expect("kept");

    assert_eq!(dense.logits.len(), kept_pf.logits.len());
    for (i, (a, b)) in dense.logits.iter().zip(kept_pf.logits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "dense vs kept-identity logits differ at i={i}: a={a} b={b}"
        );
    }
}
```

(`runner::test_helpers::load_qwen35_for_test` and `state.clone_for_test()` may not exist yet. If not, write them as `pub(crate)` helpers in `crates/runner/src/lib.rs` — they're shared with the integration test in PR C3 anyway.)

- [ ] **Step 2: Run and verify**

Run: `SUPERSONIC_QWEN35_0_8B_DIR=/mnt/data/models/Qwen3.5-0.8B cargo test -p runner --release --test specprefill_prefill_kept_identity -- --nocapture`
Expected: `1 passed`.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/tests/specprefill_prefill_kept_identity.rs crates/runner/src/lib.rs
git commit -m "specprefill: identity parity test for prefill_kept (kept = [0..T] matches dense)"
```

**End of PR C2.**

---

# PR C3 — `specprefill_engine.rs` + CLI + integration test

## Task 11: Add CLI flags

**Files:**
- Modify: `crates/runner/src/main.rs` (Cli struct around line 700)

- [ ] **Step 1: Add the flags**

Append to the `Cli` struct (after the dflash flags ~line 702):

```rust
    /// Path to the SpecPrefill (arXiv 2502.02789) draft model directory
    /// (e.g. `/mnt/data/models/Qwen3.5-0.8B`). Presence of this flag
    /// enables sparse target prefill via the speculator's importance
    /// signal. Currently only supported for `--model qwen3.5-9b`.
    #[arg(long)]
    specprefill_draft_dir: Option<PathBuf>,

    /// SpecPrefill keep ratio per chunk: fraction of tokens kept by the
    /// chunked top-K selection. Phase A2 measurements pin 0.50 as the
    /// quality-stable default on Qwen3.5-9B (cossim ≥ 0.927, argmax
    /// match). Range: [0.05, 1.0].
    #[arg(long, default_value = "0.50")]
    specprefill_keep_ratio: f32,

    /// SpecPrefill chunk size for top-K selection (paper §3.4).
    #[arg(long, default_value = "32")]
    specprefill_chunk_size: usize,

    /// SpecPrefill 1-D average-pool window for score smoothing. Must be
    /// odd. Paper uses 5-10.
    #[arg(long, default_value = "5")]
    specprefill_pool_window: usize,

    /// SpecPrefill look-ahead decode steps on the draft (paper §3.3
    /// default 4). Total query rows harvested = lookahead + 1.
    #[arg(long, default_value = "4")]
    specprefill_lookahead: usize,

    /// SpecPrefill always-keep prefix (BOS + system) length.
    #[arg(long, default_value = "4")]
    specprefill_always_keep_prefix: usize,

    /// SpecPrefill always-keep suffix (final query) length.
    #[arg(long, default_value = "4")]
    specprefill_always_keep_suffix: usize,

    /// Free the draft weights after selection runs and before the target
    /// prefill, to claw back ~1.6 GiB on a tight 24 GiB budget.
    #[arg(long, default_value_t = false)]
    specprefill_unload_draft: bool,
```

- [ ] **Step 2: Build**

Run: `cargo check -p runner --release`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/src/main.rs
git commit -m "specprefill: CLI flags (--specprefill-draft-dir et al)"
```

## Task 12: Add `validate_specprefill_flags`

**Files:**
- Modify: `crates/runner/src/policy.rs`

- [ ] **Step 1: Add the validator**

Add after `validate_dflash_flags` (line 83):

```rust
pub(crate) fn validate_specprefill_flags(
    cli: &Cli,
    model_variant: &ModelVariant,
) -> Result<()> {
    let any_specprefill_flag = cli.specprefill_draft_dir.is_some()
        || cli.specprefill_unload_draft;
    if cli.specprefill_draft_dir.is_some() {
        if !matches!(model_variant, ModelVariant::Qwen3_5_9B) {
            anyhow::bail!(
                "--specprefill-draft-dir is only supported on --model qwen3.5-9b in Phase C \
                 (got {model_variant})."
            );
        }
        if cli.batch_size != 1 {
            anyhow::bail!("SpecPrefill requires --batch-size 1");
        }
        if cli.dflash {
            anyhow::bail!("--specprefill-* and --dflash cannot be combined");
        }
        if !(0.05..=1.0).contains(&cli.specprefill_keep_ratio) {
            anyhow::bail!(
                "--specprefill-keep-ratio must be in [0.05, 1.0] (got {})",
                cli.specprefill_keep_ratio
            );
        }
        if cli.specprefill_pool_window % 2 != 1 || cli.specprefill_pool_window == 0 {
            anyhow::bail!(
                "--specprefill-pool-window must be odd and > 0 (got {})",
                cli.specprefill_pool_window
            );
        }
        if cli.specprefill_lookahead < 1 || cli.specprefill_lookahead > 16 {
            anyhow::bail!(
                "--specprefill-lookahead must be in [1, 16] (got {})",
                cli.specprefill_lookahead
            );
        }
    } else if any_specprefill_flag {
        anyhow::bail!(
            "--specprefill-* flags require --specprefill-draft-dir (no specprefill is configured)"
        );
    }
    Ok(())
}
```

- [ ] **Step 2: Wire it into main**

In `crates/runner/src/main.rs`, after `validate_dflash_flags(&cli, &model_variant)?;` at line 997, add:

```rust
validate_specprefill_flags(&cli, &model_variant)?;
```

And update the import at line 42:

```rust
use policy::{q4km_like, validate_dflash_flags, validate_gfx942_policy, validate_global_flags, validate_specprefill_flags};
```

- [ ] **Step 3: Build**

Run: `cargo check -p runner --release`

- [ ] **Step 4: Commit**

```bash
git add crates/runner/src/policy.rs crates/runner/src/main.rs
git commit -m "specprefill: validate_specprefill_flags policy + main wiring"
```

## Task 13: Create `specprefill_engine.rs` skeleton

**Files:**
- Create: `crates/runner/src/specprefill_engine.rs`
- Modify: `crates/runner/src/lib.rs` (mod declaration)

- [ ] **Step 1: Add the mod declaration**

In `crates/runner/src/lib.rs`, add `pub mod specprefill_engine;` next to the other engine mods.

- [ ] **Step 2: Write the orchestrator**

```rust
//! SpecPrefill (arXiv 2502.02789) end-to-end engine for Qwen3.5-9B target
//! + Qwen3.5-0.8B draft.
//!
//! Pattern mirrors `qwen35_dflash_engine.rs`: load both models, run a
//! custom prefill phase that drives the speculator + selection + sparse
//! target prefill, then hand off to the standard decode loop.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::weights::Qwen35Weights;

use crate::decode_engine::DecodeEngine;
use crate::prefill_engine::{prefill_kept, prefill_with_lookahead_attention};
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};
use crate::specprefill::{select_kept_positions, SelectionConfig};
use crate::Cli;

pub fn run_specprefill(
    cli: &Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    // ----- 1. CLI gating (validate_specprefill_flags ran earlier) ----
    let draft_dir = cli
        .specprefill_draft_dir
        .as_ref()
        .ok_or_else(|| anyhow!("specprefill_engine: missing --specprefill-draft-dir"))?;

    let params = match &entry.params {
        FamilyParams::Qwen35(p) => *p,
        _ => unreachable!("run_specprefill dispatched for non-qwen35 variant"),
    };

    // ----- 2. Tokeniser + target config -----------------------------
    let target_text_config = qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow!("load target config: {e}"))?
        .text_config;
    let tokenizer = crate::load_tokenizer(&cli.model_dir.join("tokenizer.json"))?;
    let prompt_ids = crate::resolve_prompt_token_ids(cli, &tokenizer)?;

    // ----- 3. Selection config -------------------------------------
    let cfg = SelectionConfig {
        keep_ratio: cli.specprefill_keep_ratio,
        chunk_size: cli.specprefill_chunk_size,
        pool_window: cli.specprefill_pool_window,
        always_keep_prefix: cli.specprefill_always_keep_prefix,
        always_keep_suffix: cli.specprefill_always_keep_suffix,
    };

    // ----- 4. VRAM budget ------------------------------------------
    // Target weights (BF16) + draft weights (BF16) + KV @ kept_len
    // (upper bound from specprefill::keep_count) + scratch + overhead.
    let kept_upper = crate::specprefill::keep_count(prompt_ids.len(), &cfg);
    let kv_per_token = target_text_config.kv_bytes_per_token(ScalarType::BF16.size_in_bytes());
    let kv_budget = kv_per_token * (kept_upper as u64);
    let target_fixed = entry.vram.fixed_bytes;
    let draft_fixed: u64 = 2 * 1024 * 1024 * 1024; // 1.6 GiB BF16 + scratch
    let estimated = ((target_fixed + draft_fixed + kv_budget) as f64 * entry.vram.overhead_factor) as u64;
    eprintln!(
        "[specprefill] vram estimate target={:.2}GiB draft={:.2}GiB kv_kept={:.2}GiB total={:.2}GiB available={:.2}GiB",
        target_fixed as f64 / 1024.0 / 1024.0 / 1024.0,
        draft_fixed as f64 / 1024.0 / 1024.0 / 1024.0,
        kv_budget as f64 / 1024.0 / 1024.0 / 1024.0,
        estimated as f64 / 1024.0 / 1024.0 / 1024.0,
        total_vram as f64 / 1024.0 / 1024.0 / 1024.0,
    );
    if estimated > total_vram {
        bail!(
            "SpecPrefill VRAM budget exceeded: need ~{:.2} GiB, have {:.2} GiB. \
             Try a higher --specprefill-keep-ratio or --specprefill-unload-draft.",
            estimated as f64 / 1024.0 / 1024.0 / 1024.0,
            total_vram as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }

    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    // ----- 5. Load draft weights + state ---------------------------
    let draft_text_config = qwen35::config::load_config(draft_dir)
        .map_err(|e| anyhow!("load draft config: {e}"))?
        .text_config;
    if draft_text_config.vocab_size != target_text_config.vocab_size {
        bail!(
            "draft vocab_size {} != target vocab_size {} — same-family check failed",
            draft_text_config.vocab_size,
            target_text_config.vocab_size,
        );
    }
    let t0 = Instant::now();
    let draft_weights = qwen35::weights::Qwen35Weights::load(
        draft_dir,
        &draft_text_config,
        ordinal,
        /* int4 */ false,
        /* fp8 */ false,
        /* group_size */ 0,
        /* fp8_block_size */ 0,
    )
    .map_err(|e| anyhow!("load draft weights: {e}"))?;
    eprintln!("[specprefill] draft weights loaded in {:.0}ms", t0.elapsed().as_millis());

    let draft_max_ctx = prompt_ids.len() + cli.specprefill_lookahead + 1;
    let draft_rotary = qwen35::rotary::RotaryTables::build(&draft_text_config, ordinal, draft_max_ctx)
        .map_err(|e| anyhow!("build draft RoPE: {e}"))?;
    let mut draft_state = qwen35::state::ModelState::new(&draft_text_config, ordinal, draft_max_ctx)
        .map_err(|e| anyhow!("alloc draft state: {e}"))?;

    let draft_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        draft_text_config.num_attention_heads,
        draft_text_config.head_dim,
        draft_max_ctx,
        params.kv_chunk_size,
    );
    let mut draft_decode = DecodeEngine::new(
        draft_weights.clone(), // Arc internally; cheap
        ordinal,
        params.proj_buf_floats,
        draft_attn_scratch,
        params.kv_chunk_size,
        false, // 0.8B uses the small kernel, not the 4B path
        cli.prefill_chunk_size,
        false,
        1,
    )?;

    // ----- 6. Speculator phase: prefill + lookahead + selection -----
    let lookahead_count = cli.specprefill_lookahead + 1;
    let speculator_start = Instant::now();
    let look = prefill_with_lookahead_attention(
        &draft_weights,
        &mut draft_state,
        &draft_rotary,
        &prompt_ids,
        ordinal,
        params.kv_chunk_size,
        cli.prefill_chunk_size,
        false,
        lookahead_count,
        &mut draft_decode,
    )?;
    eprintln!(
        "[specprefill] speculator done in {:.0}ms (full layers={}, kv_len={})",
        speculator_start.elapsed().as_millis(),
        look.layer_scores.len(),
        prompt_ids.len(),
    );

    // ----- 7. Aggregate per-token importance ------------------------
    // Score formula (paper §3.3, oracle/specprefill_oracle.py):
    // max over heads → max over layers → mean over lookahead steps.
    let t = prompt_ids.len();
    let mut importance = vec![0.0_f32; t];
    let q_heads = draft_text_config.num_attention_heads;
    for q_row in 0..lookahead_count {
        let mut row = vec![f32::NEG_INFINITY; t];
        for layer_scores in &look.layer_scores {
            // [q_heads, lookahead_count, t] flat
            for q_head in 0..q_heads {
                let base = (q_head * lookahead_count + q_row) * t;
                for k_pos in 0..t {
                    let v = layer_scores[base + k_pos];
                    if v > row[k_pos] {
                        row[k_pos] = v;
                    }
                }
            }
        }
        for k_pos in 0..t {
            importance[k_pos] += row[k_pos];
        }
    }
    for v in importance.iter_mut() {
        *v /= lookahead_count as f32;
    }

    // ----- 8. Selection ---------------------------------------------
    let kept_positions: Vec<u32> = select_kept_positions(&importance, &cfg);
    eprintln!(
        "[specprefill] kept {} / {} tokens ({:.1}%)",
        kept_positions.len(),
        t,
        100.0 * kept_positions.len() as f32 / t as f32
    );

    // ----- 9. (Optional) free draft to recover VRAM -----------------
    if cli.specprefill_unload_draft {
        drop(draft_decode);
        drop(draft_state);
        drop(draft_weights);
        drop(draft_rotary);
        eprintln!("[specprefill] draft unloaded");
    }

    // ----- 10. Load target weights + state + decode engine ----------
    let target_weights = qwen35::weights::Qwen35Weights::load(
        &cli.model_dir,
        &target_text_config,
        ordinal,
        false,
        false,
        0,
        0,
    )
    .map_err(|e| anyhow!("load target weights: {e}"))?;

    let target_context = cli.context_size.unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    let target_rotary = qwen35::rotary::RotaryTables::build(&target_text_config, ordinal, target_context)
        .map_err(|e| anyhow!("build target RoPE: {e}"))?;
    let mut target_state = qwen35::state::ModelState::new(&target_text_config, ordinal, target_context)
        .map_err(|e| anyhow!("alloc target state: {e}"))?;
    let target_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        target_text_config.num_attention_heads,
        target_text_config.head_dim,
        target_context,
        params.kv_chunk_size,
    );
    let mut target_engine = DecodeEngine::new(
        target_weights.clone(),
        ordinal,
        params.proj_buf_floats,
        target_attn_scratch,
        params.kv_chunk_size,
        params.use_4b_kernel,
        cli.prefill_chunk_size,
        false,
        1,
    )?;

    // ----- 11. Target sparse prefill --------------------------------
    let prefill_start = Instant::now();
    let prefill_result = prefill_kept(
        &target_weights,
        &mut target_state,
        &target_rotary,
        &prompt_ids,
        &kept_positions,
        ordinal,
        params.kv_chunk_size,
        cli.prefill_chunk_size,
        false,
        params.use_4b_kernel,
    )?;
    eprintln!(
        "[specprefill] target prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );

    // ----- 12. LAST_LOGITS dump for parity tests --------------------
    if cli.dump_last_logits {
        use std::io::Write as _;
        print!("\nLAST_LOGITS: ");
        for (i, x) in prefill_result.logits.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", x);
        }
        println!();
        std::io::stdout().flush().ok();
    }

    // ----- 13. Decode loop ------------------------------------------
    // Decode position cursor starts at the ORIGINAL prompt length, not
    // the kept count. The existing decode_engine takes seqlen_offset
    // per call, so this is straightforward.
    let mut next_id = DecodeEngine::greedy_sample(&prefill_result.logits);
    let mut generated: Vec<u32> = Vec::new();
    let mut pos = prompt_ids.len();
    while generated.len() < cli.max_new_tokens {
        if target_text_config.eos_token_ids().contains(&next_id) {
            generated.push(next_id);
            break;
        }
        let logits = target_engine.decode_step(next_id, pos)?;
        generated.push(next_id);
        next_id = DecodeEngine::greedy_sample(&logits);
        pos += 1;
    }

    // ----- 14. Detokenise + print ----------------------------------
    let all: Vec<u32> = prompt_ids.iter().copied().chain(generated.iter().copied()).collect();
    let text = tokenizer.decode(&all, true).map_err(|e| anyhow!("detokenize: {e}"))?;
    println!("{text}");
    Ok(())
}
```

- [ ] **Step 3: Build**

Run: `cargo check -p runner --release`
Expected: clean. Field/method names like `eos_token_ids`, `kv_bytes_per_token`, `Qwen35Weights::load` may differ — adjust to match the actual signatures by reading the existing call sites in `qwen35_dflash_engine.rs`.

- [ ] **Step 4: Commit**

```bash
git add crates/runner/src/specprefill_engine.rs crates/runner/src/lib.rs
git commit -m "specprefill: orchestrator engine"
```

## Task 14: Wire dispatch in `main.rs`

**Files:**
- Modify: `crates/runner/src/main.rs`

- [ ] **Step 1: Add the dispatch**

Inside the Qwen3.5 family branch (around line 1010, after the dflash dispatch closes), add before the existing post-dflash logic:

```rust
if cli.specprefill_draft_dir.is_some() {
    return specprefill_engine::run_specprefill(
        &cli,
        &model_variant,
        entry,
        ordinal,
        total_vram,
    );
}
```

- [ ] **Step 2: Add LAST_LOGITS to the dense Qwen3.5 path (test baseline)**

Find where `prefill_result.logits` becomes available in the standard run path (line ~1429). Right after, add:

```rust
if cli.dump_last_logits {
    use std::io::Write as _;
    print!("\nLAST_LOGITS: ");
    for (i, x) in prefill_result.logits.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{}", x);
    }
    println!();
    std::io::stdout().flush().ok();
}
```

(This makes the dense path emit LAST_LOGITS so the integration test can fetch the baseline.)

- [ ] **Step 3: Build**

Run: `cargo check -p runner --release`

- [ ] **Step 4: Smoke-test the SpecPrefill dispatch builds end-to-end**

Run (compile-only, no model load needed):
```bash
cargo build -p runner --release --bin supersonic
```
Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/main.rs
git commit -m "specprefill: dispatch + LAST_LOGITS on Qwen3.5 dense path"
```

## Task 15: Integration test `specprefill_qwen35_9b_parity`

**Files:**
- Create: `crates/runner/tests/specprefill_qwen35_9b_parity.rs`

- [ ] **Step 1: Write the test**

```rust
//! End-to-end parity: Qwen3.5-9B sparse SpecPrefill (keep=0.50,
//! Qwen3.5-0.8B draft) vs the same prompt's dense prefill. Asserts:
//!  - argmax(last-prefill-step logits) matches.
//!  - cossim ≥ 0.90.
//!  - top-5 overlap ≥ 4/5.
//!
//! Skipped silently when:
//!  - HIP backend not compiled.
//!  - SUPERSONIC_QWEN35_9B_DIR or SUPERSONIC_QWEN35_0_8B_DIR not set or path missing.
//!  - SUPERSONIC_SPECPREFILL_PARITY=0.

use gpu_hal::Backend;
use std::process::Command;

fn run_supersonic_capture_logits(
    args: &[&str],
) -> anyhow::Result<Vec<f32>> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args(args);
    cmd.arg("--dump-last-logits");
    let out = cmd.output()?;
    if !out.status.success() {
        anyhow::bail!(
            "supersonic exited {}: stderr=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stdout = String::from_utf8(out.stdout)?;
    let line = stdout
        .lines()
        .find(|l| l.starts_with("LAST_LOGITS:"))
        .ok_or_else(|| anyhow::anyhow!("LAST_LOGITS line not found"))?;
    let csv = &line["LAST_LOGITS:".len()..];
    csv.trim()
        .split(',')
        .map(|s| s.trim().parse::<f32>().map_err(Into::into))
        .collect()
}

fn cossim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
        .0
}

fn top5(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(5).map(|p| p.0).collect()
}

#[test]
fn specprefill_qwen35_9b_vs_dense_parity() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if std::env::var("SUPERSONIC_SPECPREFILL_PARITY").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_SPECPREFILL_PARITY=0");
        return;
    }
    let target = match std::env::var("SUPERSONIC_QWEN35_9B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_9B_DIR unset or missing");
            return;
        }
    };
    let draft = match std::env::var("SUPERSONIC_QWEN35_0_8B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_0_8B_DIR unset or missing");
            return;
        }
    };

    let prompt = "The cosine similarity of two parallel decoding streams is";

    let common: Vec<&str> = vec![
        "--model", "qwen3.5-9b",
        "--model-dir", target.as_str(),
        "--prompt", prompt,
        "--max-new-tokens", "1",
    ];

    let dense_logits =
        run_supersonic_capture_logits(&common).expect("dense run");

    let mut sparse_args = common.clone();
    sparse_args.extend_from_slice(&[
        "--specprefill-draft-dir", draft.as_str(),
        "--specprefill-keep-ratio", "0.50",
    ]);
    let sparse_logits =
        run_supersonic_capture_logits(&sparse_args).expect("sparse run");

    assert_eq!(
        dense_logits.len(),
        sparse_logits.len(),
        "logits length mismatch ({} vs {})",
        dense_logits.len(),
        sparse_logits.len()
    );
    let dense_argmax = argmax(&dense_logits);
    let sparse_argmax = argmax(&sparse_logits);
    let cs = cossim(&dense_logits, &sparse_logits);
    let dense_top5: std::collections::HashSet<usize> =
        top5(&dense_logits).into_iter().collect();
    let sparse_top5: std::collections::HashSet<usize> =
        top5(&sparse_logits).into_iter().collect();
    let overlap = dense_top5.intersection(&sparse_top5).count();
    eprintln!(
        "[specprefill parity] cossim={:.6} dense_argmax={} sparse_argmax={} top5_overlap={}/5",
        cs, dense_argmax, sparse_argmax, overlap
    );
    assert_eq!(dense_argmax, sparse_argmax, "argmax mismatch");
    assert!(cs >= 0.90, "cossim {cs} < 0.90");
    assert!(overlap >= 4, "top-5 overlap {overlap} < 4");
}
```

- [ ] **Step 2: Run the test on the dev box**

Run:
```bash
SUPERSONIC_QWEN35_9B_DIR=/mnt/data/models/Qwen3.5-9B \
SUPERSONIC_QWEN35_0_8B_DIR=/mnt/data/models/Qwen3.5-0.8B \
cargo test -p runner --release --test specprefill_qwen35_9b_parity -- --nocapture
```
Expected: `1 passed` with cossim ≥ 0.90 and argmax match. If cossim is in [0.85, 0.90), Phase A2 numbers suggest the prompt may be more aggregation-bound than the long prompt we measured — re-evaluate before relaxing the bar.

- [ ] **Step 3: Commit**

```bash
git add crates/runner/tests/specprefill_qwen35_9b_parity.rs
git commit -m "specprefill: e2e parity test (Qwen3.5-9B vs sparse keep=0.50)"
```

## Task 16: Final smoke + PR open

- [ ] **Step 1: Run all SpecPrefill tests**

```bash
cargo test -p runner --release specprefill -- --nocapture
```
Expected on the dev box: all skipped paths skipped, all runnable paths pass.

- [ ] **Step 2: Run a manual end-to-end command**

```bash
cargo run --release --bin supersonic -- \
    --model qwen3.5-9b \
    --model-dir /mnt/data/models/Qwen3.5-9B \
    --specprefill-draft-dir /mnt/data/models/Qwen3.5-0.8B \
    --specprefill-keep-ratio 0.50 \
    --prompt "Explain in one sentence what cosine similarity is." \
    --max-new-tokens 32
```
Expected: completes in < 10 seconds, produces a coherent sentence, prints `[specprefill] kept N/M tokens (...)` and `[specprefill] target prefill done in ... ms`.

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "SpecPrefill (arXiv 2502.02789): Phase C end-to-end on Qwen3.5-9B" \
    --body "$(cat <<'EOF'
## Summary
- Adds the SpecPrefill (arXiv 2502.02789) speculator-driven prefill path for Qwen3.5-9B target + Qwen3.5-0.8B draft.
- Greedy decode only; HIP backend; single GPU, single batch.
- Default keep_ratio=0.50 per Phase A2 measurements (cossim ≥ 0.927, argmax match on 9B).

## Test plan
- [ ] HIP `cargo test -p runner --release specprefill -- --nocapture` passes.
- [ ] Phase B parity tests still pass (`specprefill_rope_indirect_parity`).
- [ ] Manual end-to-end on a 1k-token prompt produces coherent output.
- [ ] Codex review bot P1 issues addressed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Address Codex review comments**

Per `~/.claude/projects/-home-deano-projects-SuperSonic/memory/reference_codex_review_bot.md`, fetch P1/P2 comments after the PR opens:
```bash
gh api repos/DeanoC/SuperSonic/pulls/<N>/comments | \
  jq '.[] | "--- " + .path + ":" + (.line // .original_line | tostring) + " ---\n" + .body'
```
Address P1s before requesting merge.

**End of PR C3.**

---

## Self-review checklist

Run through this before declaring the plan ready:

- [ ] **Spec coverage:** Every section of the spec maps to a task here. Look-ahead kernel → Tasks 1–4. `prefill_with_lookahead_attention` + `prefill_kept` → Tasks 5–10. Orchestrator + CLI + integration test → Tasks 11–15.
- [ ] **Placeholder scan:** No `TODO`, no "fill in details" without surrounding code. The "Sub-task: extend ModelState" prose in Task 9 step 4 is concrete (specifies field, method, capture site, parameter to add).
- [ ] **Type consistency:** `lookahead_count` is the same integer everywhere (= `cli.specprefill_lookahead + 1`). `kept_positions` is `&[u32]` everywhere. `q_heads`/`kv_heads`/`head_dim` match the speculator's `TextConfig`.
- [ ] **One open dependency:** `decode_step_with_query_capture` (Task 8) — the existing decode path's exact Q-row capture site has to be located by reading `decode_engine.rs`. The plan calls this out and gives the implementation strategy (reuse `decode_step_with_taps_kernel` body, swap capture).
