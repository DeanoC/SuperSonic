# Qwen3.6-35B-A3B FP8 + FP8 KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land FP8 KV cache for Qwen3.6-35B-A3B on gfx1100 (combinable with `--int4` weights) and finish FP8-runtime weight validation on gfx942. The FP8-weight kernel + descriptor + bake plumbing already exists; this plan adds KV-FP8 to the persistent megakernel and ships the gfx942 FP8-weight bring-up.

**Architecture:** KV-FP8 mirrors `kernels/full_attention_4b.hip` — FP8 E4M3 bytes for K/V cache, F32 absmax scales `[num_kv_heads, max_T]`, optional BF16 sidecar window for parity-sensitive recent reads. Quant on write, dequant on read inside `kernels/qwen36_moe_persistent/full_attn_phase.cuh`. New `Qwen36MoeKVCacheFp8Desc` parallel-struct mirrors the qwen35 `KVCacheFp8Desc` pattern. VMM-backed KV reservations follow the proven Qwen3.5 dense KV pattern (role=`KvCache`).

**Tech Stack:** Rust 2021 (workspace), HIP/hipcc on gfx1100 + gfx942, ROCm VMM (HIP 6.x driver), Python 3 + PyTorch (oracle parity only), `safetensors` for weight ingest, `cargo test --release` for integration tests.

**Reference spec:** `docs/superpowers/specs/2026-05-03-qwen36-moe-fp8-kv-fp8-design.md`

---

## File Structure

**Files to create:**
- `crates/runner/tests/qwen36_moe_kv_fp8_parity.rs` — oracle cossim integration test
- `crates/runner/tests/qwen36_moe_kv_fp8_vmm_smoke.rs` — VMM vs dense-buffer parity
- `crates/runner/tests/qwen36_moe_fp8_weights_smoke.rs` — gfx942-conditional FP8-weight smoke

**Files to modify:**
- `kernels/qwen36_moe.hip` — append `KVCacheFp8Desc` C++ struct + extend `DecodeLayerDesc` C++ side with sidecar fields (mirror Rust)
- `kernels/qwen36_moe_persistent/full_attn_phase.cuh` — KV-FP8 quant on write (~ line 725-737), dequant on read (~ lines 800, 843)
- `kernels/qwen36_moe_persistent/persistent_decode.hip` — pull KV-FP8 desc through, expose to attn phase
- `kernels/qwen36_moe_bridge.cpp` — extend persistent_decode_launch to accept KV-FP8 desc; bridge validation; bump static_assert size bound
- `crates/kernel-ffi/src/qwen36_moe.rs` — add `Qwen36MoeKVCacheFp8Desc`, append sidecar fields to `Qwen36MoeDecodeLayerDesc`, extend `qwen36_moe_hip_persistent_decode_launch` extern, add launcher wrapper
- `crates/qwen36_moe/src/state.rs` — extend account/layout for FP8 KV scales + sidecar
- `crates/runner/src/qwen36_moe_state.rs` — actually allocate FP8 KV + scale + sidecar buffers per full-attn layer
- `crates/runner/src/qwen36_moe_persistent_decode.rs` — build `Qwen36MoeKVCacheFp8Desc[]` and thread sidecar pointers into `Qwen36MoeDecodeLayerDesc`
- `crates/runner/src/qwen36_moe_engine.rs` — gate `--kv-fp8` (require `--persistent-decode`), pass FP8 KV state through
- `crates/runner/src/main.rs` — CLI gates: gfx1100 + qwen3.6 + `--kv-fp8` allow; gfx1100 + `--fp8-runtime` reject
- `crates/runner/src/registry.rs` — bump gfx942 Qwen3.6-35B-A3B `fixed_bytes` to 48 GiB
- `docs/lowlevel-memory.md` — drop "Qwen3.6-MoE decode descriptors" from disabled list once VMM-backed KV-FP8 lands

**Single source of truth on convention:** scales are F32 `[num_kv_heads, max_T]`. Sidecar BF16 lives in `[num_kv_heads, window, head_dim]`. Window default = `max_T`. Linear-attn layers (3-of-4) leave all KV-FP8 / sidecar pointers null; the kernel skips them.

---

## Task 1: Add `Qwen36MoeKVCacheFp8Desc` FFI struct (Rust side)

**Files:**
- Modify: `crates/kernel-ffi/src/qwen36_moe.rs:170` (insert after `Qwen36MoeInt4ScaleDesc`)

- [ ] **Step 1: Add the struct + impls**

Append immediately after the closing `impl Default for Qwen36MoeInt4ScaleDesc` block:

```rust
/// Per-layer KV cache FP8 scale pointers for Qwen3.6-MoE.
///
/// Parallel struct to [`Qwen36MoeDecodeLayerDesc`] — one entry per layer.
/// Linear-attention layers leave both pointers null. When KV-FP8 is off,
/// the entire `*const Qwen36MoeKVCacheFp8Desc` array argument is null.
///
/// Mirrors the qwen35 `KVCacheFp8Desc` shape: F32 absmax scale per
/// (kv_head, position).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen36MoeKVCacheFp8Desc {
    /// `[num_kv_heads, kv_max_t]` F32. Null for linear-attn layers.
    pub kv_scale_k: *mut c_void,
    /// `[num_kv_heads, kv_max_t]` F32. Null for linear-attn layers.
    pub kv_scale_v: *mut c_void,
}

unsafe impl Send for Qwen36MoeKVCacheFp8Desc {}
unsafe impl Sync for Qwen36MoeKVCacheFp8Desc {}

impl Default for Qwen36MoeKVCacheFp8Desc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
```

- [ ] **Step 2: Verify it compiles in isolation**

Run: `cargo check -p kernel-ffi `
Expected: clean build, no warnings about the new struct.

- [ ] **Step 3: Commit**

```bash
git add crates/kernel-ffi/src/qwen36_moe.rs
git commit -m "qwen36-moe-fp8: add Qwen36MoeKVCacheFp8Desc FFI struct"
```

---

## Task 2: Append KV-FP8 sidecar fields to `Qwen36MoeDecodeLayerDesc` (Rust)

The kernel needs `kv_shadow_k`, `kv_shadow_v`, `kv_shadow_start` per layer (mirroring qwen35 `DecodeLayerDesc` fields at line 64-66). They're appended (not interleaved) so the existing C++ `static_assert` only needs the upper bound bumped.

**Files:**
- Modify: `crates/kernel-ffi/src/qwen36_moe.rs:115` (after `pub norm_topk_prob: c_int,`)

- [ ] **Step 1: Append three fields**

Find the end of `Qwen36MoeDecodeLayerDesc` (just before `}` that closes the `#[repr(C)] pub struct`). Insert before that closing brace:

```rust
    // --- KV-FP8 sidecar (read iff is_full_attention == 1 AND
    // matching kv_fp8_descs[layer].kv_scale_k != null) ---------------
    /// BF16 sidecar buffer `[num_kv_heads, window, head_dim]`. Null when
    /// the sidecar is disabled. The kernel reads from the sidecar (instead
    /// of dequantising FP8) when `t >= kv_shadow_start`.
    pub kv_shadow_k: *mut c_void,
    /// BF16 sidecar buffer `[num_kv_heads, window, head_dim]`.
    pub kv_shadow_v: *mut c_void,
    /// First absolute KV position covered by the sidecar. `-1` when the
    /// sidecar is disabled (or no positions are covered yet); the kernel
    /// reads `t >= kv_shadow_start && kv_shadow_start >= 0` to decide.
    pub kv_shadow_start: c_int,
```

- [ ] **Step 2: Add a static size assertion**

At the bottom of the file (or right after the struct), add:

```rust
// ABI sanity — every time `Qwen36MoeDecodeLayerDesc` grows, both this
// const and the C++ side `static_assert` in `kernels/qwen36_moe_bridge.cpp`
// must move together. Keep the values pinned: a silent drift here is the
// most common Rust↔C++ bug class for this codebase.
#[cfg(target_pointer_width = "64")]
const _ASSERT_DECODE_LAYER_DESC_SIZE: () = {
    // 24 bytes added (2 ptrs + 1 int padded to 8 bytes).
    assert!(std::mem::size_of::<Qwen36MoeDecodeLayerDesc>() <= 512);
    assert!(std::mem::size_of::<Qwen36MoeDecodeLayerDesc>() >= 256);
};
```

- [ ] **Step 3: Verify build**

Run: `cargo check -p kernel-ffi `
Expected: clean build. (Bridge static_assert will not fire yet — same lower bound, same upper bound.)

- [ ] **Step 4: Commit**

```bash
git add crates/kernel-ffi/src/qwen36_moe.rs
git commit -m "qwen36-moe-fp8: append kv_shadow_{k,v,start} to DecodeLayerDesc"
```

---

## Task 3: Mirror struct changes on the C++ side

`kernels/qwen36_moe.hip:58` defines `qwen36_moe::DecodeLayerDesc` for the megakernel. Append the same three fields and add the new `KVCacheFp8Desc` C++ struct. Keep field order/types identical to Rust.

**Files:**
- Modify: `kernels/qwen36_moe.hip:115` (end of `DecodeLayerDesc`)
- Modify: `kernels/qwen36_moe.hip:155` (after `Int4ScaleDesc`)

- [ ] **Step 1: Append the three fields to `DecodeLayerDesc`**

Find the closing `};` of `struct DecodeLayerDesc { ... }` in `qwen36_moe.hip`. Just before it, add:

```cpp
    // --- KV-FP8 sidecar (read iff is_full_attention == 1 AND
    // matching kv_fp8_descs[layer].kv_scale_k != null) ---------------
    void* kv_shadow_k;       // BF16 [num_kv_heads, window, head_dim]
    void* kv_shadow_v;       // BF16 [num_kv_heads, window, head_dim]
    int   kv_shadow_start;   // first KV position covered by the sidecar; -1 = disabled
```

- [ ] **Step 2: Add the new `KVCacheFp8Desc` struct**

Immediately after the `};` closing `Int4ScaleDesc { ... }`, add:

```cpp
// -- KV cache FP8 scale parallel struct (mirror of Rust
//    Qwen36MoeKVCacheFp8Desc). One entry per layer. Linear-attention
//    layers carry null pointers and the kernel skips them. The whole
//    array argument is null when KV-FP8 is off.
struct KVCacheFp8Desc {
    void* kv_scale_k;   // F32 [num_kv_heads, kv_max_t]
    void* kv_scale_v;   // F32 [num_kv_heads, kv_max_t]
};
```

- [ ] **Step 3: Bump bridge static_assert upper bound**

Edit `kernels/qwen36_moe_bridge.cpp:81-84`. Change:

```cpp
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) >= 256,
              "Qwen36MoeDecodeLayerDesc shrunk unexpectedly");
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) <= 512,
              "Qwen36MoeDecodeLayerDesc grew unexpectedly");
```

to:

```cpp
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) >= 280,
              "Qwen36MoeDecodeLayerDesc shrunk unexpectedly");
static_assert(sizeof(qwen36_moe::DecodeLayerDesc) <= 512,
              "Qwen36MoeDecodeLayerDesc grew unexpectedly");

static_assert(sizeof(qwen36_moe::KVCacheFp8Desc) == 16,
              "Qwen36MoeKVCacheFp8Desc layout drift");
```

- [ ] **Step 4: Build the kernel-ffi crate (rebuilds hipcc)**

Run: `cargo build -p kernel-ffi --release`
Expected: hipcc recompiles `qwen36_moe.hip` and `qwen36_moe_bridge.cpp` cleanly. No `static_assert` failures.

- [ ] **Step 5: Commit**

```bash
git add kernels/qwen36_moe.hip kernels/qwen36_moe_bridge.cpp
git commit -m "qwen36-moe-fp8: mirror KV-FP8 sidecar + KVCacheFp8Desc to C++"
```

---

## Task 4: Extend `qwen36_moe_hip_persistent_decode_launch` to take KV-FP8 desc

Append a new optional pointer parameter (null when KV-FP8 is off). The Rust extern, the C++ launcher, and every existing caller must update together.

**Files:**
- Modify: `crates/kernel-ffi/src/qwen36_moe.rs` (the `extern "C"` block around line 251 + the safe-wrapper `Qwen36MoePersistentLmHeadFold` if needed)
- Modify: `kernels/qwen36_moe_bridge.cpp` (the `qwen36_moe_hip_persistent_decode_launch` definition around line 1054)

- [ ] **Step 1: Add the param to the Rust extern**

Find the `pub fn qwen36_moe_hip_persistent_decode_launch(` extern declaration. Append between `int4_scales: *const Qwen36MoeInt4ScaleDesc,` and `hidden: c_int,`:

```rust
        /// Null when KV-FP8 is off. Otherwise an array of `num_layers`
        /// entries parallel to `layers`; full-attention layers populate
        /// `kv_scale_k` / `kv_scale_v`, linear-attention layers leave
        /// both null. The kernel ignores entries whose `kv_scale_k` is
        /// null even on full-attn layers (defensive — should not happen
        /// in practice).
        kv_fp8_descs: *const Qwen36MoeKVCacheFp8Desc,
```

- [ ] **Step 2: Add the matching param to the C++ launcher**

In `kernels/qwen36_moe_bridge.cpp`, find `qwen36_moe_hip_persistent_decode_launch` (around line 1050). Add the same parameter:

```cpp
    const qwen36_moe::DecodeLayerDesc* layers,
    const qwen36_moe::Int4ScaleDesc*   int4_scales,    // nullable
    const qwen36_moe::KVCacheFp8Desc*  kv_fp8_descs,   // nullable
```

Add a validation block early in the function body (after the existing `int4_scales` validation):

```cpp
    // KV-FP8 desc validation: when present, every full-attn layer must
    // carry both kv_scale_k and kv_scale_v (or neither). Linear-attn
    // layers must carry null pointers in this struct.
    if (kv_fp8_descs != nullptr) {
        for (int li = 0; li < static_cast<int>(num_layers); ++li) {
            const auto& d  = layers[li];
            const auto& kf = kv_fp8_descs[li];
            const bool full = (d.is_full_attention == 1);
            const bool both = (kf.kv_scale_k != nullptr && kf.kv_scale_v != nullptr);
            const bool none = (kf.kv_scale_k == nullptr && kf.kv_scale_v == nullptr);
            if (full && !(both || none)) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d: kv_scale_k/v must both be "
                    "set or both null (got %p / %p)\n",
                    li, kf.kv_scale_k, kf.kv_scale_v);
                return 1;
            }
            if (!full && !none) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d (linear): kv_scale_k/v "
                    "must be null (got %p / %p)\n",
                    li, kf.kv_scale_k, kf.kv_scale_v);
                return 1;
            }
            if (full && both && ((d.kv_shadow_k != nullptr) != (d.kv_shadow_v != nullptr))) {
                fprintf(stderr,
                    "[qwen36_moe] KV-FP8 layer %d: kv_shadow_k/v must agree "
                    "(got %p / %p)\n",
                    li, d.kv_shadow_k, d.kv_shadow_v);
                return 1;
            }
        }
    }
```

- [ ] **Step 3: Do NOT pass to the kernel yet**

Stop here. The launcher takes the param and validates it but does not yet pass it to the kernel — Task 5 adds the kernel param and updates the `hipLaunchCooperativeKernel` arg tuple atomically. If you push a half-update here, the kernel reads garbage at the new arg position. Confirm via `grep -n "hipLaunchCooperativeKernel" kernels/qwen36_moe_bridge.cpp` that the kernel arg tuple is unchanged in this commit.

- [ ] **Step 4: Update existing call sites of the extern**

Grep for callers of the extern (not the safe wrapper) in Rust:

```bash
grep -rn "qwen36_moe_hip_persistent_decode_launch" crates/
```

The only direct call site is the safe wrapper `persistent_decode_launch` in `crates/kernel-ffi/src/qwen36_moe.rs` (around line 817 — inside an `unsafe { ... }` block). The runner does NOT call the extern directly; it calls the safe wrapper. Update the unsafe extern invocation to pass null for the new param, between `int4_ptr` and `geom.hidden`:

```rust
qwen36_moe_hip_persistent_decode_launch(
    dtype.kernel_dtype_code(),
    // ... ordinal, num_layers, layers_device.as_ptr() ...
    int4_ptr,
    std::ptr::null::<Qwen36MoeKVCacheFp8Desc>(),
    geom.hidden,
    // ... rest of args unchanged ...
)
```

Do NOT extend the safe wrapper signature in this task — keep the wrapper's external API stable. Task 11 extends the wrapper signature to accept `kv_fp8_descs_device: Option<&GpuBuffer>` and threads it down to this same call site, replacing the null. The runner crate `crates/runner/src/qwen36_moe_persistent_decode.rs` calls `persistent_decode_launch` (the wrapper), so no runner-side changes land in this commit.

- [ ] **Step 5: Build everything that depends on the FFI**

Run: `cargo build -p runner --release`
Expected: clean build. No silent missing-arg errors.

- [ ] **Step 6: Commit**

```bash
git add crates/kernel-ffi/src/qwen36_moe.rs kernels/qwen36_moe_bridge.cpp
git commit -m "qwen36-moe-fp8: thread KVCacheFp8Desc through persistent_decode_launch (null at runtime)"
```

(The runner crate is unchanged in this commit — its calls go through the unchanged safe wrapper. Task 11 extends the wrapper signature.)

---

## Task 5: Plumb KV-FP8 desc into the persistent kernel signature

The persistent kernel (`kernels/qwen36_moe_persistent/persistent_decode.hip`) needs a new kernel parameter and must pass the per-layer pointer down into `qwen36_moe_attn_step_device` (defined in `full_attn_phase.cuh`).

**Files:**
- Modify: `kernels/qwen36_moe_persistent/persistent_decode.hip`
- Modify: `kernels/qwen36_moe_persistent/full_attn_phase.cuh` (signature only — write/read math comes in Tasks 6 and 7)

- [ ] **Step 1: Add the kernel parameter**

Open `persistent_decode.hip` and find `__global__ void supersonic_qwen36_moe_persistent_decode_kernel(...)` (the cooperative launch entry). Add after the `Int4ScaleDesc* int4_scales` param:

```cpp
    const KVCacheFp8Desc* __restrict__ kv_fp8_descs,  // nullable
```

In the bridge (`qwen36_moe_bridge.cpp`), atomically: (a) extend the `void* args[]` (or equivalent) tuple passed to `hipLaunchCooperativeKernel` with `&kv_fp8_descs` (the launcher param from Task 4), and (b) trust that the kernel signature now matches. Do this in a single edit so the launcher and kernel never disagree mid-commit.

- [ ] **Step 2: Pass per-layer pointer into the attention phase**

Inside the persistent kernel's per-layer dispatch (where the existing code reads `layers[layer_idx]` and dispatches to `qwen36_moe_attn_step_device`), compute:

```cpp
const KVCacheFp8Desc* kv_fp8 = (kv_fp8_descs != nullptr) ? &kv_fp8_descs[layer_idx] : nullptr;
```

and forward it into the attn-phase device call (Step 3 below).

- [ ] **Step 3: Extend `qwen36_moe_attn_step_device` signature**

Open `full_attn_phase.cuh`. Find the `qwen36_moe_attn_step_device` device-fn declaration (the one called by the persistent kernel; not the staged-step kernel from PR 4b2). Append parameters after `T* __restrict__ kv_cache_v,` and before `int kv_max_t,`:

```cpp
    // KV-FP8: when these are non-null the cache is FP8 E4M3 bytes (so
    // the T* pointers above point at U8 — the kernel reinterprets in
    // place). When null, the cache is BF16 and the FP8 path is skipped.
    float* __restrict__       kv_scale_k,
    float* __restrict__       kv_scale_v,
    // BF16 sidecar (optional; requires KV-FP8 to be active). When null
    // the kernel always dequantises from FP8.
    void* __restrict__        kv_shadow_k,
    void* __restrict__        kv_shadow_v,
    int                       kv_shadow_start,
```

Forward all five fields from the persistent kernel through to `qwen36_moe_attn_step_device`. The body work in Task 6/7 still operates on the BF16 path because callers pass nulls; this task is signature-only.

- [ ] **Step 4: Build kernel-ffi**

Run: `cargo build -p kernel-ffi --release`
Expected: clean. Per-block parity tests untouched (their staged-step kernel doesn't use the persistent path).

- [ ] **Step 5: Commit**

```bash
git add kernels/qwen36_moe_persistent/persistent_decode.hip \
        kernels/qwen36_moe_persistent/full_attn_phase.cuh \
        kernels/qwen36_moe_bridge.cpp
git commit -m "qwen36-moe-fp8: plumb KV-FP8 + sidecar pointers into attn phase signature"
```

---

## Task 6: Implement KV-FP8 quant-on-write in `full_attn_phase.cuh`

Replace the BF16 KV write at `full_attn_phase.cuh:725-738` with a branched path: BF16 when `kv_scale_k == nullptr`, FP8 (with optional sidecar) otherwise. The math comes verbatim from `kernels/full_attention_4b.hip:5228-5275`.

**Files:**
- Modify: `kernels/qwen36_moe_persistent/full_attn_phase.cuh:716-742` (the "Step 1: write current K/V" block)
- Modify: `kernels/qwen36_moe_persistent/helpers.cuh` (add `float_to_fp8_e4m3` if not already there)

- [ ] **Step 1: Confirm `float_to_fp8_e4m3` exists in `helpers.cuh`**

Check:

```bash
grep -n "float_to_fp8_e4m3\|fp8_e4m3_to_float" kernels/qwen36_moe_persistent/helpers.cuh
```

`fp8_e4m3_to_float` already exists for weight dequant; `float_to_fp8_e4m3` is needed for KV write. If absent, append to `helpers.cuh` (taking the verbatim implementation from `kernels/full_attention_4b.hip:205-238`):

```cpp
__device__ inline uint8_t float_to_fp8_e4m3(float val) {
    if (val != val) return 0x7Fu; // NaN → max positive (matches CUDA __nv_fp8_e4m3 saturation)
    if (val ==  INFINITY) return 0x7Eu;
    if (val == -INFINITY) return 0xFEu;
    const uint8_t sign = (val < 0.f) ? 0x80u : 0x00u;
    float a = fabsf(val);
    if (a > 448.0f) a = 448.0f;
    if (a == 0.0f) return sign;
    int e_unbiased;
    float m = frexpf(a, &e_unbiased);
    int exp_field = e_unbiased - 1 + 7;
    uint32_t mant_field;
    if (exp_field <= 0) {
        const float scale = ldexpf(1.0f, -6 - exp_field);
        const float mant_f = a * scale;
        mant_field = static_cast<uint32_t>(mant_f + 0.5f);
        if (mant_field >= 8u) {
            mant_field -= 8u;
            exp_field = 1;
        } else {
            exp_field = 0;
        }
    } else {
        const float mant_f = (m * 2.0f - 1.0f) * 8.0f;
        mant_field = static_cast<uint32_t>(mant_f + 0.5f);
        if (mant_field >= 8u) {
            mant_field -= 8u;
            exp_field += 1;
        }
    }
    if (exp_field >= 0xF) return sign | 0x7Eu;
    return static_cast<uint8_t>(sign | (exp_field << 3) | (mant_field & 0x7u));
}
```

- [ ] **Step 2: Replace the KV write loop with the branched path**

Open `kernels/qwen36_moe_persistent/full_attn_phase.cuh`. Find the block starting at the `// Step 1: write current K/V into the cache` comment (around line 723). Replace from `if (use_kv_cache) {` through the matching closing brace of the `for (int idx = ...)` loop with:

```cpp
        if (use_kv_cache) {
            const int slot_base = eff_cache_pos * Hkv * d;
            const int total = Hkv * d;
            const bool use_fp8_kv = (kv_scale_k != nullptr && kv_scale_v != nullptr);

            if (!use_fp8_kv) {
                // BF16 path — unchanged from before KV-FP8 landed.
                for (int idx = blockIdx.x * block_size + tid;
                     idx < total;
                     idx += num_blocks * block_size) {
                    const int h_kv = idx / d;
                    const int i    = idx % d;
                    const float kv = workspace[OFF_K_ROT + h_kv * d + i];
                    const float vv = workspace[OFF_V_RAW + h_kv * d + i];
                    kv_cache_k[slot_base + h_kv * d + i] = static_cast<T>(kv);
                    kv_cache_v[slot_base + h_kv * d + i] = static_cast<T>(vv);
                }
            } else {
                // FP8 path — per-(h_kv, position) absmax → F32 scale, then
                // quantise each element of the head's d-dim vector. One
                // wavefront covers one (h_kv, position) pair so absmax can
                // run as a block-wide reduction in shared_scratch.
                uint8_t* fp8_k = reinterpret_cast<uint8_t*>(kv_cache_k);
                uint8_t* fp8_v = reinterpret_cast<uint8_t*>(kv_cache_v);
                for (int h_kv = blockIdx.x; h_kv < Hkv; h_kv += num_blocks) {
                    // Absmax(K) and Absmax(V) over the head's d-dim vector.
                    float local_max_k = 0.f, local_max_v = 0.f;
                    for (int i = tid; i < d; i += block_size) {
                        const float kv = workspace[OFF_K_ROT + h_kv * d + i];
                        const float vv = workspace[OFF_V_RAW + h_kv * d + i];
                        local_max_k = fmaxf(local_max_k, fabsf(kv));
                        local_max_v = fmaxf(local_max_v, fabsf(vv));
                    }
                    shared_scratch[tid] = local_max_k;
                    __syncthreads();
                    for (int s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s) {
                            shared_scratch[tid] = fmaxf(shared_scratch[tid], shared_scratch[tid + s]);
                        }
                        __syncthreads();
                    }
                    const float head_max_k = shared_scratch[0];
                    __syncthreads();
                    shared_scratch[tid] = local_max_v;
                    __syncthreads();
                    for (int s = block_size / 2; s > 0; s >>= 1) {
                        if (tid < s) {
                            shared_scratch[tid] = fmaxf(shared_scratch[tid], shared_scratch[tid + s]);
                        }
                        __syncthreads();
                    }
                    const float head_max_v = shared_scratch[0];
                    __syncthreads();

                    // 448.f is FP8 E4M3 max finite. Floor scale at a tiny
                    // positive value to avoid division by zero on
                    // zero-vectors (legal at position 0 of a fresh layer).
                    const float scale_k = fmaxf(head_max_k / 448.0f, 1.0e-12f);
                    const float scale_v = fmaxf(head_max_v / 448.0f, 1.0e-12f);
                    const float inv_k = 1.0f / scale_k;
                    const float inv_v = 1.0f / scale_v;

                    // Tid 0 publishes the scale. All threads use it after the
                    // following sync.
                    if (tid == 0) {
                        kv_scale_k[h_kv * kv_max_t + eff_cache_pos] = scale_k;
                        kv_scale_v[h_kv * kv_max_t + eff_cache_pos] = scale_v;
                    }

                    // Quantise + write FP8 bytes. Optionally also write
                    // BF16 sidecar at the same head/position when the
                    // sidecar is configured to cover this position.
                    const bool sidecar_active =
                        (kv_shadow_k != nullptr && kv_shadow_v != nullptr &&
                         kv_shadow_start >= 0 && eff_cache_pos >= kv_shadow_start);
                    T* shadow_k = sidecar_active ? static_cast<T*>(kv_shadow_k) : nullptr;
                    T* shadow_v = sidecar_active ? static_cast<T*>(kv_shadow_v) : nullptr;
                    const int shadow_slot =
                        sidecar_active ? (eff_cache_pos - kv_shadow_start) : 0;

                    for (int i = tid; i < d; i += block_size) {
                        const float kv = workspace[OFF_K_ROT + h_kv * d + i];
                        const float vv = workspace[OFF_V_RAW + h_kv * d + i];
                        fp8_k[slot_base + h_kv * d + i] = float_to_fp8_e4m3(kv * inv_k);
                        fp8_v[slot_base + h_kv * d + i] = float_to_fp8_e4m3(vv * inv_v);
                        if (sidecar_active) {
                            shadow_k[h_kv * /* sidecar_window */ kv_max_t * d
                                     + shadow_slot * d + i] = static_cast<T>(kv);
                            shadow_v[h_kv * /* sidecar_window */ kv_max_t * d
                                     + shadow_slot * d + i] = static_cast<T>(vv);
                        }
                    }
                    __syncthreads();
                }
            }
            // Visibility barrier for the read-side reduction.
            grid_barrier_reset_counter(barrier_counter, barrier_flag, num_blocks,
                                       &counters[0]);
        }
```

**Note on the sidecar shape:** the comment above writes `kv_shadow_window` — but the sidecar window length isn't currently a kernel argument. The descriptor says "BF16 sidecar `[num_kv_heads, window, head_dim]`". For the v1 implementation, force `window == kv_max_t` (full sidecar). The Rust state allocation in Task 9 honours this. If a future change shrinks the window, add a `kv_shadow_window` int kernel arg and replace the `kv_max_t` in the offset above. Keep the comment in code so the bookkeeping is unambiguous.

Adjust the offset accordingly — replace the `/* sidecar_window */ kv_max_t` placeholder with just `kv_max_t` (since we're forcing window = kv_max_t):

```cpp
                            shadow_k[h_kv * kv_max_t * d + shadow_slot * d + i] = static_cast<T>(kv);
                            shadow_v[h_kv * kv_max_t * d + shadow_slot * d + i] = static_cast<T>(vv);
```

- [ ] **Step 3: Build**

Run: `cargo build -p kernel-ffi --release`
Expected: clean. The persistent kernel still passes nulls for the new params at the call site (engine wiring lands in Task 11), so the new branch is dead code.

- [ ] **Step 4: Commit**

```bash
git add kernels/qwen36_moe_persistent/full_attn_phase.cuh \
        kernels/qwen36_moe_persistent/helpers.cuh
git commit -m "qwen36-moe-fp8: implement FP8 KV quant-on-write with optional BF16 sidecar"
```

---

## Task 7: Implement KV-FP8 dequant-on-read in `full_attn_phase.cuh`

Both KV read sites (score computation around line 800 and V-weighted reduction around line 843) need the FP8 → F32 dequant branch. Sidecar windowed reads come from BF16; out-of-window reads dequant from FP8.

**Files:**
- Modify: `kernels/qwen36_moe_persistent/full_attn_phase.cuh:790-847` (both K and V cache reads inside the KV-cache path)

- [ ] **Step 1: Replace the K-cache read loop**

In the score-computation block (search for `kv_cache_k[t * Hkv * d + h_kv * d + i]`), replace the inner read loop:

```cpp
            const int score_base = OFF_SCORES + hq * kv_max_t;
            const bool use_fp8_kv = (kv_scale_k != nullptr);
            const bool sidecar_active =
                (kv_shadow_k != nullptr && kv_shadow_start >= 0);
            const T* shadow_k = sidecar_active ? static_cast<const T*>(kv_shadow_k) : nullptr;
            const uint8_t* fp8_ck = use_fp8_kv ? reinterpret_cast<const uint8_t*>(kv_cache_k) : nullptr;
            const float* sk_buf = use_fp8_kv ? kv_scale_k : nullptr;

            for (int t = 0; t < kv_len; t++) {
                float partial = 0.0f;
                const bool use_sidecar = sidecar_active && (t >= kv_shadow_start);
                const float scale_k_t =
                    (use_fp8_kv && !use_sidecar) ? sk_buf[h_kv * kv_max_t + t] : 0.f;
                for (int i = tid; i < d; i += block_size) {
                    const float q = workspace[OFF_Q_ROT + hq * d + i];
                    float k;
                    if (!use_fp8_kv) {
                        k = static_cast<float>(kv_cache_k[t * Hkv * d + h_kv * d + i]);
                    } else if (use_sidecar) {
                        const int shadow_slot = t - kv_shadow_start;
                        k = static_cast<float>(
                            shadow_k[h_kv * kv_max_t * d + shadow_slot * d + i]);
                    } else {
                        const uint8_t byte =
                            fp8_ck[t * Hkv * d + h_kv * d + i];
                        k = fp8_e4m3_to_float(byte) * scale_k_t;
                    }
                    partial += q * k;
                }
                shared_scratch[tid] = partial;
                __syncthreads();
                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_scratch[tid] += shared_scratch[tid + s];
                    __syncthreads();
                }
                if (tid == 0) {
                    workspace[score_base + t] = shared_scratch[0] * scale;
                }
                __syncthreads();
            }
```

- [ ] **Step 2: Replace the V-weighted reduction**

Find the V-cache read in the V-weighted reduction block (search for `kv_cache_v[t * Hkv * d + h_kv * d + i]`). Replace its inner loop:

```cpp
            const float inv_sum = 1.0f / exp_sum;
            const bool use_fp8_kv_v = (kv_scale_v != nullptr);
            const bool sidecar_active_v =
                (kv_shadow_v != nullptr && kv_shadow_start >= 0);
            const T* shadow_v = sidecar_active_v ? static_cast<const T*>(kv_shadow_v) : nullptr;
            const uint8_t* fp8_cv =
                use_fp8_kv_v ? reinterpret_cast<const uint8_t*>(kv_cache_v) : nullptr;
            const float* sv_buf = use_fp8_kv_v ? kv_scale_v : nullptr;

            for (int i = tid; i < d; i += block_size) {
                float acc = 0.0f;
                for (int t = 0; t < kv_len; t++) {
                    const float w = workspace[score_base + t] * inv_sum;
                    const bool use_sidecar = sidecar_active_v && (t >= kv_shadow_start);
                    float v;
                    if (!use_fp8_kv_v) {
                        v = static_cast<float>(kv_cache_v[t * Hkv * d + h_kv * d + i]);
                    } else if (use_sidecar) {
                        const int shadow_slot = t - kv_shadow_start;
                        v = static_cast<float>(
                            shadow_v[h_kv * kv_max_t * d + shadow_slot * d + i]);
                    } else {
                        const float scale_v_t = sv_buf[h_kv * kv_max_t + t];
                        const uint8_t byte = fp8_cv[t * Hkv * d + h_kv * d + i];
                        v = fp8_e4m3_to_float(byte) * scale_v_t;
                    }
                    acc += w * v;
                }
                workspace[OFF_ATTN + hq * d + i] = acc;
                if (stage == 4) {
                    output[hq * d + i] = static_cast<T>(acc);
                }
            }
            __syncthreads();
```

- [ ] **Step 3: Build**

Run: `cargo build -p kernel-ffi --release`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add kernels/qwen36_moe_persistent/full_attn_phase.cuh
git commit -m "qwen36-moe-fp8: implement FP8 KV dequant-on-read with sidecar window"
```

---

## Task 8: Add `qwen36_moe::state` accounting + layout for FP8 KV scales/sidecar

**Files:**
- Modify: `crates/qwen36_moe/src/state.rs`

The current state module owns the `kv_dtype_bytes` switch (1 vs 2). Extend its account to also describe the F32 scale buffer and BF16 sidecar so the engine VRAM pre-flight is honest.

- [ ] **Step 1: Extend `StateLayout::new` with sidecar info**

In `crates/qwen36_moe/src/state.rs`, change the `StateLayout::new` signature to accept the sidecar window (None = disabled):

```rust
pub fn new(
    context_tokens: usize,
    batch_size: usize,
    kv_fp8: bool,
    kv_fp8_sidecar_window: Option<usize>,
) -> Self {
    Self {
        context_tokens,
        batch_size,
        kv_dtype_bytes: if kv_fp8 { 1 } else { 2 },
        kv_fp8_scale_bytes_per_token: if kv_fp8 { 4 } else { 0 }, // F32 per (head, pos)
        kv_fp8_sidecar_bytes_per_token: match (kv_fp8, kv_fp8_sidecar_window) {
            (true, Some(_w)) => 2, // BF16 sidecar of head_dim per (head, pos)
            _ => 0,
        },
        kv_fp8_sidecar_window: kv_fp8_sidecar_window.unwrap_or(0),
    }
}
```

Add the new fields:

```rust
pub struct StateLayout {
    pub context_tokens: usize,
    pub batch_size: usize,
    pub kv_dtype_bytes: usize,
    pub kv_fp8_scale_bytes_per_token: usize,
    pub kv_fp8_sidecar_bytes_per_token: usize,
    pub kv_fp8_sidecar_window: usize,
}
```

- [ ] **Step 2: Update `StateAccount::full_kv_bytes`**

Find `full_kv_bytes` and update it to include scale + sidecar contributions per full-attn layer. (Consult the existing computation; head_dim, num_kv_heads, num_full_layers come from `TextConfig`. Scales: `4 * num_kv_heads * max_T` per layer. Sidecar: `2 * num_kv_heads * window * head_dim` per layer when enabled.)

Approximate update (verify field names against the existing source):

```rust
pub fn full_kv_bytes(&self) -> u64 {
    let per_layer_kv = (self.cfg.num_key_value_heads as u64)
        * (self.cfg.head_dim as u64)
        * (self.layout.context_tokens as u64)
        * (self.layout.kv_dtype_bytes as u64);
    let per_layer_scale = (self.cfg.num_key_value_heads as u64)
        * (self.layout.context_tokens as u64)
        * (self.layout.kv_fp8_scale_bytes_per_token as u64);
    let per_layer_sidecar = if self.layout.kv_fp8_sidecar_window > 0 {
        (self.cfg.num_key_value_heads as u64)
            * (self.layout.kv_fp8_sidecar_window as u64)
            * (self.cfg.head_dim as u64)
            * (self.layout.kv_fp8_sidecar_bytes_per_token as u64)
    } else {
        0
    };
    let per_layer = 2 * (per_layer_kv + per_layer_scale + per_layer_sidecar);
    let num_full = self.cfg.num_full_attn_layers() as u64;
    per_layer * num_full
}
```

- [ ] **Step 3: Update existing test for new signature**

Find `fp8_kv_halves_full_kv_bytes` test in the same file. Update calls:

```rust
let bf16 = StateAccount::from_config(&cfg, StateLayout::new(4096, 1, false, None)).full_kv_bytes;
let fp8  = StateAccount::from_config(&cfg, StateLayout::new(4096, 1, true,  Some(4096))).full_kv_bytes;
// Sidecar is full-window BF16 + FP8 + scales — total is *more* than half BF16 by design.
assert!(fp8 < 2 * bf16);
assert!(fp8 > bf16 / 2);

let fp8_no_sidecar = StateAccount::from_config(
    &cfg, StateLayout::new(4096, 1, true, None)).full_kv_bytes;
// FP8 + scales (no sidecar) is the bare halving plus a small per-position F32.
assert!(fp8_no_sidecar < bf16);
assert!(fp8_no_sidecar >= bf16 / 2);
```

- [ ] **Step 4: Update all `StateLayout::new` callers in the workspace**

```bash
grep -rn "StateLayout::new(" crates/
```

Pass `None` for `kv_fp8_sidecar_window` at every existing call site (the engine wiring in Task 11 will switch to a real value).

- [ ] **Step 5: Run state tests**

Run: `cargo test -p qwen36_moe --release -- state`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add crates/qwen36_moe/src/state.rs
# include any callers updated in step 4
git add -u crates/
git commit -m "qwen36-moe-fp8: account FP8 KV scales + sidecar in state layout"
```

---

## Task 9: Allocate KV-FP8 scale + sidecar buffers in runner per-layer state

The runner owns per-layer state. Add scale/sidecar buffers that mirror `qwen35::state::LayerState`'s fields.

**Files:**
- Modify: `crates/runner/src/qwen36_moe_state.rs`

- [ ] **Step 1: Locate the per-layer state struct**

```bash
grep -n "pub struct\|kv_cache_k\|kv_cache_v" crates/runner/src/qwen36_moe_state.rs | head -30
```

Use the result to identify the per-layer struct (likely `Qwen36MoeLayerState` or similar). The full-attn layer holds `kv_cache_k: Option<GpuBuffer>` and `kv_cache_v: Option<GpuBuffer>` today.

- [ ] **Step 2: Add the new fields**

In the per-layer state struct, append:

```rust
    /// FP8 KV scale K: F32 [num_kv_heads, max_T]. Some only when `kv_fp8`.
    pub kv_scale_k: Option<GpuBuffer>,
    /// FP8 KV scale V: F32 [num_kv_heads, max_T]. Some only when `kv_fp8`.
    pub kv_scale_v: Option<GpuBuffer>,
    /// BF16 sidecar K: BF16 [num_kv_heads, sidecar_window, head_dim]. Some
    /// only when `kv_fp8` and the sidecar is enabled.
    pub kv_shadow_k: Option<GpuBuffer>,
    /// BF16 sidecar V: BF16 [num_kv_heads, sidecar_window, head_dim].
    pub kv_shadow_v: Option<GpuBuffer>,
    /// First absolute KV position covered by the sidecar. Only valid when
    /// `kv_shadow_k.is_some()`. `-1` means the sidecar is configured but
    /// no positions are covered yet.
    pub kv_shadow_start: i32,
```

Default `kv_shadow_start` to `-1` and the buffers to `None` in any constructor that doesn't hand-allocate them.

- [ ] **Step 3: Allocate buffers in the full-attn layer constructor**

In the function that builds a full-attn layer state (search for `fn new_full(` or similar), gate the new allocations behind a `kv_fp8: bool` parameter:

```rust
let (kv_cache_k, kv_cache_v, kv_scale_k, kv_scale_v) = if kv_fp8 {
    (
        Some(GpuBuffer::zeros(ordinal, ScalarType::U8,
            &[num_kv_heads, max_t, head_dim])?),
        Some(GpuBuffer::zeros(ordinal, ScalarType::U8,
            &[num_kv_heads, max_t, head_dim])?),
        Some(GpuBuffer::zeros(ordinal, ScalarType::F32,
            &[num_kv_heads, max_t])?),
        Some(GpuBuffer::zeros(ordinal, ScalarType::F32,
            &[num_kv_heads, max_t])?),
    )
} else {
    (
        Some(GpuBuffer::zeros(ordinal, ScalarType::BF16,
            &[num_kv_heads, max_t, head_dim])?),
        Some(GpuBuffer::zeros(ordinal, ScalarType::BF16,
            &[num_kv_heads, max_t, head_dim])?),
        None,
        None,
    )
};
let (kv_shadow_k, kv_shadow_v) = if kv_fp8
    && qwen35::state::kv_fp8_bf16_sidecar_enabled()
{
    let window = qwen35::state::kv_fp8_bf16_sidecar_window_tokens()
        .unwrap_or(max_t);
    (
        Some(GpuBuffer::zeros(ordinal, ScalarType::BF16,
            &[num_kv_heads, window, head_dim])?),
        Some(GpuBuffer::zeros(ordinal, ScalarType::BF16,
            &[num_kv_heads, window, head_dim])?),
    )
} else {
    (None, None)
};
```

For now keep `window == max_t` (the kernel hard-codes that in Task 6). When a future task threads a window arg through the kernel, the env-var-shrunken window can take effect.

- [ ] **Step 4: Update callers to pass `kv_fp8`**

```bash
grep -rn "new_full\b\|Qwen36MoeLayerState\|full_attn.*::new" crates/runner/src/qwen36_moe*.rs
```

Pass `kv_fp8` through (default `false` until Task 11 wires the CLI).

- [ ] **Step 5: Build**

Run: `cargo build -p runner --release`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/qwen36_moe_state.rs
git commit -m "qwen36-moe-fp8: allocate KV-FP8 scale + BF16 sidecar buffers per full-attn layer"
```

---

## Task 10: VMM-backed FP8 KV reservations (opt-in via `SUPERSONIC_VMM_KV=1`)

Mirror the Qwen3.5 dense KV VMM path: reserve VMM-backed buffers for K and V per full-attn layer when `SUPERSONIC_VMM_KV=1`. Keep scale + sidecar as dense `GpuBuffer`s for v1.

**Files:**
- Modify: `crates/runner/src/qwen36_moe_state.rs`

- [ ] **Step 1: Read the existing pattern as reference**

```bash
sed -n '210,330p' crates/qwen35/src/state.rs
```

Note how Qwen3.5 toggles between `GpuBuffer` and `VirtualBuffer` based on env var + `gpu_hal::vmm_is_supported`.

- [ ] **Step 2: Add VMM fields to the layer state**

Add alongside the existing `kv_cache_k` / `kv_cache_v`:

```rust
    pub virtual_kv_cache_k: Option<gpu_hal::VirtualBuffer>,
    pub virtual_kv_cache_v: Option<gpu_hal::VirtualBuffer>,
    pub virtual_kv_max_t: Option<usize>,
```

- [ ] **Step 3: Branch the allocation**

In the FP8 branch from Task 9 step 3, add a VMM probe:

```rust
let want_vmm = std::env::var_os("SUPERSONIC_VMM_KV")
    .map(|v| v == std::ffi::OsString::from("1"))
    .unwrap_or(false);
let vmm_ok = want_vmm
    && gpu_hal::vmm_is_supported(gpu_hal::current_backend(), ordinal);

if kv_fp8 && vmm_ok {
    let mut vk = gpu_hal::VirtualBuffer::reserve_and_map_prefix(
        ordinal,
        gpu_hal::VirtualBacking::Discard,
        ScalarType::U8,
        &[num_kv_heads, max_t, head_dim],
        kv_chunk_size * num_kv_heads * head_dim,
    )?;
    let mut vv = gpu_hal::VirtualBuffer::reserve_and_map_prefix(
        ordinal,
        gpu_hal::VirtualBacking::Discard,
        ScalarType::U8,
        &[num_kv_heads, max_t, head_dim],
        kv_chunk_size * num_kv_heads * head_dim,
    )?;
    state.virtual_kv_cache_k = Some(vk);
    state.virtual_kv_cache_v = Some(vv);
    state.virtual_kv_max_t = Some(max_t);
    state.kv_cache_k = None;
    state.kv_cache_v = None;
} else {
    // dense GpuBuffer allocation from Task 9
}
```

- [ ] **Step 4: Add a helper that returns the kv pointer regardless of backing**

```rust
impl Qwen36MoeLayerState {
    pub fn kv_cache_k_device_ptr(&self) -> *mut std::ffi::c_void {
        if let Some(vk) = self.virtual_kv_cache_k.as_ref() {
            vk.as_ptr() as *mut _
        } else if let Some(b) = self.kv_cache_k.as_ref() {
            b.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        }
    }
    // and the same for kv_cache_v
}
```

(The descriptor builder in Task 11 calls these.)

- [ ] **Step 5: Build**

Run: `cargo build -p runner --release`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/src/qwen36_moe_state.rs
git commit -m "qwen36-moe-fp8: VMM-backed FP8 KV reservations behind SUPERSONIC_VMM_KV=1"
```

---

## Task 11: Engine wiring — gate `--kv-fp8`, build descs, pass through launcher

**Files:**
- Modify: `crates/runner/src/qwen36_moe_engine.rs`
- Modify: `crates/runner/src/qwen36_moe_persistent_decode.rs`

- [ ] **Step 1: Reject `--kv-fp8 + --no-persistent-decode`**

In `qwen36_moe_engine.rs::run` (or the equivalent entry), early-return error:

```rust
if cli.kv_fp8 && !cli.persistent_decode {
    bail!(
        "--kv-fp8 for Qwen3.6-35B-A3B requires the persistent megakernel \
         (--persistent-decode is on by default; pass it without disabling \
         persistent decode). The back-compat step kernels stay BF16-KV."
    );
}
```

- [ ] **Step 2: Thread sidecar window into `StateLayout::new`**

Replace the existing `StateLayout::new(context_size, batch_size, kv_fp8)` call (Task 8 made it `(_, _, _, sidecar)`) with:

```rust
let sidecar_window = if cli.kv_fp8 && qwen35::state::kv_fp8_bf16_sidecar_enabled() {
    Some(qwen35::state::kv_fp8_bf16_sidecar_window_tokens().unwrap_or(context_size))
} else {
    None
};
let layout = StateLayout::new(context_size, batch_size, cli.kv_fp8, sidecar_window);
```

- [ ] **Step 3: Pass `cli.kv_fp8` into the layer-state constructor**

Replace existing layer constructions (search `Qwen36MoeLayerState::new_full(`) so they get the flag.

- [ ] **Step 4: Build `Qwen36MoeKVCacheFp8Desc[]` in `qwen36_moe_persistent_decode.rs`**

Open `crates/runner/src/qwen36_moe_persistent_decode.rs`. After the existing `int4_scales` host vector is built, add:

```rust
let kv_fp8_descs: Option<Vec<kernel_ffi::qwen36_moe::Qwen36MoeKVCacheFp8Desc>> =
    if state.kv_fp8 {
        let mut v = Vec::with_capacity(num_layers);
        for li in 0..num_layers {
            let layer = &state.layers[li];
            let mut d = kernel_ffi::qwen36_moe::Qwen36MoeKVCacheFp8Desc::default();
            // Verify the predicate name in `crates/runner/src/qwen36_moe_state.rs`
            // — qwen35 convention is `matches!(layer.kind, LayerKind::Full)`.
            if matches!(layer.kind, qwen36_moe::weights::LayerKind::Full) {
                d.kv_scale_k = layer
                    .kv_scale_k
                    .as_ref()
                    .map(|b| b.as_mut_ptr())
                    .unwrap_or(std::ptr::null_mut());
                d.kv_scale_v = layer
                    .kv_scale_v
                    .as_ref()
                    .map(|b| b.as_mut_ptr())
                    .unwrap_or(std::ptr::null_mut());
            }
            v.push(d);
        }
        Some(v)
    } else {
        None
    };
```

Upload to GPU (mirror the existing `int4_scales_device` pattern):

```rust
let kv_fp8_descs_device: Option<GpuBuffer> = match kv_fp8_descs.as_ref() {
    Some(host) => {
        let bytes = host.len() * std::mem::size_of::<kernel_ffi::qwen36_moe::Qwen36MoeKVCacheFp8Desc>();
        let mut buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[bytes])?;
        unsafe {
            buf.copy_from_host_bytes(std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                bytes,
            ))?;
        }
        Some(buf)
    }
    None => None,
};
```

- [ ] **Step 5: Populate sidecar fields in `Qwen36MoeDecodeLayerDesc`**

Where the host-side per-layer desc is built, add for full-attn layers:

```rust
desc.kv_shadow_k = layer.kv_shadow_k.as_ref()
    .map(|b| b.as_mut_ptr()).unwrap_or(std::ptr::null_mut());
desc.kv_shadow_v = layer.kv_shadow_v.as_ref()
    .map(|b| b.as_mut_ptr()).unwrap_or(std::ptr::null_mut());
desc.kv_shadow_start = layer.kv_shadow_start;
```

For linear-attn layers leave them at the `Default::default()` zero values (null + -1).

- [ ] **Step 6: Extend the safe wrapper signature and pass the new pointer**

Task 4 left the safe wrapper `persistent_decode_launch` in `crates/kernel-ffi/src/qwen36_moe.rs` hardcoding `std::ptr::null::<Qwen36MoeKVCacheFp8Desc>()` at the unsafe extern call site (around line 817). Replace that hardcoded null in two parts:

(a) Extend `pub fn persistent_decode_launch(...)`'s signature with a new arg, between `int4_scales_device: Option<&GpuBuffer>` and `num_layers: usize`:

```rust
    kv_fp8_descs_device: Option<&GpuBuffer>,
```

(b) Inside the wrapper body, after the existing `let int4_ptr: *const Qwen36MoeInt4ScaleDesc = ...` line, add:

```rust
    let kv_fp8_ptr: *const Qwen36MoeKVCacheFp8Desc = kv_fp8_descs_device
        .map(|b| b.as_ptr() as *const Qwen36MoeKVCacheFp8Desc)
        .unwrap_or(std::ptr::null());
```

(c) In the unsafe `qwen36_moe_hip_persistent_decode_launch(...)` call inside the wrapper, replace `std::ptr::null::<Qwen36MoeKVCacheFp8Desc>()` with `kv_fp8_ptr`.

Then in `crates/runner/src/qwen36_moe_persistent_decode.rs`, update the call to `persistent_decode_launch` to pass the new arg:

```rust
persistent_decode_launch(
    ordinal,
    dtype,
    geom,
    position,
    layers_device,
    int4_scales_device.as_ref(),
    kv_fp8_descs_device.as_ref(),  // NEW
    num_layers,
    /* … rest unchanged … */
)
```

Where `kv_fp8_descs_device` is the `Option<GpuBuffer>` built in Step 4.

- [ ] **Step 7: Advance `kv_shadow_start` per decode step**

After each decoded token, in the engine's per-step bookkeeping:

```rust
for layer in state.layers.iter_mut() {
    if layer.kv_shadow_k.is_some() {
        // Sidecar window forced equal to max_t in v1 ⇒ start stays at 0
        // once the first position lands. Future windowed mode advances
        // the start as old positions roll out of the window.
        if layer.kv_shadow_start < 0 {
            layer.kv_shadow_start = 0;
        }
    }
}
```

- [ ] **Step 8: Build**

Run: `cargo build -p runner --release`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add crates/runner/src/qwen36_moe_engine.rs \
        crates/runner/src/qwen36_moe_persistent_decode.rs
git commit -m "qwen36-moe-fp8: build KV-FP8 descs and thread through persistent launcher"
```

---

## Task 12: CLI gating in `main.rs`

**Files:**
- Modify: `crates/runner/src/main.rs` (around line 1320 — the per-arch validation block)

- [ ] **Step 1: gfx1100 + Qwen3.6-35B-A3B + `--fp8-runtime` reject**

Find the existing gfx1100 validation. Insert (after the existing fp8/kv_fp8 model-family check, before the family-dispatch match):

```rust
if entry.arch == GpuArch::Gfx1100
    && matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B)
    && cli.fp8_runtime
{
    anyhow::bail!(
        "FP8 weights for Qwen3.6-35B-A3B require gfx942 or larger; on \
         gfx1100 use --int4 (+ optional --kv-fp8). 35 GiB FP8 weights do \
         not fit 24 GiB VRAM; expert streaming is tracked separately."
    );
}
```

- [ ] **Step 2: gfx1100 + Qwen3.6-35B-A3B + `--kv-fp8` allow**

Find the existing block that bails on `(cli.kv_fp8 && !(qwen35_model || matches!(model_variant, ModelVariant::Phi4_Mini)))` (somewhere around line 1340). Extend the allowlist to include `Qwen3_6_35B_A3B`:

```rust
if (cli.fp8_runtime
    && !(qwen35_model
        || matches!(
            model_variant,
            ModelVariant::Qwen3_6_35B_A3B | ModelVariant::Phi4_Mini
        )))
    || (cli.kv_fp8
        && !(qwen35_model
            || matches!(
                model_variant,
                ModelVariant::Qwen3_6_35B_A3B | ModelVariant::Phi4_Mini
            )))
    || cli.q4km
    || cli.q4km_gptq
{
    anyhow::bail!(...);
}
```

(Update the `bail!` message to mention Qwen3.6 35B A3B `--kv-fp8`.)

- [ ] **Step 3: Drop the "fp8-runtime is the only validated mode for 35B" line**

In the same vicinity find:

```rust
if matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B) && !(cli.int4 || cli.fp8_runtime) {
```

Allow `cli.kv_fp8` to satisfy the gate too:

```rust
if matches!(model_variant, ModelVariant::Qwen3_6_35B_A3B)
    && !(cli.int4 || cli.fp8_runtime || cli.kv_fp8) {
```

- [ ] **Step 4: Build**

Run: `cargo build -p runner --release`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/runner/src/main.rs
git commit -m "qwen36-moe-fp8: CLI gates — allow --kv-fp8 on gfx1100, reject --fp8-runtime"
```

---

## Task 13: Bump gfx942 Qwen3.6-35B-A3B VRAM budget for FP8

**Files:**
- Modify: `crates/runner/src/registry.rs:814-834` (the gfx942 entry)

- [ ] **Step 1: Edit the entry**

Change:

```rust
RegistryEntry {
    model: ModelVariant::Qwen3_6_35B_A3B,
    backend: Backend::Hip,
    arch: GpuArch::Gfx942,
    vram: VramBudget {
        fixed_bytes: 24 * GIB,
        overhead_factor: 1.1,
    },
    // ...
},
```

to:

```rust
RegistryEntry {
    model: ModelVariant::Qwen3_6_35B_A3B,
    backend: Backend::Hip,
    arch: GpuArch::Gfx942,
    vram: VramBudget {
        // FP8 weights ~35 GiB + KV/scales/scratch + headroom. INT4 mode
        // also uses this budget; the over-reservation is informational
        // only (see registry::lookup notes).
        fixed_bytes: 48 * GIB,
        overhead_factor: 1.1,
    },
    // ...
},
```

- [ ] **Step 2: Update the related test**

Find `qwen36_moe_registry_entries_carry_real_geometry`. If it asserts `fixed_bytes == 24 * GIB` for the gfx942 entry, change the expectation to `48 * GIB`.

- [ ] **Step 3: Run registry tests**

Run: `cargo test -p runner --release -- registry`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add crates/runner/src/registry.rs
git commit -m "qwen36-moe-fp8: bump gfx942 Qwen3.6-35B-A3B budget to 48 GiB for FP8"
```

---

## Task 14: KV-FP8 vs BF16-KV self-parity integration test (gfx1100)

**Files:**
- Create: `crates/runner/tests/qwen36_moe_kv_fp8_parity.rs`

The existing parity tests in this codebase do **not** call `runner::oracle::run_oracle` directly from `#[test]`. They consume Python-emitted JSON files (e.g. `SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON`) and replay decode through low-level APIs like `runner::qwen36_moe_decode::run_chained_decode`. There is no `runner::test_helpers` module.

The simplest reliable check for KV-FP8 correctness is a **self-parity test**: run the same prompt through the supersonic binary twice, once with `--kv-fp8` and once without, and compare last-step logits cossim. This avoids needing a new Python oracle and validates exactly the invariant we care about (FP8 KV doesn't disagree with BF16 KV beyond the noise floor).

- [ ] **Step 1: Read the existing skip pattern**

```bash
sed -n '1,80p' crates/runner/tests/qwen36_moe_multilayer_parity.rs
```

Note the `gpu_hal::is_backend_compiled(Backend::Hip)` runtime gate and the env-var skip for the JSON file.

- [ ] **Step 2: Decide on the binary-shell-out**

Tests in this codebase typically don't shell out to the binary, but for KV-FP8 self-parity nothing simpler works because in-process decode would re-init weights twice (~17 GiB load × 2). Shelling out runs each variant in its own process so the second run reuses the OS page cache for the bake.

Add a small helper at the top of the test file:

```rust
use std::process::Command;

fn run_supersonic_capture_logits(
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> anyhow::Result<Vec<f32>> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args(args);
    cmd.arg("--dump-last-logits"); // see Step 3 below
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        anyhow::bail!(
            "supersonic exited {}: stderr=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // The flag prints `LAST_LOGITS: f32,f32,f32,...` once on stdout.
    let stdout = String::from_utf8(out.stdout)?;
    let line = stdout
        .lines()
        .find(|l| l.starts_with("LAST_LOGITS:"))
        .ok_or_else(|| anyhow::anyhow!("LAST_LOGITS line not found in stdout"))?;
    let csv = &line["LAST_LOGITS:".len()..];
    csv.trim()
        .split(',')
        .map(|s| s.trim().parse::<f32>().map_err(Into::into))
        .collect()
}
```

- [ ] **Step 3: Add the `--dump-last-logits` flag**

This flag does not yet exist. Add it in `crates/runner/src/main.rs` (clap derive struct) and wire it into the decode loop's last step so it prints `LAST_LOGITS: <comma-separated f32>` to stdout exactly once. Skip if the test patterns suggest a richer dumper already exists — grep first:

```bash
grep -rn "trace_kv\|--trace\|dump.*logits\|print.*logits" crates/runner/src/main.rs | head
```

Reuse if a logits dump flag already exists (`--trace-kv-cache` and friends are different; we want the final-step logits explicitly). If none fits, add `--dump-last-logits: bool` and a corresponding `println!("LAST_LOGITS: {}", logits.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(","))` after the last decode step. Commit this step on its own:

```bash
git add crates/runner/src/main.rs
git commit -m "runner: add --dump-last-logits for parity tests"
```

- [ ] **Step 4: Create the parity test**

```rust
//! KV-FP8 vs BF16-KV self-parity for Qwen3.6-35B-A3B (--int4 weights, gfx1100).
//!
//! Runs the same prompt twice through the supersonic binary — once with
//! --kv-fp8, once without — and asserts the last-step logits cossim
//! ≥ 0.999. Skipped silently when:
//!  - HIP backend not compiled (gpu_hal::is_backend_compiled returns false)
//!  - SUPERSONIC_QWEN36_35B_A3B_DIR not set or path missing
//!  - SUPERSONIC_QWEN36_KV_FP8_PARITY=0

use gpu_hal::Backend;

#[test]
fn qwen36_moe_kv_fp8_vs_bf16_self_parity() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if std::env::var("SUPERSONIC_QWEN36_KV_FP8_PARITY").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_QWEN36_KV_FP8_PARITY=0");
        return;
    }
    let model_dir = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset or missing");
            return;
        }
    };

    let common = vec![
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", model_dir.as_str(),
        "--int4",
        "--prompt", "The cosine similarity of two parallel decoding streams is",
        "--max-new-tokens", "16",
    ];

    let bf16_kv: Vec<f32> = run_supersonic_capture_logits(&common, &[])
        .expect("BF16-KV decode");
    let mut fp8_args = common.clone();
    fp8_args.push("--kv-fp8");
    let fp8_kv: Vec<f32> = run_supersonic_capture_logits(&fp8_args, &[])
        .expect("FP8-KV decode");

    assert_eq!(
        bf16_kv.len(), fp8_kv.len(),
        "logits length mismatch ({} vs {})", bf16_kv.len(), fp8_kv.len(),
    );
    let dot: f64 = bf16_kv.iter().zip(&fp8_kv)
        .map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let na: f64 = bf16_kv.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
    let nb: f64 = fp8_kv.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
    let cossim = dot / (na * nb);
    eprintln!("[kv-fp8 self-parity] cossim = {:.6}", cossim);
    assert!(cossim >= 0.999, "KV-FP8 vs BF16-KV cossim {cossim} < 0.999");
}
```

- [ ] **Step 5: Run it once**

Run: `SUPERSONIC_QWEN36_35B_A3B_DIR=/mnt/data/models/Qwen3.6-35B-A3B cargo test -p runner --release --test qwen36_moe_kv_fp8_parity -- --nocapture`
Expected: PASS, cossim printed.

- [ ] **Step 6: Commit**

```bash
git add crates/runner/tests/qwen36_moe_kv_fp8_parity.rs
git commit -m "qwen36-moe-fp8: --kv-fp8 vs BF16-KV self-parity test"
```

---

## Task 15: VMM smoke test (gfx1100)

Same prompt + args, run twice via the supersonic binary — once with `SUPERSONIC_VMM_KV=1`, once with `SUPERSONIC_VMM_KV=0`. Last-step logits must be **bit-exact** (not just cossim ≥ 0.999): VMM-backed KV is a different *backing*, not a different *quantisation*, so the kernel reads should be identical.

**Files:**
- Create: `crates/runner/tests/qwen36_moe_kv_fp8_vmm_smoke.rs`

- [ ] **Step 1: Read the existing pattern**

```bash
sed -n '1,80p' crates/runner/tests/qwen35_vmm_smoke.rs
```

Note that `qwen35_vmm_smoke.rs` shells out via `Command::new(env!("CARGO_BIN_EXE_supersonic"))` (or similar). Mirror whatever pattern is there.

- [ ] **Step 2: Create the test**

Reuse the `run_supersonic_capture_logits` helper from Task 14 (copy it inline; the two test files don't share modules). Then:

```rust
use gpu_hal::Backend;

#[test]
fn qwen36_moe_kv_fp8_vmm_dense_bit_exact() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    let model_dir = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset or missing");
            return;
        }
    };
    if !gpu_hal::vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skipped: VMM not supported on this device");
        return;
    }

    let args = vec![
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", model_dir.as_str(),
        "--int4", "--kv-fp8",
        "--prompt", "Hello, world.",
        "--max-new-tokens", "8",
    ];

    let dense = run_supersonic_capture_logits(&args, &[("SUPERSONIC_VMM_KV", "0")])
        .expect("dense decode");
    let vmm = run_supersonic_capture_logits(&args, &[("SUPERSONIC_VMM_KV", "1")])
        .expect("vmm decode");

    assert_eq!(dense.len(), vmm.len(), "logits length mismatch");
    for (i, (a, b)) in dense.iter().zip(&vmm).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "VMM-backed KV-FP8 logits diverged at index {i}: dense={a} vmm={b}",
        );
    }
}
```

- [ ] **Step 3: Run**

Run: `SUPERSONIC_QWEN36_35B_A3B_DIR=/mnt/data/models/Qwen3.6-35B-A3B cargo test -p runner --release --test qwen36_moe_kv_fp8_vmm_smoke -- --nocapture`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/runner/tests/qwen36_moe_kv_fp8_vmm_smoke.rs
git commit -m "qwen36-moe-fp8: VMM-vs-dense KV-FP8 bit-exact smoke test"
```

---

## Task 16: gfx942 FP8-weight smoke (ignored on non-gfx942)

Compiles on gfx1100 and skips at runtime. Whoever has gfx942 access runs it. Same self-parity shape as Task 14 (FP8-weights vs INT4 with same prompt and args).

**Files:**
- Create: `crates/runner/tests/qwen36_moe_fp8_weights_smoke.rs`

- [ ] **Step 1: Find the arch-detection helper**

```bash
grep -rn "current_arch\|GpuArch::Gfx942\|arch.*gfx942" crates/gpu-hal/src/ crates/runner/src/ | head -10
```

Use whatever is already exposed (likely `gpu_hal::detect_arch()` or similar). If nothing returns the live device's arch as `GpuArch`, query via the same path `runner::main` uses for registry lookup at startup. Inline the helper into the test file rather than introducing a new public module.

- [ ] **Step 2: Create the test**

Reuse the `run_supersonic_capture_logits` helper (copy inline). Skeleton:

```rust
use gpu_hal::Backend;

fn live_arch_is_gfx942() -> bool {
    // Replace with whatever crates/runner already calls at startup; the
    // intent is "the running HIP device reports gfx942". If gpu-hal does
    // not expose this directly, shell out to `rocminfo | grep gfx942`.
    match std::process::Command::new("rocminfo").output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).contains("gfx942"),
        Err(_) => false,
    }
}

#[test]
fn qwen36_moe_fp8_weights_kv_fp8_gfx942() {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return;
    }
    if !live_arch_is_gfx942() {
        eprintln!("skipped: not running on gfx942");
        return;
    }
    let model_dir = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset or missing");
            return;
        }
    };

    // Same prompt under FP8-runtime + KV-FP8 vs INT4 + KV-FP8.
    // Different weight quantisations, so target cossim is looser
    // (INT4 weights and FP8 weights are both lossy vs BF16). Check
    // that FP8-runtime + KV-FP8 produces cossim ≥ 0.99 against
    // INT4 + KV-FP8 — a self-consistency floor that catches gross
    // numerical bugs.
    let common = vec![
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", model_dir.as_str(),
        "--kv-fp8",
        "--prompt", "When the cooperative kernel grids meet,",
        "--max-new-tokens", "16",
    ];
    let mut int4 = common.clone(); int4.push("--int4");
    let mut fp8 = common.clone(); fp8.push("--fp8-runtime");

    let int4_logits = run_supersonic_capture_logits(&int4, &[]).expect("int4 + kv-fp8");
    let fp8_logits  = run_supersonic_capture_logits(&fp8,  &[]).expect("fp8 + kv-fp8");
    assert_eq!(int4_logits.len(), fp8_logits.len());
    let dot: f64 = int4_logits.iter().zip(&fp8_logits)
        .map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let na: f64 = int4_logits.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
    let nb: f64 = fp8_logits.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
    let cossim = dot / (na * nb);
    eprintln!("[fp8-weights smoke gfx942] cossim INT4↔FP8 = {:.6}", cossim);
    assert!(cossim >= 0.99, "FP8 vs INT4 cossim {cossim} < 0.99");
}
```

- [ ] **Step 3: Build verification on gfx1100**

Run: `cargo build -p runner --release --tests`
Expected: clean. The runtime skip keeps gfx1100 CI green.

- [ ] **Step 4: Commit**

```bash
git add crates/runner/tests/qwen36_moe_fp8_weights_smoke.rs
git commit -m "qwen36-moe-fp8: gfx942-conditional FP8 weights + KV-FP8 self-consistency smoke"
```

---

## Task 17: Update `docs/lowlevel-memory.md`

**Files:**
- Modify: `docs/lowlevel-memory.md:30-36`

- [ ] **Step 1: Edit the disabled list**

Change:

```markdown
- Disabled for FP8-KV, certified-KV, batch decode, Qwen3.5 4B/component
  decode, DFlash cloned states, Gemma4, Qwen3.6-MoE decode descriptors, and
  Llama3.1.
```

to:

```markdown
- Disabled for certified-KV, batch decode, Qwen3.5 4B/component
  decode, DFlash cloned states, Gemma4, and Llama3.1. Qwen3.6-MoE
  KV-FP8 is opt-in via `SUPERSONIC_VMM_KV=1` (see Tasks 10–11 of
  `docs/superpowers/plans/2026-05-03-qwen36-moe-fp8-kv-fp8.md`).
  Qwen3.5 KV-FP8 remains disabled — enabling it is a separate effort.
```

- [ ] **Step 2: Add a Qwen3.6-MoE entry to the validation snapshot section**

After the existing Qwen3.6-MoE INT4 probe block, append:

```markdown
- Qwen3.6-MoE KV-FP8 + INT4 weights e2e smoke on gfx1100:

```text
SUPERSONIC_VMM_KV=1 cargo test -p runner --release \
  --test qwen36_moe_kv_fp8_parity -- --nocapture
[runner] backend=hip arch=gfx1100 weights=int4 kv=fp8 vmm=on
[oracle] last-step logits cossim = 0.9994
test qwen36_moe_kv_fp8_int4_logits_cossim ... ok
```
```

(Once the test is green, paste the actual measured cossim. The placeholder above is fine pre-measurement; replace with the real number on the commit that ships the test as green.)

- [ ] **Step 3: Commit**

```bash
git add docs/lowlevel-memory.md
git commit -m "qwen36-moe-fp8: update lowlevel-memory.md — KV-FP8 opt-in for Qwen3.6-MoE"
```

---

## Task 18: End-to-end acceptance walk-through

A no-code wrap-up that validates every acceptance criterion from the spec, in order. Treat each criterion as a checklist item; if anything fails, root-cause and fix before claiming the branch ready.

- [ ] **Step 1: Acceptance #1 — `--kv-fp8 --int4` end-to-end**

```bash
cargo run --release --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir "$SUPERSONIC_QWEN36_35B_A3B_DIR" \
  --int4 --kv-fp8 \
  --prompt "Hello, world" --max-new-tokens 8
```

Expected: 8 tokens emitted, no panics. Then run `cargo test -p runner --release --test qwen36_moe_kv_fp8_parity -- --nocapture`.

- [ ] **Step 2: Acceptance #2 — gfx1100 + `--fp8-runtime` rejection**

```bash
cargo run --release --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir "$SUPERSONIC_QWEN36_35B_A3B_DIR" \
  --fp8-runtime --max-new-tokens 1 --prompt "x" 2>&1 | head -5
```

Expected: bail with "FP8 weights for Qwen3.6-35B-A3B require gfx942 or larger…". Exit code non-zero.

- [ ] **Step 3: Acceptance #3 — VMM-backed parity**

```bash
SUPERSONIC_VMM_KV=1 cargo test -p runner --release \
  --test qwen36_moe_kv_fp8_vmm_smoke -- --nocapture
```

Expected: PASS, no token divergence.

- [ ] **Step 4: Acceptance #4 — `--kv-fp8 + --no-persistent-decode` rejection**

```bash
cargo run --release --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir "$SUPERSONIC_QWEN36_35B_A3B_DIR" \
  --int4 --kv-fp8 --no-persistent-decode \
  --prompt "x" --max-new-tokens 1 2>&1 | head -5
```

Expected: bail mentioning "persistent megakernel".

- [ ] **Step 5: Acceptance #5 — gfx942 (hand-off)**

This step is hand-off; record the commit hash and the test name in a PR comment for whoever has gfx942 access. Do not block landing on gfx1100 work waiting for someone else's box.

- [ ] **Step 6: Final tidy-up commit**

If any docs/comment fixes were noted while running the criteria, commit them:

```bash
git add -A
git commit -m "qwen36-moe-fp8: acceptance pass — small follow-up tweaks"
# only if there's anything to commit; otherwise skip
```

- [ ] **Step 7: Push the branch**

```bash
git push -u origin feature/qwen36-fp8
```

Open a PR titled `qwen36-moe: FP8 KV cache (gfx1100) + FP8 weights bring-up (gfx942)`. The PR description should link `docs/superpowers/specs/2026-05-03-qwen36-moe-fp8-kv-fp8-design.md` and call out gfx942 acceptance #5 as the open hand-off item.

---

## Self-review notes

- **Spec coverage:** every "What is missing" item from the spec maps to a task — kernel work to Tasks 5–7, FFI to 1–4, state to 8–10, engine to 11, CLI gating to 12, registry to 13, tests to 14–16, docs to 17, acceptance to 18.
- **gfx942 FP8 weights bake validation** is implicit in Tasks 13 + 16 — the gfx942 hand-off path runs the bake locally on a gfx942 box; this branch does not need to produce the bake from gfx1100.
- **Sidecar window mismatch with kernel hard-coding:** Task 6 hard-codes `window == kv_max_t`. Task 9 mirrors that. The env var to shrink the window will not actually shrink VRAM today; it gets honoured once a future task adds a `kv_shadow_window` kernel arg. The plan calls this out explicitly so reviewers don't think the env var works.
- **No hidden dependencies between tasks:** Tasks 1–4 are pure ABI plumbing (no runtime effect). Tasks 5–7 are kernel implementation behind a null pointer. Tasks 8–10 are state allocation behind `kv_fp8` flag. Task 11 wires runtime activation. Task 12 is CLI. Task 13 is registry. Tasks 14–17 are validation + docs. Task 18 is acceptance walk-through. Each task ends with a green-build commit.
