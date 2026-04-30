# gfx1150: GPU L2 bypass on host-mapped memory

This note documents the regression that closed PR #54
(`feat/unified-memory-allocator`) and motivates the per-platform
`BufferPolicy` mapping introduced in its replacement.

## Background

PR #54 routed every `GpuBuffer` allocation on `MemoryArchitecture::Unified`
arches (gfx1150 APU) through `hipHostMalloc(MAPPED) +
hipHostGetDevicePointer` instead of `hipMalloc`. The hypothesis was
"unified memory means the H2D copy is wasted, eliminate it for free."

Empirically on AMD Radeon 890M (RDNA3.5, ROCm 7.0.3) this produced a
~2.2× decode regression on Qwen3.5-0.8B INT4 and ~1.5× on Gemma4-E2B
INT4 — opposite direction. Functional output stayed bit-exact
(`gpu_oracle_max_delta=0.0000`).

## What we tried

- **Drop `COHERENT`, keep `MAPPED`** — no change (regression remained).
- **Add `hipHostGetDevicePointer`** to use device-side pointers per HIP
  API contract — no change. (Still the right thing to do regardless.)
- **rocprof PMC counters** for L2 hit/miss — unsupported on gfx1150
  (`Agent HW architecture is not supported, no counter metrics found`).

## What we measured

`rocprofv3 --kernel-trace` confirms the regression is uniform across
kernels, not isolated to one:

| kernel                                         | discrete (ms) | unified (ms) | U/D |
|---|---:|---:|---:|
| persistent_decode_kernel (8 calls)             | 333.8 | 541.6 | 1.62× |
| matmul_int4_dequant_wmma_small_m_kernel (159×) |  34.2 | 259.8 | 7.58× |
| dozens of tiny kernels (cast/rms_norm/etc)     | each 1.2-4.3× slower | | |
| **TOTAL kernel time**                          | 385.1 | 814.8 | **2.12×** |

Total kernel time matches the ms/tok regression. Every memory-touching
kernel is slower, with the biggest hits on kernels that touch small
slices repeatedly.

## Microbenchmark — root cause

Standalone hipcc bench
([`tests/gfx1150/membench_l2_bypass.hip`](../tests/gfx1150/membench_l2_bypass.hip)),
peak GPU clocks (`power_dpm_force_performance_level=high`):

| test                                    | discrete | unified | ratio |
|---|---:|---:|---:|
| streaming 256 MiB read (no reuse)       | 97 GiB/s | 94 GiB/s | 1.03× |
| streaming 1 GiB read (no reuse)         | 94 GiB/s | 94 GiB/s | 1.00× |
| 4 MiB repeated read (fits L2/MALL)      | 96 GiB/s | **56 GiB/s** | 1.71× |
| 256 KiB repeated read (fits L2)         | 82 GiB/s | **32 GiB/s** | 2.60× |
| 256 buffers × 1 MiB scatter             |  3.5 µs  | **4.7 µs**   | 1.34× |
| 1024 buffers × 256 KiB scatter          |  3.3 µs  | **9.2 µs**   | 2.79× |

**Streaming bandwidth is identical.** Cache-eligible repeated reads
collapse 1.7-2.8× on host-mapped memory. That is the fingerprint of
GPU L2 *not engaging* for `hipHostMalloc(MAPPED)` allocations.

## Mechanism

AMD's APU coherence model treats `hipHostMalloc(MAPPED)` memory as
coarse-grained CPU-coherent — the GPU has to bypass L2 to keep its
view consistent with potential concurrent CPU writes. That guarantee
is what enables the zero-copy "host writes, GPU reads" pattern, but
it costs every GPU read.

There is no "host wrote once, GPU treats as device-cacheable" flag in
`hipHostMalloc` / `hipExtMallocWithFlags` on this stack. Available
flags are `Default` / `Finegrained` / `Uncached` / `Contiguous`; the
combination "host-mapped + GPU-cacheable + relaxed-coherence" simply
isn't exposed.

## Why decode in particular hurts

The persistent megakernel re-reads layer weights within each step,
and dozens of tiny kernels (cast, rms_norm, transpose, conv state
extract, …) operate on small activation slices that fit comfortably
in L2. On `hipMalloc`'d memory every reuse is a cache hit; on
host-mapped memory every load is a DRAM round-trip. That's why even
"compute-bound" tiny kernels regressed — they were quietly cache-bound
and the regression unmasked it.

## What this implies for the abstraction

The Apple-silicon "unified memory" mental model does **not** transfer
to RDNA3.5. Apple's GPU caches host-allocated memory; AMD's APU GPU
does not. RDNA4 laptops may behave differently again — we have no data.

So the right shape isn't "Unified arch ⇒ use host-mapped everywhere"
or "Discrete arch ⇒ never use host-mapped". It's:

- The caller supplies a **`BufferKind`** intent — `Persistent` (weights,
  activations, KV cache; needs L2 reuse) or `Scratch` (one-shot, no
  reuse, save the H2D copy if cheap).
- The platform supplies a **`BufferPolicy`** mapping kind → strategy
  — gfx1150 maps `Scratch → HostMapped`; everywhere else maps both to
  `Default` until measurement says otherwise.

`gpu-hal::ops::alloc` then just executes the platform's resolved
choice. New arches plug in by editing one table; the call sites don't
move.

## Re-running the benchmark

```bash
hipcc --offload-arch=gfx1150 -O3 \
  tests/gfx1150/membench_l2_bypass.hip -o /tmp/membench_l2_bypass

# Pin GPU clocks for repeatable numbers (sysfs is root-owned).
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

/tmp/membench_l2_bypass

echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

If the cache-fitting rates equalize on a future stack (newer ROCm,
RDNA4, etc.), revisit `ArchProfile::buffer_policy` — the Scratch-only
opt-in could become a Persistent default.
