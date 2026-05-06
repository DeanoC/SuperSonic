# Machine Profile — Design Spec

**Date:** 2026-05-06
**Status:** Draft, awaiting user review
**Owner:** DeanoC

## Summary

A new `machine-profile` workspace crate that measures real CPU and GPU
hardware capabilities through explicit microkernels and writes a structured
profile to disk. Two consumers read the profile:

1. **SuperSonic runner** — reads a small subset at startup to select kernel
   variants and tile sizes.
2. **A future offline optimizer** — reads the full profile (and other
   published profiles) to drive autotuning.

The profile carries both *measured* and *theoretical-peak* (catalog-derived)
numbers so consumers can reason in absolute or relative terms.

## Goals

- Measure, not vendor-report. Every roofline number in the profile comes from
  a real microkernel — LDS bandwidth, HBM bandwidth, sustained WMMA TFLOPS,
  PCIe transfer curves on the GPU side; AVX/AVX-512/BF16/AMX/NEON peak
  GFLOPS, DRAM (STREAM-style) bandwidth, cache latency curves on the CPU side.
- Backend-agnostic at the Rust layer; HIP-only kernels in v1 with clean
  trait-based stubs for CUDA and Metal.
- A self-describing, schema-versioned profile that the runner and optimizer
  both consume without ambiguity.
- Stable per-machine fingerprint so cached profiles can be invalidated when
  hardware or drivers change.

## Non-Goals

- Running profiling kernels inside the SuperSonic decode hot path. Profiling
  is offline. Runner only *reads* a cached profile.
- Replacing or reorganising `kernel-lab`. Kernel-lab evaluates SuperSonic's
  application kernels for regression diffing; this crate measures hardware.
- Building the optimizer itself. The spec defines the *interface* (profile
  schema) the optimizer will consume.
- Cross-vendor unification of GPU profiling kernels. Each backend gets its
  own `kernels/` directory for the same reason `kernel-ffi` separates the
  Qwen and Gemma 4 kernels: hipcc/nvcc/Metal codegen are fragile and benefit
  from isolation.

## Approach

A new self-contained workspace crate `crates/machine-profile/` with:

- Its own `kernels/` directory containing HIP `.hip` source for GPU
  microkernels.
- Its own `build.rs` mirroring `kernel-ffi/build.rs`'s arch detection but
  scoped to profiling kernels.
- A library API (`Profile`, `Fingerprint`, etc.) consumed by the runner and
  the future optimizer.
- A CLI binary `machine-profile` for one-shot measurement, inspection, and
  publishing.
- A small static catalog of known device theoretical peaks for ratio
  reporting.

This isolation matches the project's existing pattern of separating fragile
kernel compilation units. Adding profiling kernels to `kernel-ffi` would
risk codegen regressions in the model megakernels, which CLAUDE.md
explicitly warns against.

## Crate Layout

```
crates/machine-profile/
├── Cargo.toml
├── build.rs                       # hipcc compilation gated by supersonic_backend_hip
├── src/
│   ├── lib.rs                     # Profile, Fingerprint, public errors
│   ├── fingerprint.rs             # Blake3 hash of CPU+GPU+driver identifiers
│   ├── store.rs                   # JSON serde, local cache, --publish
│   ├── catalog.rs                 # static device theoretical-peak table
│   ├── cpu/
│   │   ├── mod.rs
│   │   ├── identify.rs            # CPUID / /proc/cpuinfo / AT_HWCAP
│   │   ├── topology.rs            # sockets, P/E cores, NUMA nodes
│   │   ├── cache.rs               # L1/L2/L3 size + measured latency/BW
│   │   ├── vector_kernels.rs      # peak GFLOPS per (ISA, dtype)
│   │   └── dram.rs                # STREAM-style read/write/copy, NUMA-aware
│   └── gpu/
│       ├── mod.rs                 # trait GpuProfiler { fn run(...) -> ... }
│       ├── hip.rs                 # invokes profile kernels via FFI
│       ├── cuda.rs                # stub: NotImplemented
│       └── metal.rs               # stub: NotImplemented
├── kernels/
│   ├── lds_bandwidth.hip
│   ├── hbm_bandwidth.hip
│   ├── wmma_peak.hip
│   ├── pcie_bandwidth.hip
│   └── profile_bridge.cpp         # C ABI launchers
└── src/bin/machine_profile.rs     # CLI binary
```

## Storage & Refresh (Hybrid)

- **Local cache (source of truth at runtime):**
  `~/.supersonic/machine-profile/<fingerprint>.json`
- **Published copy (opt-in, for the optimizer):**
  `<repo>/profiles/<fingerprint>.json`, written via
  `machine-profile run --publish`. Sanitised to remove hostname, absolute
  paths, and any user-identifying fields; carries the same fingerprint and
  measurements.
- **Refresh:** Runner calls `machine-profile::load_or_measure()` at startup.
  If no cached profile matches the current fingerprint, measurement runs
  inline (best-effort, never blocks startup; failures are logged and the
  runner falls back to non-profile-aware defaults). Manual `machine-profile
  run` always rebuilds the cache.

## Profile Schema (v1)

```json
{
  "schema_version": 1,
  "profile_version": "machine-profile/0.1.0",
  "fingerprint": "blake3:...",
  "fingerprint_components": { "cpu": "...", "gpus": ["..."], "driver": "..." },
  "captured_at": "2026-05-06T12:34:56Z",
  "warnings": [],
  "cpu": {
    "vendor": "AMD",
    "model": "Ryzen 9 7950X",
    "stepping": 2,
    "microcode": "0xa601206",
    "isa": ["AVX2", "AVX-512F", "AVX-512BF16", "AVX-VNNI"],
    "topology": {
      "sockets": 1,
      "cores_total": 16,
      "cores_p": 16,
      "cores_e": 0,
      "threads_per_core": 2,
      "numa_nodes": [
        { "id": 0, "cpus": [0, 1, "..."], "ram_bytes": 67108864000 }
      ]
    },
    "cache": {
      "l1d": {
        "size_bytes": 32768,
        "line_bytes": 64,
        "ways": 8,
        "measured_lat_ns": 0.9,
        "measured_bw_gb_s": 1300.0
      },
      "l2": { "...": "..." },
      "l3": { "...": "..." }
    },
    "vector_peak": {
      "fp32": {
        "measured_gflops_per_core": 200.0,
        "measured_gflops_aggregate": 3200.0,
        "theoretical_gflops_aggregate": 3500.0,
        "ratio": 0.91
      },
      "fp16": { "...": "..." },
      "bf16": { "...": "..." }
    },
    "dram": {
      "single_thread_read_gb_s": 28.0,
      "stream_read_gb_s": 75.0,
      "stream_write_gb_s": 60.0,
      "stream_copy_gb_s": 65.0,
      "theoretical_peak_gb_s": 83.2,
      "ratio_copy": 0.78
    }
  },
  "gpus": [
    {
      "backend": "HIP",
      "device_index": 0,
      "arch_name": "gfx1100",
      "pci_id": "0x744c",
      "uuid": "GPU-...",
      "memory_arch": "Discrete",
      "total_vram_bytes": 25757220864,
      "cu_count": 48,
      "wave_size": 32,
      "lds_per_cu_bytes": 65536,
      "lds_bw_per_cu_gb_s": 4500.0,
      "lds_bw_aggregate_gb_s": 215000.0,
      "vram_bw": {
        "read_gb_s": 720.0,
        "write_gb_s": 700.0,
        "copy_gb_s": 690.0,
        "theoretical_peak_gb_s": 800.0,
        "ratio_read": 0.9
      },
      "mma_peak": {
        "f16": { "measured_tflops": 110.0, "theoretical_tflops": 122.8, "ratio": 0.89 },
        "bf16": { "...": "..." },
        "fp8_e4m3": { "...": "..." },
        "i8": { "...": "..." }
      },
      "pcie": {
        "generation": "PCIe 4.0 x16",
        "h2d_gb_s_by_size": [{ "bytes": 1048576, "gb_s": 12.0 }],
        "d2h_gb_s_by_size": [{ "bytes": 1048576, "gb_s": 12.0 }],
        "duplex_gb_s": 22.0
      },
      "clock_rate_khz_measured": 2500000
    }
  ],
  "system": {
    "ram_bytes": 67108864000,
    "os": "linux 6.19.14-300.fc44.x86_64",
    "kernel_driver": "amdgpu 6.10"
  }
}
```

The schema is strictly additive within `schema_version: 1`. Breaking changes
bump the version; consumers must check `schema_version` before deserialising.

## Components & Responsibilities

### `Profile` (lib.rs)
Root struct. `Profile::measure()` orchestrates CPU + GPU passes and is
robust to per-component failure: each microkernel returns a `Result`;
failures populate the field as `null` and append a `Warning { component,
reason }` to `profile.warnings` rather than aborting the whole run.

### `Fingerprint` (fingerprint.rs)
Blake3 hash over a deterministic concatenation of:
- CPU vendor + model + stepping + microcode revision
- For each GPU: backend + arch_name + pci_id + uuid + driver version
- ISA flag set

Public `Fingerprint::components()` exposes the input fields so callers can
decide whether a cached profile is still valid even if hash inputs change in
a future schema version.

### `store.rs`
- `load(fingerprint) -> Option<Profile>` reads
  `~/.supersonic/machine-profile/<fingerprint>.json`.
- `save(&Profile)` writes atomically (temp file + rename).
- `publish_to(repo_root, &Profile)` writes a sanitised copy under
  `<repo_root>/profiles/<fingerprint>.json`. Sanitisation strips:
  `system.os`, `fingerprint_components.driver` if it contains a hostname,
  and any future host-identifying fields.

### `catalog.rs`
Static slice of ~12 entries:
- GPU: gfx1100, gfx1150, gfx90a; sm_86, sm_89, sm_90; Apple M2/M3/M4 GPU.
- CPU: Zen3, Zen4, Raptor Lake, Sapphire Rapids, Apple M-series.

Each entry stores theoretical peak BF16/F16/F32/I8 throughput and theoretical
DRAM/HBM bandwidth. Lookup is by exact arch + pci_id for GPUs and a brand +
model regex for CPUs. A miss returns `None`; the profile leaves
`theoretical_*` as `null` and the optimizer treats them as unknown.

### `cpu/vector_kernels.rs`
One inner loop per (ISA, dtype). Compile-time guarded by `#[cfg(target_feature
= "...")]`; runtime-dispatched via `is_x86_feature_detected!` so a single
binary covers AVX2 / AVX-512F / AVX-512BF16 / AVX-VNNI / AMX-BF16. Inline
assembly is used where intrinsics fail to generate FMA-saturated loops. NEON
and SVE are added behind `target_arch = "aarch64"`.

Each loop runs for a target wall-clock duration (default 200 ms) and reports
peak GFLOPS over a sliding median window. Per-core measurement first, then
saturation across all P-cores using thread pinning (`sched_setaffinity` on
Linux).

### `cpu/dram.rs`
Classic STREAM kernels (Copy, Scale, Add, Triad) plus a single-thread read
microkernel. NUMA-aware: each NUMA node is measured independently; the
profile reports per-node and aggregate numbers.

### `gpu/hip.rs`
Thin Rust orchestrator that uses `gpu-hal` for device enumeration and
allocation, then invokes the four profiling kernels via FFI. Each kernel
runs N = 20 measurement passes after K = 5 warmups; the result is the median
plus p10/p90 of throughput. Aggregate numbers (e.g. `lds_bw_aggregate_gb_s`)
are derived by launching enough wavefronts to saturate the device.

### Profiling kernels (`kernels/*.hip`)
- `lds_bandwidth.hip` — sustained vector load/store on LDS, one block per
  CU.
- `hbm_bandwidth.hip` — saturating HBM read, write, and copy with stride
  patterns covering coalesced and gather access.
- `wmma_peak.hip` — packed 16×16×16 `wmma_f16/bf16/fp8/i8` loops; per-CU
  TFLOPS scaled by `cu_count` for the aggregate.
- `pcie_bandwidth.hip` — H2D, D2H, and duplex transfers across a size sweep
  (4 KiB → 256 MiB) using both pageable and pinned host memory; reports the
  curve.
- `profile_bridge.cpp` — C ABI launchers consumed by `gpu/hip.rs` via FFI.

## Consumer Integration

### Runner
A new `runner::profile` module exposes `load_or_measure() -> Result<Profile>`.
On startup the runner calls this and uses a small read surface:
`memory_arch`, `mma_peak.bf16.measured_tflops`, `vram_bw.read_gb_s`, and
`pcie.duplex_gb_s`. Profile measurement is best-effort: failure logs a
warning and the runner proceeds with non-profile-aware defaults.

### Optimizer (future)
Globs `<repo>/profiles/*.json`, deserialises each, and uses the full schema.
Out of scope for this spec; the design only fixes the interface.

## Error Handling

- Per-microkernel `Result`. A failure adds an entry to `profile.warnings`
  (with component name and reason) and leaves the corresponding fields
  `null`; it does not abort the rest of the run.
- Sanity floors gate publishing: e.g. HBM BW must exceed 50 GB/s and BF16
  WMMA must exceed 1 TFLOPS, otherwise the kernel is treated as failed.
  Floors are tunable per backend.
- IO errors on cache write are non-fatal: the measurement result is returned
  to the caller; the next run will re-measure.

## Testing

- **Unit (host-only):** fingerprint stability and sensitivity; catalog
  lookup hits and misses; profile round-trip serialise → deserialise.
- **Smoke (gated):** `cargo test --features gpu_smoke -- --ignored` runs a
  cut-down (one-second) measurement against the local GPU and asserts the
  sanity floors.
- **Snapshot regression:** once a real measurement is sane, commit
  `profiles/gfx1100-baseline.json`. CI re-runs and compares within a
  tolerance window (e.g. ±10 % on each measured number) to catch
  measurement-code regressions.

## CLI Surface

```
machine-profile run [--publish] [--cpu-only|--gpu-only] [--quick]
machine-profile show [--raw]
machine-profile fingerprint
machine-profile clear-cache
machine-profile bench-only <kernel-name>
machine-profile catalog
```

`--quick` runs each microkernel for 50 ms instead of 200 ms; intended for
smoke testing and CI.

## Build Wiring

`build.rs` in `crates/machine-profile/`:
- Detects HIP arch via `rocminfo` or the `HIP_ARCH` env var, mirroring
  `kernel-ffi/build.rs`.
- Compiles `kernels/*.hip` and `kernels/*.cpp` only when
  `supersonic_backend_hip` is set.
- CUDA/Metal stubs are pure Rust — `build.rs` is a no-op for those backends
  in v1.
- Profiling kernels are isolated under `crates/machine-profile/kernels/` to
  avoid hipcc cross-contamination with `kernel-ffi/kernels/`, per the
  CLAUDE.md guidance about gfx11xx codegen fragility.

## Worktree

Implementation will land in a dedicated git worktree branch
`worktree-machine-profile`. The worktree is created via the repository's
existing `using-git-worktrees` workflow.

## Risks & Open Questions

- **Inline asm portability.** Vector peak kernels rely on FMA-saturated
  loops; if intrinsics don't yield clean codegen we may need inline asm,
  which complicates the build for non-x86_64 hosts. Mitigation: gate inline
  asm behind `#[cfg(target_arch = "x86_64")]` and provide an intrinsic
  fallback.
- **WMMA on gfx1100.** Mixed reports about sustained WMMA throughput on
  RDNA3; the measurement may surface lower-than-theoretical peaks. The
  catalog ratio will make this visible rather than hiding it.
- **NUMA on the dev box.** Single-socket Zen4 has only one node; NUMA paths
  exist for future multi-socket boxes but won't be exercised on the primary
  dev hardware.
- **STREAM compiler optimisation.** Aggressive optimisers can collapse the
  triad loop. Mitigation: use `std::hint::black_box` and `volatile`-style
  patterns where needed.
