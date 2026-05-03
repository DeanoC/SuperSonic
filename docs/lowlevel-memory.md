# Low-Level Virtual Memory

SuperSonic now has an internal `gpu-hal` VMM layer for the HIP/CUDA
intersection: reserve device VA, create physical device allocations, map,
set access, unmap, and release. Metal reports unsupported and continues to use
the existing allocator.

The first consumer is Qwen3.5 BF16 dense KV in single-sequence mode. When a
context limit is known and VMM support probes successfully, full-attention K/V
caches reserve stable virtual addresses for the full context. On HIP and CUDA,
Qwen uses separate K and V reservations per full-attention layer. CUDA remains
opt-in with `SUPERSONIC_VMM_KV=1` for this first runtime lane. HIP VMM mappings
use a conservative 2 MiB minimum page size even when ROCm
reports a smaller recommended granularity; raw HIP repro tests showed repeated
sub-2 MiB remap/restore corrupts earlier mappings on gfx1100, while the same
sequence with 2 MiB mappings survives. The floor can be overridden for
experiments with `SUPERSONIC_HIP_VMM_MIN_MAP_BYTES=0`. `VirtualBuffer` therefore
distinguishes logical tensor bytes from physical VMM page bytes: `len_bytes`,
`logical_resident_bytes`, and `logical_backup_bytes` describe model data, while
`reserved_bytes`, `resident_bytes`, and `mapped_bytes` describe page-aligned
VMM state. Kernel descriptors still receive raw pointers while the virtual KV
path is active, and the base pointer no longer changes on KV growth.

The allocator-facing layer is intentionally small. `VirtualArena` owns named
`VirtualAllocation`s, each tagged with a role (`KvCache`, `Weights`,
`MoeExpert`, `Scratch`, or `Other`) and backed by a `VirtualBuffer`. This keeps
low-level VMM bookkeeping in the HAL while higher-level residency policies can
aggregate logical bytes, page-resident bytes, backup bytes, and mapping counts
by allocation role.

Current scope:

- Decode-active VMM is enabled internally for Qwen3.5 BF16 dense KV and for
  Qwen3.6-MoE INT4 routed expert slabs when the VMM expert mode is active.
- Disabled for FP8-KV, certified-KV, batch decode, Qwen3.5 4B/component
  decode, DFlash cloned states, Gemma4, and Llama3.1.
- Baked tensors can now be loaded into `VirtualArena` allocations through
  `model-store::BakedStore::load_to_virtual_arena`. `BakedStore` also exposes
  split reserve/upload APIs so a tensor can reserve a stable virtual base while
  only selected byte ranges are made resident.
- Qwen3.6-MoE has a sparse routed-expert residency manager in
  `runner::qwen36_moe_residency`. It reserves the fused
  `mlp.experts.gate_up_proj` / `mlp.experts.down_proj` slabs, pins the VMM
  backing pages that contain routed `(layer, expert, projection)` slices from
  the mmap-backed bake, evicts with a bounded page-LRU policy, and invalidates
  every logical resident slice covered by an unmapped VMM page.
- Qwen3.6-MoE sparse expert residency is backend-aware at the VMM policy layer
  and is unit-tested on supported HIP/CUDA VMM backends. Full Qwen3.6-MoE
  decode kernels are still HIP-only; CUDA runtime decode remains blocked before
  kernel launch.
- `SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=N` activates router-driven sparse MoE
  islands for Qwen3.6-MoE INT4 decode. The chained decode path runs FFN stage 1
  first, downloads the `top_k` expert ids, pins those experts' gate/up and down
  slices, then runs the normal stage-5 FFN launch against the stable virtual
  slab pointers. The cap is expressed in experts; after all sparse expert
  tensors are registered, SuperSonic derives the VMM page budget from the
  actual worst-case gate/up and down page footprint for one routed expert.
  Logical slice hit/miss counters still track router demand; page hit/miss
  counters track physical residency. Sparse islands copy the full VMM backing
  page for any touched expert slice, so every expert sharing that resident page
  contains real bake data rather than zero fill. Sparse islands currently
  disable the persistent megakernel for that run; persistent router prefetch
  needs a future split or in-kernel residency protocol.
- `SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON=/path/report.json` records sparse
  MoE residency telemetry for Qwen3.6-MoE runs: per-forward hits, misses,
  uploads, evictions, resident slices, resident pages, page-backed slice count,
  resident physical bytes, plus summary peaks. This is intended for tuning
  `SUPERSONIC_MOE_ISLAND_CAP_EXPERTS` against real prompts instead of relying
  on one-token smoke numbers.
- On the HIP/gfx1100 validation machine, page-budgeted sparse MoE passed the
  two-token Qwen3.6-MoE smoke at the default 256-expert cap, the smallest
  top-k-sized 8-expert cap, and a 320-expert cap (`640` resident VMM pages).
  The earlier high-cap page-not-present fault was tied to slice-budgeting more
  logical residents than the physical VMM page working set could represent.
- `SUPERSONIC_VMM_WEIGHT_PROBE=1` makes Qwen3.6-MoE dry-run load
  `lm_head.weight` into a virtual `Weights` allocation when an INT4 bake is
  present.
- `SUPERSONIC_VMM_MOE_ISLAND_PROBE=1` or
  `SUPERSONIC_VMM_MOE_ISLAND_PROBE_LAYERS=N` makes Qwen3.6-MoE dry-run load
  the first N layers' fused expert slabs into virtual `MoeExpert` allocations
  and report logical/resident/reserved bytes without running decode.
- `SUPERSONIC_VMM_MOE_ISLANDS=0` disables Qwen3.6-MoE decode-active virtual
  expert slabs. When unset, Qwen3.6-MoE INT4 decode auto-enables them on
  supported HIP devices and falls back to dense expert buffers if VMM loading
  fails. `SUPERSONIC_VMM_MOE_ISLANDS=1` makes unsupported or failed VMM expert
  loading a hard error.
- `SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=N` switches Qwen3.6-MoE INT4 decode from
  fully resident virtual expert slabs to sparse router-prefetched islands with
  at most `N` experts' two routed projections tracked resident at once.
- `SUPERSONIC_VMM_KV=0` disables the Qwen3.5 integration.
- `SUPERSONIC_VMM_KV=1` requests it and logs if the backend cannot support it.
  HIP may auto-enable when unset; CUDA requires this explicit opt-in.
- `SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL=1` backs virtual KV to host RAM after
  prefill, unmaps the device pages, reports zero resident bytes, then restores
  decode state before decode. By default this uses a compact logical-prefix
  backup and materializes regular dense `GpuBuffer` K/V caches for decode.
- `SUPERSONIC_VMM_KV_RESTORE_TO_VMM=1` keeps the virtual K/V caches active
  after eviction. This path uses exact full-image HAL CPU backups, remaps the
  same virtual address ranges, restores them in two phases, and leaves kernel
  descriptors pointing at stable VMM addresses.
- `SUPERSONIC_FORCE_VMM_UNSUPPORTED=1` forces the fallback path for tests.

The HAL type also carries explicit `CpuBackup` versus `Discard` backing tags.
The Qwen3.5 dense KV proof reserves VMM-backed caches and uses compact CPU
logical backups for the opt-in eviction policy. The lower-level HAL still
exercises real D2H backup, unmap/release, remap, and H2D restore in focused HIP
and CUDA tests.

## Validation Snapshot

The branch was validated on HIP/gfx1100 with ROCm VMM support. CUDA coverage
uses the same HAL VMM primitives and the Qwen3.5 ignored smoke test can be run
on CUDA devices with `SUPERSONIC_VMM_KV=1`:

- Raw HIP VMM repro: repeated sub-2 MiB remap/restore corrupts earlier mappings
  on gfx1100; the same sequence passes with a 2 MiB mapping floor.
- HAL arena tests:
  `SUPERSONIC_BACKENDS=hip cargo test -p gpu-hal --test vmm_round_trip vmm_arena_ -- --nocapture`
- CUDA HAL VMM primitive tests:
  `SUPERSONIC_BACKENDS=cuda cargo test -p gpu-hal --test cuda_vmm_round_trip -- --nocapture`
- Baked tensor VMM upload test:
  `SUPERSONIC_BACKENDS=hip cargo test -p model-store virtual_arena_loads_baked_weight_and_expert_tensors -- --nocapture`
- Qwen3.6-MoE sparse residency manager tests:
  `SUPERSONIC_BACKENDS=hip cargo test -p runner qwen36_moe_residency -- --nocapture`
- CUDA Qwen3.6-MoE sparse residency manager tests:
  `SUPERSONIC_BACKENDS=cuda cargo test -p runner qwen36_moe_residency -- --nocapture`
- Qwen3.6-MoE sparse router-prefetch smoke:
  `SUPERSONIC_BACKENDS=hip SUPERSONIC_VMM_MOE_ISLANDS=1 SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=8 cargo run --release --bin supersonic -- --backend hip --model qwen3.6-35b-a3b --model-dir /mnt/data/models/Qwen3.6-35B-A3B --int4 --prompt "Hello" --max-new-tokens 1`
  generated `[11]` and reported `resident=32.00MiB` for the 15 GiB routed
  expert VA reservation after the run.
- Qwen3.6-MoE sparse-vs-dense VMM e2e gate:
  `SUPERSONIC_BACKENDS=hip SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR=/mnt/data/models/Qwen3.6-35B-A3B cargo test --release -p runner --test qwen36_moe_sparse_vmm_smoke -- --ignored --nocapture`
  compares dense virtual expert slabs with sparse router-prefetched slabs over
  multiple generated tokens and validates the telemetry JSON cap/peak fields.
  The default test cap is 256 experts; override with
  `SUPERSONIC_TEST_QWEN36_MOE_SPARSE_CAP_EXPERTS=8` for the smallest
  top-k-sized residency window, or with a larger value when investigating HIP
  live-mapping limits.
- Qwen3.5 virtual KV e2e with eviction and restore-to-VMM:
  `SUPERSONIC_VMM_KV=1 SUPERSONIC_VMM_KV_EVICT_AFTER_PREFILL=1 SUPERSONIC_VMM_KV_RESTORE_TO_VMM=1 ... --validate`
  passed and kept stable VMM pointers through decode.
- Qwen3.6-MoE release bake probe against
  `/mnt/data/models/Qwen3.6-35B-A3B/.supersonic/v2-int4-gptq`:

```text
[INT4 baked package]
  weights.bin:     18.47 GiB    (1374 tensors indexed)
  INT4 / Raw:      15.89 GiB / 2.58 GiB
  ready-for-decode: YES

  [VMM residency probe]
    allocations:      3    mappings=3
    logical/resident: 626.50 MiB / 628.00 MiB
    logical resident: 626.50 MiB    reserved=628.00 MiB
    - lm_head.weight
    - model.layers.0.mlp.experts.gate_up_proj
    - model.layers.0.mlp.experts.down_proj
```

- Full Qwen3.6-MoE HF snapshot under `/mnt/data/models/Qwen3.6-35B-A3B`
  matches Hub file sizes: 40 files, 26 safetensor shards, 66.99 GiB. Dry-run
  reports 1045 safetensor tensors and 64.56 GiB, matching analytic BF16
  accounting.
