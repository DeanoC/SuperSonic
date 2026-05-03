# Low-Level Virtual Memory

SuperSonic now has an internal `gpu-hal` VMM layer for the HIP/CUDA
intersection: reserve device VA, create physical device allocations, map,
set access, unmap, and release. Metal reports unsupported and continues to use
the existing allocator.

The first consumer is Qwen3.5 BF16 dense KV in single-sequence mode. When a
context limit is known and VMM support probes successfully, full-attention K/V
caches reserve stable virtual addresses for the full context. On the current
HIP path Qwen uses separate K and V reservations per full-attention layer.
HIP VMM mappings use a conservative 2 MiB minimum page size even when ROCm
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

- Decode-active VMM is enabled internally for Qwen3.5 BF16 dense KV only.
- Disabled for FP8-KV, certified-KV, batch decode, Qwen3.5 4B/component
  decode, DFlash cloned states, Gemma4, Qwen3.6-MoE decode descriptors, and
  Llama3.1.
- Baked tensors can now be loaded into `VirtualArena` allocations through
  `model-store::BakedStore::load_to_virtual_arena`. This is the first real
  virtual-weight/MoE-island path: it uses the bake mmap as the source of truth,
  assigns allocation roles (`Weights` or `MoeExpert`), maps stable VMM
  addresses, and uploads tensor bytes into those mappings. This is not yet a
  decode-active weight path; existing Qwen3.6-MoE decode descriptors still own
  and consume dense `GpuBuffer` weights.
- `SUPERSONIC_VMM_WEIGHT_PROBE=1` makes Qwen3.6-MoE dry-run load
  `lm_head.weight` into a virtual `Weights` allocation when an INT4 bake is
  present.
- `SUPERSONIC_VMM_MOE_ISLAND_PROBE=1` or
  `SUPERSONIC_VMM_MOE_ISLAND_PROBE_LAYERS=N` makes Qwen3.6-MoE dry-run load
  the first N layers' fused expert slabs into virtual `MoeExpert` allocations
  and report logical/resident/reserved bytes. This is a residency probe only;
  decode still uses the existing `GpuBuffer` descriptors.
- `SUPERSONIC_VMM_KV=0` disables the Qwen3.5 integration.
- `SUPERSONIC_VMM_KV=1` requests it and logs if the backend cannot support it.
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
tests.

## Validation Snapshot

The branch was validated on HIP/gfx1100 with ROCm VMM support:

- Raw HIP VMM repro: repeated sub-2 MiB remap/restore corrupts earlier mappings
  on gfx1100; the same sequence passes with a 2 MiB mapping floor.
- HAL arena tests:
  `SUPERSONIC_BACKENDS=hip cargo test -p gpu-hal --test vmm_round_trip vmm_arena_ -- --nocapture`
- Baked tensor VMM upload test:
  `SUPERSONIC_BACKENDS=hip cargo test -p model-store virtual_arena_loads_baked_weight_and_expert_tensors -- --nocapture`
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
