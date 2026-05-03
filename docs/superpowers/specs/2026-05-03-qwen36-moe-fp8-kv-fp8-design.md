# Qwen3.6-35B-A3B FP8 + FP8 KV — design

Branch: `feature/qwen36-fp8`
Target boxes: AMD gfx1100 (RX 7900 XTX, 24 GiB) and AMD gfx942 (MI300X-class).
Status: design approved 2026-05-03 (brainstorming).

## Goal

Land FP8-runtime weights and FP8 KV cache for Qwen3.6-35B-A3B in
SuperSonic, scoped to:

- **gfx942**: full FP8 weight path + FP8 KV cache. End-to-end validated.
- **gfx1100**: FP8 KV cache combinable with `--int4` weights (FP8 weights
  do not fit 24 GiB and require a separate streaming-experts effort,
  tracked elsewhere). `--fp8-runtime` is rejected on gfx1100 with a
  clear "use --int4 + --kv-fp8" error.

Out of scope (explicitly tracked as follow-ups):

- gfx1100 FP8-weight residency (35 GiB on a 24 GiB card; needs
  VMM-streamed expert islands and a chained-launch FP8 path).
- KV-FP8 for the back-compat non-persistent `qwen36_moe.hip` step
  kernels — `--kv-fp8` requires the persistent megakernel.
- KV-FP8 for any other model family (Qwen3.5 KV-FP8 stays as is —
  this branch does not flip its `lowlevel-memory.md` "disabled" status).

## What is already in place

Substantial pieces of the FP8 weight path landed before this branch
opened. The remaining FP8-weight work on gfx942 is mostly bake +
budget + validation, not new plumbing.

- **Persistent megakernel FP8 weight matvec**:
  `kernels/qwen36_moe_persistent/{helpers,full_attn_phase,linear_attn_phase,ffn_phase,lm_head_phase}.cuh`
  already implement `fp8_matvec_partial` / `fp8_dequant_scalar`. All
  four phases gate on `int4_group_size < 0` (FP8 mode, block_size =
  `-int4_group_size`). Validated structurally; bake-fed end-to-end run
  is what's missing on this branch.
- **Engine FP8 weight wiring**: `qwen36_moe_engine.rs` has
  `Qwen36WeightMode::Fp8`, loads `model_store::bake_dir_fp8`, and
  populates `_scale_inv` pointers in all three sidecar builders
  (`FullAttnInt4Sidecars`, `LinearAttnInt4Sidecars`, `FfnInt4Sidecars`)
  with `group_size = -QWEN36_MOE_FP8_BLOCK_SIZE` (= -128).
- **Bake script**: `oracle/bake_fp8.py` has a qwen3.6-MoE branch
  (`raw_fallback_qwen36_moe_fp8` at line 406) that fuses per-expert
  HF FP8 tensors into the `gate_up_proj` / `down_proj` slabs and
  emits Fp8Native layout with `_scale_inv` companions.
- **VMM HAL** (`crates/gpu-hal/src/vmm.rs`, landed PR #141): role-tagged
  `VirtualArena` (`KvCache`, `Weights`, `MoeExpert`, ...), reserve +
  map_range_bytes + evict_to_host (CpuBackup) + restore_backup,
  validated on HIP/gfx1100 with a 2 MiB page floor. Qwen3.5 dense KV
  is the working reference consumer.
- **KV-FP8 sizing in qwen3.6-MoE**: `crates/qwen36_moe/src/state.rs`
  already accepts `kv_fp8: bool` for byte sizing. Engine threads
  `cli.kv_fp8` through `StateLayout::new`. No kernel-side quant/dequant
  yet — that is the new work here.

## What is missing (the work)

Grouped by component.

### 1. KV-FP8 in the persistent megakernel

`kernels/qwen36_moe_persistent/full_attn_phase.cuh` is the only file
that needs new dequant/quant code (linear-attn layers have no KV
cache; FFN doesn't touch KV; lm_head doesn't touch KV).

Quant scheme (mirrors `kernels/full_attention_4b.hip`):

- K and V cache stored as FP8 E4M3 bytes, shape
  `[num_kv_heads, max_T, head_dim]` U8.
- Scales `kv_scale_k`, `kv_scale_v` shape `[num_kv_heads, max_T]` F32,
  one absmax-derived scale per (head, position).
- Optional BF16 sidecar `[num_kv_heads, window, head_dim]` for
  parity-sensitive reads of recent positions; range gated by
  `kv_shadow_start`. Default window = `max_T` (full sidecar);
  `SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW` shrinks it,
  `SUPERSONIC_DEBUG_DISABLE_KV_FP8_BF16_SIDECAR` disables it.
- Write path: when the layer's `kv_scale_k` is non-null, the phase
  computes the head's absmax over the new-position vector, divides
  to scale, casts each element to FP8 E4M3 via `float_to_fp8_e4m3`,
  writes the FP8 byte to the cache and the scale to
  `kv_scale_*[head, pos]`. If a sidecar is configured for the
  position, the BF16 source is also written to the sidecar.
- Read path: at attention-time, when `kv_scale_k` is non-null and
  `t >= kv_shadow_start`, read BF16 from the sidecar; otherwise
  dequantize FP8 byte × scale via `fp8_e4m3_to_float`.

The `full_attention_4b.hip` implementation is the reference — same
struct fields, same write-then-attend ordering, same sidecar-window
gating. We lift the math, not the surrounding kernel structure.

### 2. KV-FP8 descriptors in `kernel-ffi/src/qwen36_moe.rs`

Add a parallel struct mirroring `FP8ScaleDesc`'s pattern:

```rust
#[repr(C)]
pub struct Qwen36MoeKVCacheFp8Desc {
    pub kv_scale_k: *mut c_void,    // [num_kv_heads, max_T] F32
    pub kv_scale_v: *mut c_void,
}
```

And extend `Qwen36MoeDecodeLayerDesc` with optional sidecar fields
(`kv_shadow_k`, `kv_shadow_v`, `kv_shadow_start`) — these are null
for layers with KV-FP8 disabled or for linear-attn layers.

`qwen36_moe_hip_persistent_decode_launch` gains a new
`kv_fp8_descs: *const Qwen36MoeKVCacheFp8Desc` parameter (null when
`--kv-fp8` is off). Bridge validates: a non-null array must accompany
non-null sidecar pointers in the layer descs for full-attn layers.

### 3. Qwen3.6-MoE state allocates KV-FP8 buffers

`crates/qwen36_moe/src/state.rs` (and the per-layer state owned by
the runner) needs:

- KV cache buffers as U8 `[num_kv_heads, max_T, head_dim]` when `kv_fp8`.
- Scale buffers `kv_scale_k`, `kv_scale_v` as F32 `[num_kv_heads, max_T]`.
- Optional BF16 sidecar `kv_shadow_k`, `kv_shadow_v`
  `[num_kv_heads, window, head_dim]` when sidecar enabled;
  `kv_shadow_start` cursor tracking the first position covered.

Helpers re-used from `crates/qwen35/src/state.rs`:

- `qwen35::state::kv_fp8_bf16_sidecar_enabled()` env gate.
- `qwen35::state::kv_fp8_bf16_sidecar_window_tokens()` env gate.

Both are already `pub fn` in `crates/qwen35/src/state.rs`; the
qwen3.6-MoE state module imports them directly. No duplication, no
shared crate. The env-var names and semantics are intentionally
shared across model families so a user setting
`SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW=512` once gets the
same window everywhere.

### 4. VMM-backed FP8 KV reservations

Following `crates/qwen35/src/state.rs` and `qwen35_vmm_smoke.rs`:

- For each full-attn layer (every 4th of 40 — 10 layers total) reserve
  one VMM allocation per K and per V via `VirtualArena` with role
  `VirtualAllocationRole::KvCache`.
- Logical bytes = `num_kv_heads * max_T * head_dim * 1` (FP8). Mapped
  prefix at the configured kv_chunk_size, grown via `map_range_bytes`
  as decode advances.
- Scale buffers and BF16 sidecars stay as dense `GpuBuffer`s for v1
  (small footprint, no growth pattern, no eviction value).
- Gate path on `SUPERSONIC_VMM_KV` env var. HIP auto-enables when the
  variable is unset and VMM support probes successfully; `0` disables and
  `1` forces/request on any backend. CUDA still requires the explicit opt-in.
- `lowlevel-memory.md` is updated to remove "FP8-KV disabled" from the
  Qwen3.6-MoE row.

### 5. Engine + CLI wiring

`crates/runner/src/qwen36_moe_engine.rs`:

- Reject `--kv-fp8` without `--persistent-decode` (the back-compat
  step kernels stay BF16-KV).
- Build `Qwen36MoeKVCacheFp8Desc[]` from the layer state when
  `cli.kv_fp8`; null otherwise.
- Pass new descriptor pointer to
  `qwen36_moe_hip_persistent_decode_launch`.
- Surface eviction/probe stats from `VirtualArena` in dry-run output
  so we can see KV residency without running validation.

`crates/runner/src/main.rs`:

- gfx1100 + Qwen3_6_35B_A3B + `--kv-fp8`: allow (with `--int4` or
  bare BF16 weights — though BF16 weights won't fit, that's an
  existing constraint, not new).
- gfx1100 + Qwen3_6_35B_A3B + `--fp8-runtime`: reject with
  `"FP8 weights for Qwen3.6-35B-A3B require gfx942 or larger; on
  gfx1100 use --int4 (+ optional --kv-fp8). 35 GiB FP8 weights do
  not fit 24 GiB VRAM; expert streaming is tracked separately."`
- gfx942 + Qwen3_6_35B_A3B + `--fp8-runtime`: allow. Combinable with
  `--kv-fp8`.

### 6. Registry entries

`crates/runner/src/registry.rs`:

- Existing gfx1100 entry: bump VRAM budget if FP8-KV needs more
  scratch (KV scales add ~`num_kv_heads * max_T * 8` bytes/layer ×
  10 full-attn layers — small). Keep the 19 GiB budget unless
  measurement shows otherwise.
- gfx942 entry currently 24 GiB INT4. **Bump the same entry's
  `fixed_bytes` to 48 GiB.** `registry::lookup` keys on
  `(model, backend, arch)` and returns the first match, so we cannot
  add a second entry distinguished only by weight mode without
  changing the lookup signature — out of scope for this spec. The
  budget is a pre-flight informational check, not a hard cap; using
  the FP8 worst-case for both modes only causes a wider VRAM
  reservation log line on the INT4 path, no behavioural change.

### 7. Python bake validation (gfx942 path)

`oracle/bake_fp8.py` already supports qwen3.6-MoE in the raw
fallback. Validation work:

- Run it against `Qwen/Qwen3.6-35B-A3B-FP8` end-to-end and verify
  the resulting `weights.bin` + manifest size matches the analytic
  FP8 byte total.
- Add `python3 oracle/bake_fp8.py --self-verify` reading back a
  random sample of expert tensors and checking against the source
  HF safetensors.

KV-FP8 validation does not need a new oracle: the existing
`qwen36_moe_oracle.py` emits HF reference logits, and the parity
test asserts cossim on the final-step logits. Intermediate KV state
is not inspected.

### 8. Tests

- `crates/runner/tests/qwen36_moe_kv_fp8_parity.rs` — runs
  `--kv-fp8` decode for ≥ 16 tokens against the HF oracle reference
  and asserts last-step logits cossim ≥ 0.999. Uses the existing
  `qwen36_moe_oracle.py` harness through `crates/runner/src/oracle.rs`.
- `crates/runner/tests/qwen36_moe_kv_fp8_vmm_smoke.rs` — boots the
  KV-FP8 path with `SUPERSONIC_VMM_KV=1` on HIP, asserts decoded
  tokens match the dense-`GpuBuffer` path bit-exactly (the same
  invariant `qwen35_vmm_smoke.rs` enforces).
- `crates/runner/tests/qwen36_moe_fp8_weights_smoke.rs`
  (gfx942-conditional, ignored on gfx1100) — boots
  `--fp8-runtime --kv-fp8`, decodes 16 tokens, asserts cossim ≥ 0.999.

## Acceptance criteria

The branch is ready to land when, on this gfx1100 box:

1. `--kv-fp8 --int4` decodes Qwen3.6-35B-A3B end-to-end and the
   parity test passes.
2. `--fp8-runtime` returns the expected error.
3. `SUPERSONIC_VMM_KV=1 --kv-fp8 --int4` decodes and matches the
   dense-buffer path.
4. KV-FP8 + `--persistent-decode=false` returns the expected error.

And on gfx942 (validated by whoever has access — the gfx1100 box
cannot run gfx942):

5. `--fp8-runtime --kv-fp8` decodes and the parity test passes.

## Risk / open questions

- **VMM-backed FP8 KV is novel for this codebase.** Qwen3.5 KV-FP8
  was deliberately *not* on VMM. We're enabling it here because the
  shape is identical to dense KV (just smaller-byte tensors). If
  anything goes sideways during integration, fall back to dense
  `GpuBuffer` for FP8 KV and keep VMM gated to a follow-up.
- **Sidecar window default = full max_T.** Mirrors the 4B path. The
  full-attn KV is 10 layers × `num_kv_heads * head_dim * max_T * 2`
  BF16 bytes; for typical decode contexts (max_T ≤ 8192) this is
  small relative to weights/scratch. Measure during bring-up; if
  it's tight, set the env-var window to a few hundred recent
  positions and accept the parity hit on older positions.
- **gfx942 VRAM budget = 48 GiB** is a guess. Real measurement on
  the bake will pin it; budget is reportable, not enforced beyond
  pre-flight.
- **Persistent kernel + KV-FP8 ordering.** `full_attn_phase.cuh`
  writes the new K/V before reading past K/V. Quantize-then-write
  must precede the attention read of any t < cache_pos. Mirror
  4B's barrier placement; do not invent a new ordering.
