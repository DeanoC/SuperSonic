# Qwen3.6 Long-Context VMM/KV-FP8 Profile Plan - 2026-05-04

This branch profiles the merged long-context full-attention tiled path at larger
contexts, with emphasis on dense VMM BF16 KV, KV-FP8, sparse MoE VMM, and
sparse+KV-FP8.

## Worktree

```text
/home/deano/projects/SuperSonicBase-qwen36-longctx-vmm-fp8-profiles
branch: research/qwen36-longctx-vmm-fp8-profiles
base: 3cd856d Merge pull request #205 from DeanoC/perf/qwen36-longctx-full-attn
```

## GPU Coordination

Another local agent is using the same GPU intermittently. Do not launch a
profile row unless `rocm-smi --showuse --showmemuse --showpidgpus` shows:

- GPU use at or below 5%
- VRAM use at or below 5%
- no listed GPU PIDs

The wrapper `tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py` enforces this
before each benchmark row.

## Primary Profile

Start with the large VMM/KV-FP8 profile:

```bash
SUPERSONIC_BACKENDS=hip cargo build --release --bin supersonic

python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile vmm-fp8-large \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --max-new-tokens 4 \
  --timeout 2400 \
  --out-dir target/qwen36_longctx_profiles/vmm-fp8-large
```

This runs one row at a time:

| Context | Modes |
|---:|:---|
| 8192 | `int4-vmm`, `int4-kv-fp8` |
| 16384 | `int4-vmm`, `int4-kv-fp8` |

If the 16384 rows are stable and memory headroom is acceptable, extend to:

```bash
python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile vmm-fp8-xl \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --max-new-tokens 4 \
  --timeout 3600 \
  --out-dir target/qwen36_longctx_profiles/vmm-fp8-xl
```

## Sparse Follow-Up

After dense VMM/KV-FP8 rows, run sparse MoE VMM rows:

```bash
python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile sparse-vmm-fp8 \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --sparse-caps 320 \
  --max-new-tokens 4 \
  --timeout 3000 \
  --out-dir target/qwen36_longctx_profiles/sparse-vmm-fp8
```

This adds `cap320` and `cap320-kv-fp8` rows at 8192 and 16384.

## Readout

For each context, compare:

- prefill seconds
- generation wall ms
- total ms/token
- `full_attn_ms_avg`, `linear_attn_ms_avg`, `ffn_ms_avg`
- total VMM resident GiB
- MoE VMM resident GiB
- KV VMM resident GiB
- generated ID agreement against `int4-vmm`

The main question is whether KV-FP8 reduces resident memory enough at 16k/32k to
offset any decode slowdown, and whether sparse MoE VMM changes the bottleneck
from full-attention back to FFN/residency.
