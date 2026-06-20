# Qwen3.6 Lucebox +20% Performance Guide

This guide defines the measurement and implementation path for getting
SuperSonic's Qwen3.6-27B DFlash path to 20% above the current `main`
baseline on the local RX 7900 XTX (`gfx1100`) machine.

Current `main` already includes the merged PR #261 work. The important
boundary is that baseline and profile data must be regenerated from current
`main`, not read from stale local artifacts:

```bash
git log --oneline --decorate -8
```

The local `main` history currently includes:

```text
954e2dc Merge pull request #261 from DeanoC/codex/qwen36-dflash-verify-batch
3e3a7ba Gate Q6_K MMQ lm_head by WMMA support
cf165a8 Align Qwen3.6 DFlash with Lucebox
```

Older `target/qwen36_he_supersonic_*.json` files from before PR #261 must be
treated as stale. In particular, runs that report `fused verify does not fit`
and fall back to sequential target verify are useful historical breadcrumbs,
not valid current baselines.

## Target

The optimization target is based on fresh current-`main` SuperSonic numbers:

```text
target_tok_s = current_main_mean_tok_s * 1.20
```

Use the full Lucebox HumanEval 10-prompt Qwen3.6-27B DFlash suite with
`n_gen=256`. Report Lucebox separately as the reference comparison. Lucebox's
top-level README advertises RX 7900 XTX HIP around `50 tok/s`, while
`server/README.md` notes that `gfx1100` prefers DDTree `budget=8`. The local
`server/docs/HIP_PERF_PLAN.md` says that budget shift lifted Lucebox's canonical
gfx1100 run from `49.81 tok/s` to `76.02 tok/s`, so use the local Lucebox run
as the source of truth before treating any static number as a hard target.

Track these values in every report:

- SuperSonic mean tok/s, min tok/s, max tok/s, and per-prompt rows.
- Lucebox mean tok/s under the same machine state.
- Acceptance length or accepted tokens per round.
- `1.2x` target tok/s derived from the fresh SuperSonic baseline.
- Whether prefill is excluded from decode tok/s.

Do not accept a mean-only win if one or two prompts collapse. The follow-up PR
needs prompt-level evidence.

## Current Snapshot

As of the first implementation pass after PR #261, direct DDTree rollback is no
longer blocked by the old F32 recurrent-trace allocation failure. The tree path
now uses the same compressed rollback trace policy as append rollback:

- Default HIP trace dtype: Q8, unless `SUPERSONIC_DFLASH_DISABLE_Q8_ROLLBACK_TRACE`
  is set.
- Fallback HIP trace dtype: BF16, unless
  `SUPERSONIC_DFLASH_DISABLE_BF16_ROLLBACK_TRACE` is set.
- Debug fallback: F32 when both compressed trace modes are disabled.

Fresh local result with direct DDTree rollback:

```text
file: target/qwen36_lucebox20/direct_q8_tree_10x256.json
config: budget=8, top_k=8, direct_rollback=1, n_gen=256, 10 prompts
mean: 32.18 tok/s
min:  24.46 tok/s
max:  43.07 tok/s
target_1p2x: 38.62 tok/s
```

Current implementation result after the tree-verify speedup work:

```text
file: target/qwen36_lucebox20/tree_tiled_scratch_budget14_top4_chain_10x256.json
config: budget=14, top_k=4, direct_rollback=1, chain_seed=on, n_gen=256, 10 prompts
mean: 38.69 tok/s
min:  27.64 tok/s
max:  67.02 tok/s
target_1p2x: 38.62 tok/s
```

Per-prompt result:

| Prompt | tok/s |
|---|---:|
| has_close_elements | 35.83 |
| separate_paren_groups | 24.46 |
| truncate_number | 26.43 |
| below_zero | 36.67 |
| mean_absolute_deviation | 32.53 |
| intersperse | 36.34 |
| parse_nested_parens | 32.62 |
| filter_by_substring | 29.13 |
| sum_product | 43.07 |
| rolling_max | 24.76 |

The previous append-reverify fallback run at budget 8 completed only 8 of 10
prompts and averaged about `5.26 tok/s`; it is no longer a useful performance
baseline, though it remains a safety fallback. Direct rollback is now the
required baseline mode for this guide.

The internal profile for the direct run shows verify as the dominant component.
Example rows:

```text
has_close_elements: rounds=42 mean_accepted_per_round=6.10
  draft=855ms verify=6083ms rollback=140ms decode=7146ms
rolling_max: rounds=61 mean_accepted_per_round=4.20
  draft=1236ms verify=8830ms rollback=173ms decode=10338ms
```

The first implementation phase is therefore Track B: reduce tree verify cost
and improve acceptance on low rows. The current `1.2x` target from this baseline
is `38.62 tok/s`, before comparing against a fresh Lucebox run on the same
machine state.

## Baseline Measurement

Start from a clean tree and rebuild the release binary:

```bash
cd /home/deano/projects/SuperSonicBase
git status --short --branch
HIP_ARCH=gfx1100 cargo build --release
```

Capture the machine state before performance runs:

```bash
cat profiles/gfx1100-baseline.json
rocm-smi
rocm-smi --showclocks --showpower --showtemp --showfan
```

Run the current-`main` SuperSonic baseline:

```bash
SUPERSONIC_DFLASH_DDTREE_VERIFY=1 \
SUPERSONIC_DFLASH_DDTREE_BUDGET=14 \
SUPERSONIC_DFLASH_DDTREE_TOP_K=4 \
SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1 \
python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
  --binary target/release/supersonic \
  --model qwen3.6-27b \
  --model-dir /mnt/data/tmp/supersonic-qwen36-27b-lucebox \
  --quant q4km \
  --backend hip \
  --context-size 512 \
  --n-gen 256 \
  --dflash \
  --dflash-draft-dir /mnt/data/tmp/qwen36-27b-dflash-q8-bf16 \
  --ignore-eos \
  --out-json target/qwen36_lucebox20/current_main_supersonic_budget14_top4.json
```

If direct rollback fails, first check whether compressed rollback traces were
disabled in the environment. A failure in direct mode is a correctness or memory
regression; do not replace the baseline with append-reverify numbers without
calling that out explicitly.

Then run the Lucebox reference from the checked-out repo. Keep its model and
draft paths matched to the SuperSonic run:

```bash
cd /home/deano/projects/lucebox-hub/server
python3 scripts/bench_he.py \
  --ddtree-budget 8 \
  --n-gen 256
```

If Lucebox requires different local model arguments, record the exact command
and explain the difference in the comparison notes. The comparison is only
valid when prompt set, generation length, model quant, draft quant, backend, and
thermal state are materially equivalent.

## Sweep And Profile

Run a narrow DDTree sweep first. Lucebox says `gfx1100` prefers budget 8, but
SuperSonic's implementation should prove that locally:

```bash
mkdir -p target/qwen36_lucebox20
for budget in 4 8 12 16 22 28 36; do
  SUPERSONIC_DFLASH_DDTREE_VERIFY=1 \
  SUPERSONIC_DFLASH_DDTREE_BUDGET="$budget" \
  SUPERSONIC_DFLASH_DDTREE_TOP_K=8 \
  SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1 \
  python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
    --binary target/release/supersonic \
    --model qwen3.6-27b \
    --model-dir /mnt/data/tmp/supersonic-qwen36-27b-lucebox \
    --quant q4km \
    --backend hip \
    --context-size 512 \
    --n-gen 256 \
    --dflash \
    --dflash-draft-dir /mnt/data/tmp/qwen36-27b-dflash-q8-bf16 \
    --ignore-eos \
    --out-json "target/qwen36_lucebox20/supersonic_budget${budget}.json"
done
```

Repeat the best budget with `SUPERSONIC_DFLASH_DDTREE_TOP_K=4` and with
`SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK` unset. Keep whichever combination
improves mean tok/s without hurting prompt-level acceptance.

For the best candidate, collect SuperSonic's internal attribution:

```bash
SUPERSONIC_DFLASH_DDTREE_VERIFY=1 \
SUPERSONIC_DFLASH_DDTREE_BUDGET=14 \
SUPERSONIC_DFLASH_DDTREE_TOP_K=4 \
SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1 \
SUPERSONIC_DFLASH_PROFILE_DRAFT=1 \
SUPERSONIC_DFLASH_PROFILE_APPEND=1 \
SUPERSONIC_DFLASH_PROFILE_VERIFY=1 \
SUPERSONIC_DFLASH_PROFILE_FFI=1 \
SUPERSONIC_DFLASH_TRACE_ACCEPT=1 \
python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
  --binary target/release/supersonic \
  --model qwen3.6-27b \
  --model-dir /mnt/data/tmp/supersonic-qwen36-27b-lucebox \
  --quant q4km \
  --backend hip \
  --context-size 512 \
  --n-gen 256 \
  --dflash \
  --dflash-draft-dir /mnt/data/tmp/qwen36-27b-dflash-q8-bf16 \
  --ignore-eos \
  --tail-chars 12000 \
  --out-json target/qwen36_lucebox20/supersonic_profile_best.json
```

Use `rocprofv3` for external attribution. Start with timeline-level evidence:

```bash
mkdir -p target/qwen36_lucebox20/rocprof
export PATH="$PATH:/opt/rocm/bin"
SUPERSONIC_DFLASH_DDTREE_VERIFY=1 \
SUPERSONIC_DFLASH_DDTREE_BUDGET=14 \
SUPERSONIC_DFLASH_DDTREE_TOP_K=4 \
SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK=1 \
rocprofv3 \
  --kernel-trace \
  --memory-copy-trace \
  --memory-allocation-trace \
  --stats \
  --summary \
  --output-directory target/qwen36_lucebox20/rocprof \
  --output-file supersonic_best \
  --output-format csv json \
  -- \
  python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
    --binary target/release/supersonic \
    --model qwen3.6-27b \
    --model-dir /mnt/data/tmp/supersonic-qwen36-27b-lucebox \
    --quant q4km \
    --backend hip \
    --context-size 512 \
    --n-gen 256 \
    --dflash \
    --dflash-draft-dir /mnt/data/tmp/qwen36-27b-dflash-q8-bf16 \
    --ignore-eos \
    --limit 1 \
    --out-json target/qwen36_lucebox20/supersonic_rocprof_one_prompt.json
```

After identifying the hottest kernels, run targeted counter passes with
`rocprofv3 --pmc`. Keep counter sets small; ROCprofiler can fail if all
counters cannot be collected in one pass. Use `rocprofv3 --list-avail` to see
available counters on the installed ROCm stack.

On this machine `rocprofv3` was not on `PATH` during the first implementation
pass. Install or expose the ROCm profiler before treating the guide's profiling
gate as complete.

## Bottleneck Decisions

Use the measurements to pick exactly one first implementation track.

| Finding | First track |
|---|---|
| Acceptance length trails Lucebox at similar per-round cost | Tune DDTree budget, top-k, temperature, chain seed, and direct rollback. Compare per-prompt AL before touching kernels. |
| Verify dominates decode time | Optimize `prefill_tree_verify`, tree rollback, visibility handling, and lm-head. Avoid reintroducing sequential verify. |
| Draft dominates decode time | Profile draft attention, MLP, fuser, lm-head, and tap copies. Reduce kernel launches and memory traffic before changing algorithm shape. |
| FFI, sync, allocation, or D2H time dominates | Remove per-round allocations, unnecessary `reset_sync` boundaries, and host transfers from the decode loop. |
| One kernel dominates and counters show memory stalls | Improve coalescing, LDS use, and data layout; reduce redundant global reads. |
| One kernel dominates and counters show low occupancy | Reduce register pressure, tune `__launch_bounds__`, reduce per-block shared memory, and check `hipcc --resource-usage`. |

AMD's HIP performance guidance is the frame for kernel work:

- Use roofline thinking to distinguish compute-bound, memory-bound, and
  overhead-bound paths.
- Hide latency with enough eligible waves and independent work.
- Watch register pressure because excessive registers reduce occupancy.
- Use `__launch_bounds__` and `hipcc --resource-usage` to control and inspect
  register allocation.
- Prefer block sizes that avoid partial waves on Radeon `gfx1100` wave32 paths.

## Implementation Tracks

### Track A: DDTree acceptance tuning

Use this track when SuperSonic's per-prompt acceptance length is below Lucebox.

Expected changes:

- Keep the existing env-var surface stable:
  `SUPERSONIC_DFLASH_DDTREE_VERIFY`,
  `SUPERSONIC_DFLASH_DDTREE_BUDGET`,
  `SUPERSONIC_DFLASH_DDTREE_TOP_K`,
  `SUPERSONIC_DFLASH_DDTREE_TEMP`, and
  `SUPERSONIC_DFLASH_DDTREE_DIRECT_ROLLBACK`.
- Add new controls only if a measured experiment needs them and their default
  behavior is off.
- Promote the winning `gfx1100` default only after the 10-prompt suite shows
  better mean tok/s and no prompt-level collapse.

Acceptance:

- Mean tok/s moves toward the `1.2x` target.
- Per-prompt AL explains the improvement.
- SuperSonic remains comparable to Lucebox on prompt set and generation length.

### Track B: Tree verify and rollback cost

Use this track when verify or rollback dominates the internal profile.

Expected changes:

- Keep tree verification batched. Do not fall back to sequential target verify.
- Attribute `prefill_tree_verify` into full attention, linear attention, MLP,
  lm-head, tap capture, rollback capture, and rollback apply.
- Eliminate redundant visibility/parent processing if it is repeated per layer.
- Reuse buffers across rounds; avoid per-round allocation and D2H transfers.
- Compare append-reverify against direct tree rollback on the same prompt set.

Acceptance:

- Verify milliseconds per accepted token falls materially.
- `SUPERSONIC_DFLASH_TRACE_ACCEPT=1` shows stable or improved acceptance.
- Prompt-level outputs remain deterministic under greedy decode.

### Track C: Draft forward cost

Use this track when draft time is the largest remaining component.

Expected changes:

- Profile the draft fuser, attention, MLP, lm-head, argmax, tap copy, and tree
  probe sections.
- Reduce host-visible intermediate copies first.
- Fuse or batch small draft-side kernels only after profiles show launch
  overhead or memory traffic is material.

Acceptance:

- Draft milliseconds per round drops without reducing acceptance length.
- No extra target-side work is added to hide a draft regression.

### Track D: HIP kernel occupancy and memory path

Use this track when `rocprofv3` counters point at one or two hot kernels.

Expected changes:

- Compile candidate kernels with resource reporting and record register/LDS
  changes.
- Tune launch bounds and block sizes for `gfx1100` wave32 behavior.
- Move temporary per-thread arrays to shared memory only when it improves
  occupancy and does not create a worse LDS bottleneck.
- Improve coalescing and reduce redundant global memory reads before adding
  more arithmetic.

Acceptance:

- The hot kernel improves in isolation and in the full 10-prompt suite.
- No regression appears in existing HIP parity tests.

## PR Gate

A performance PR pursuing this guide is ready when it includes:

- Fresh current-`main` baseline JSON.
- Fresh Lucebox reference command and result.
- DDTree sweep table.
- Internal profile for the selected configuration.
- `rocprofv3` trace or counter evidence for any kernel-level claim.
- New result with mean tok/s at least `1.2x` the fresh SuperSonic baseline.
- Per-prompt table showing no hidden collapse.
- Correctness/parity tests for any changed verifier, rollback, or kernel code.

If the PR does not hit `1.2x`, it can still land as an enabling optimization
only if the report explains the new bottleneck and updates the next track.

## References

- SuperSonic DFlash reference: [dflash.md](../dflash.md)
- SuperSonic performance headline table: [performance.md](../performance.md)
- Lucebox local repo: `/home/deano/projects/lucebox-hub/README.md`
- Lucebox AMD notes: `/home/deano/projects/lucebox-hub/server/README.md`
- Lucebox gfx1100 HIP perf plan:
  `/home/deano/projects/lucebox-hub/server/docs/HIP_PERF_PLAN.md`
- AMD rocprofv3 usage:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>
- AMD HIP performance optimization:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html>
- AMD HIP performance guidelines:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html>
