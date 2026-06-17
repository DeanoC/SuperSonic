# Qwen3.6 Sparse VMM Long-Context Regression - 2026-05-04

Branch: `perf/qwen36-sparse-vmm-longctx-regression`  
Worktree: `/home/deano/projects/SuperSonicBase-qwen36-sparse-vmm-regression`  
Base: `535ed60` (`Merge pull request #206 from DeanoC/research/qwen36-longctx-vmm-fp8-profiles`)

## Starting Point

The previous profile showed this 8k result:

| Mode | Prefill s | Decode ms/tok | Full-attn ms/tok | Total resident GiB | MoE resident GiB |
|:---|---:|---:|---:|---:|---:|
| `int4-vmm` | 313.01 | 56.27 | 55.82 | 15.16 | 15.00 |
| `cap320` | 1135.73 | 179.79 | 164.66 | 1.41 | 1.25 |

The headline problem was that sparse `cap320` cut residency dramatically but
appeared to make full attention 3x slower.

## Finding

The sparse `cap320` path uses segmented persistent decode:

1. lookahead prefetch
2. router-only persistent launch
3. route D2H
4. demand page-in/remap
5. FFN-only persistent launch

Before this branch, `run_sparse_with_expert_prefetch` recorded the entire
segmented path wall time in `DecodeOutputs.kernel_full_attn_us`. That made
`full_attn_ms_avg` include router D2H, MoE residency page-in/remap, and FFN
resume work. The previous `full_attn_ms_avg=164.66` for `cap320` is therefore
not a valid full-attention measurement.

## Change

`run_sparse_with_expert_prefetch` now reports:

- `kernel_full_attn_us`: router-only launch plus top-k route download.
- `kernel_ffn_us`: lookahead prefetch, demand prefetch/page-in, and FFN-only
  launch time.

This does not make sparse faster by itself. It makes the stage timing useful
for the next pass, where the real question is how much time is in residency
page-in/remap versus FFN-only kernels.

The follow-up timing split adds:

- `lookahead_prefetch_ms_avg`
- `router_launch_ms_avg`
- `route_d2h_ms_avg`
- `demand_prefetch_ms_avg`
- `ffn_launch_ms_avg`

These are printed as `[qwen36-moe sparse-breakdown]` under
`--emit-stage-timings` and captured in the long-context benchmark JSON as
`sparse_breakdown`.

The long-context profiling harness now also exposes the existing sparse
runtime policy knobs:

- `--sparse-prefetch`
- `--sparse-prefetch-ranks`
- `--sparse-prefetch-transition-min-obs`
- `--sparse-protected-experts`
- `--sparse-protect-demand`
- `--sparse-async-prefetch`
- `--sparse-async-staging-pages`
- `--sparse-prefetch-evict`
- `--sparse-prefetch-evict-min-prob`
- `--sparse-hot-protect-min-hits`
- `--sparse-fixed-hot-experts`
- `--sparse-fixed-hot-min-hits`

The GPU-idle wrapper forwards those flags to each row so policy variants can be
run without hand-editing environment variables.

## Validation

- `rustfmt --check crates/runner/src/qwen36_moe/persistent_decode.rs`
- `cargo test -p runner qwen36_moe::vmm_config --lib`
- `cargo check -p runner --bin supersonic`
- `SUPERSONIC_BACKENDS=hip cargo build --release --bin supersonic`
- `python3 -m py_compile tests/gfx1100/bench_qwen36_longctx.py tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py`
- parser smoke for `[qwen36-moe sparse-breakdown]`
- 8k `cap320` repro under GPU-idle gating:

```bash
python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile sparse-vmm-fp8 \
  --contexts 8192 \
  --modes sparse \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --sparse-caps 320 \
  --max-new-tokens 4 \
  --timeout 3000 \
  --max-mem-use 8 \
  --out-dir target/qwen36_sparse_vmm_regression/cap320-attribution
```

Corrected 8k `cap320` attribution:

| Mode | Prefill s | Decode ms/tok | Chain ms/tok | Full/router ms/tok | FFN/residency ms/tok | Total resident GiB |
|:---|---:|---:|---:|---:|---:|---:|
| `cap320` | 1146.35 | 166.40 | 163.06 | 48.19 | 114.43 | 1.41 |

The corrected run confirms sparse decode is still slow, but it is not a
full-attention regression. The dominant bucket is FFN/residency. Compared with
the dense 8k profile (`56.27 ms/tok` total), sparse is still 2.96x slower, but
the next optimization target is page-in/remap and FFN-only segmented execution.

Sub-bucket 8k `cap320` repro:

| Mode | Decode ms/tok | Chain ms/tok | Router launch ms/tok | Route D2H ms/tok | Demand prefetch ms/tok | FFN launch ms/tok |
|:---|---:|---:|---:|---:|---:|---:|
| `cap320` | 182.94 | 169.49 | 46.91 | 1.85 | 108.10 | 12.14 |

This identifies the main sparse decode cost as demand page-in/remap, not the
FFN-only kernel. Lookahead prefetch was effectively zero (`0.04 ms/tok`) in
this run, which means the current transition prefetch policy is not hiding the
demand residency work for the measured generation tokens.

Residency telemetry explains why:

| Mode | Prefetch requests | Prefetch skipped | Prefetch uploaded bytes | Page misses | Uploaded bytes | Unmapped bytes |
|:---|---:|---:|---:|---:|---:|---:|
| `cap320` | 2,285,846 | 2,052,482 | 0 | 4,350,230 | 9,123,093,544,960 | 9,121,751,367,680 |

Transition lookahead requested many candidates, but the resident page budget
was already full. It therefore skipped prefetch uploads and left demand
page-in/remap to do roughly 9.1 TB of uploads/unmaps over the 8k profile.

Async transition prefetch repro:

```bash
python3 tests/gfx1100/profile_qwen36_longctx_vmm_fp8.py \
  --profile sparse-vmm-fp8 \
  --contexts 8192 \
  --modes sparse \
  --binary target/release/supersonic \
  --model-dir /mnt/data/models/Qwen3.6-35B-A3B \
  --sparse-caps 320 \
  --max-new-tokens 4 \
  --timeout 3000 \
  --max-mem-use 8 \
  --sparse-prefetch transition \
  --sparse-prefetch-ranks 4 \
  --sparse-async-prefetch \
  --sparse-async-staging-pages 32 \
  --out-dir target/qwen36_sparse_vmm_regression/cap320-transition-r4-async32
```

| Mode | Decode ms/tok | Chain ms/tok | Router launch ms/tok | Route D2H ms/tok | Demand prefetch ms/tok | FFN launch ms/tok | Async scheduled pages | Async capacity skips |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32` | 180.92 | 167.53 | 46.78 | 1.83 | 106.25 | 12.15 | 0 | 2,052,080 |

Async prefetch did not schedule any pages at `cap320` because the page budget
was full. This confirms that the next optimization target is prefetch capacity
or eviction policy, not merely enabling the async prefetch stream.

This branch adds an opt-in prefetch eviction experiment:

- Runtime env: `SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT=1`
- Harness flag: `--sparse-prefetch-evict`
- Progress logging:
  - Harness: `--heartbeat-seconds N`
  - Runner: `--progress-heartbeat-seconds N`

Default behavior is unchanged: prefetch remains non-disruptive and skips when
the page budget is full. With the opt-in enabled, prefetch may use the existing
LRU page eviction path to admit lookahead pages. Telemetry now records
`prefetch_evicted_pages` so the run can distinguish admitted prefetches from
capacity skips.

The corrected async-evict run was not wedged; progress logs showed steady
token-by-token prefill. However, the policy is still a regression:

| Mode | Decode ms/tok | Chain ms/tok | Lookahead prefetch ms/tok | Demand prefetch ms/tok | FFN ms/tok | Async scheduled pages | Async capacity skips | Prefetch hits | Prefetch evicted pages |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32-evict` | 263.41 | 212.56 | 96.71 | 103.96 | 11.90 | 2,259,176 | 0 | 74 | 2,259,176 |

This confirms the async scheduling fix works mechanically, but unconditional
prefetch eviction destroys reuse: prefetch hit rate is effectively zero, page
misses rise to 6.37M, and total upload/unmap traffic rises to about 13.36 TB.
Do not enable this policy by default. The next useful experiment should make
prefetch admission selective, for example by protecting the current/recent
demand working set or requiring transition confidence before evicting.

The follow-up implementation keeps `SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT=1`
opt-in, but makes transition-mode eviction selective:

- Runtime env: `SUPERSONIC_MOE_ISLAND_PREFETCH_EVICT_MIN_PROB`
- Harness flag: `--sparse-prefetch-evict-min-prob`
- Default threshold: `0.90`

Transition candidates below the threshold may still prefetch into spare page
capacity, but they no longer evict resident pages. Previous-token modes keep the
old all-or-nothing eviction behavior because they do not have predictor
confidence. This gives the next profile a bounded way to test async page-in
without letting low-confidence lookahead churn the whole expert cache.

Selective eviction at the default `0.90` threshold is mechanically effective:
the run schedules only 333 async prefetch pages instead of 2.26M, and prefetch
eviction drops from 2.26M pages to 333 pages. It also improves the end-to-end
row over no-evict async prefetch, but the dominant cost is still demand page-in:

| Mode | Decode ms/tok | Chain ms/tok | Lookahead prefetch ms/tok | Demand prefetch ms/tok | FFN ms/tok | Async scheduled pages | Async capacity skips | Prefetch hits | Prefetch evicted pages |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32-evict-p0.90` | 169.97 | 167.61 | 0.04 | 107.19 | 11.89 | 333 | 2,053,694 | 231,813 | 333 |

The result suggests high-confidence transition eviction is safe but too sparse
to solve the regression by itself. The next optimization target should be the
demand miss stream: either increase/reshape the resident budget, protect a
larger recent demand window, or use an explicit recent-expert cache/admission
policy before spending more effort on raw async prefetch throughput.

The next experiment adds demand-route protection:

- Runtime env: `SUPERSONIC_MOE_ISLAND_PROTECT_DEMAND=1`
- Harness flag: `--sparse-protect-demand`
- Requires an explicit `SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS` /
  `--sparse-protected-experts` budget.

When enabled, every current demand route is inserted into the existing protected
page LRU instead of only routes repeated from the previous token. The policy does
not increase resident capacity; it only changes which pages are preferred as
eviction victims within the configured sparse cap. This makes it a direct test
of whether the demand miss stream is caused by evicting too much of the recent
working set.

The `protect128` run does not improve performance:

| Mode | Decode ms/tok | Chain ms/tok | Demand prefetch ms/tok | Page misses | Protected pages | Protect hits | Protect demotions |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32-evict-p0.90` | 169.97 | 167.61 | 107.19 | 4,349,883 | 0 | 0 | 0 |
| `cap320-transition-r4-protect128-protect-demand-async-s32-evict-p0.90` | 176.43 | 173.97 | 113.48 | 4,348,625 | 256 | 4,573,440 | 4,446,481 |

Demand protection barely reduces page misses, while adding heavy protected-page
churn. The issue is therefore not simply that recent demand pages are losing
LRU priority. The next useful direction is admission/placement by layer or
expert frequency: keep a small persistent hot set and let the rest remain
demand-paged, instead of protecting every current route.

The hot-expert follow-up adds a thresholded protection policy:

- Runtime env: `SUPERSONIC_MOE_ISLAND_HOT_PROTECT_MIN_HITS`
- Harness flag: `--sparse-hot-protect-min-hits`
- Requires an explicit `SUPERSONIC_MOE_ISLAND_PROTECTED_EXPERTS` /
  `--sparse-protected-experts` budget.

The runner keeps per-layer expert hit counts during the session. Once a demand
route reaches the configured hit threshold, the route is admitted into the
existing protected-page LRU. This avoids protecting every transient route, while
still testing whether a small persistent hot expert set can stabilize the sparse
resident working set.

The `hot32 protect128` run is directionally better than protecting all demand
routes, but it still does not beat selective `p0.90`:

| Mode | Decode ms/tok | Chain ms/tok | Demand prefetch ms/tok | Page misses | Uploaded TB | Prefetch hits | Async scheduled pages | Protect hits | Protect demotions |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32-evict-p0.90` | 169.97 | 167.61 | 107.19 | 4,349,883 | 9.12 | 231,813 | 333 | 0 | 0 |
| `cap320-transition-r4-protect128-protect-demand-async-s32-evict-p0.90` | 176.43 | 173.97 | 113.48 | 4,348,625 | 9.12 | 232,188 | 333 | 4,573,440 | 4,446,481 |
| `cap320-transition-r4-protect128-hot32-async-s32-evict-p0.90` | 173.00 | 170.59 | 110.31 | 4,337,493 | 9.10 | 255,047 | 193 | 3,998,572 | 3,881,511 |

Hot protection reduces page misses by about 12k versus selective `p0.90`, and
increases prefetch hits by about 23k, but the protected-page churn remains high
and latency regresses by 3.0 ms/tok. The next useful optimization should stop
using protection as the primary admission mechanism and instead reshape the
resident budget itself: layer-aware caps, expert-frequency placement at load
time, or a small fixed hot set that does not churn through the protected LRU.

The next follow-up adds a fixed hot expert page set:

- Runtime env: `SUPERSONIC_MOE_ISLAND_FIXED_HOT_EXPERTS`
- Runtime env: `SUPERSONIC_MOE_ISLAND_FIXED_HOT_MIN_HITS`
- Harness flags: `--sparse-fixed-hot-experts`,
  `--sparse-fixed-hot-min-hits`

This path still uses per-layer demand-route hit counts, but it no longer routes
hot experts through the protected-page LRU. Once a route reaches the configured
threshold, its resident pages are admitted into a separate fixed-hot set until
that budget is full. Fixed-hot pages are lower eviction priority than regular
and protected pages, and repeated hot routes do not refresh an LRU timestamp or
demote another hot page.

The fixed-hot runs are the best sparse rows in this set:

| Mode | Decode ms/tok | Chain ms/tok | Demand prefetch ms/tok | Page misses | Uploaded TB | Prefetch hits | Async scheduled pages | Fixed-hot pages | Fixed-hot skipped | Protect demotions |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cap320-transition-r4-async-s32-evict-p0.90` | 169.97 | 167.61 | 107.19 | 4,349,883 | 9.12 | 231,813 | 333 | 0 | 0 | 0 |
| `cap320-transition-r4-protect128-hot32-async-s32-evict-p0.90` | 173.00 | 170.59 | 110.31 | 4,337,493 | 9.10 | 255,047 | 193 | 0 | 0 | 3,881,511 |
| `cap320-transition-r4-fixedhot128h32-async-s32-evict-p0.90` | 167.77 | 165.50 | 105.26 | 4,109,961 | 8.62 | 199,453 | 333 | 256 | 3,627,587 | 0 |
| `cap320-transition-r4-fixedhot256h16-async-s32-evict-p0.90` | 165.41 | 161.32 | 99.53 | 3,818,669 | 8.01 | 365,753 | 319 | 512 | 3,630,763 | 0 |
| `cap320-transition-r4-fixedhot256h16-repeat1-async-s32-evict-p0.90` | 175.87 | 162.23 | 100.81 | 3,828,534 | 8.03 | 358,763 | 319 | 512 | 3,642,237 | 0 |
| `cap320-transition-r4-fixedhot256h32-async-s32-evict-p0.90` | 167.07 | 164.69 | 104.42 | 3,771,173 | 7.91 | 405,377 | 333 | 512 | 3,283,299 | 0 |
| `cap320-transition-r4-fixedhot256h64-async-s32-evict-p0.90` | 171.59 | 158.42 | 98.34 | 3,701,641 | 7.76 | 457,533 | 333 | 512 | 2,715,389 | 0 |

Fixed-hot admission with a 128-expert budget cuts page misses by about 240k
versus selective `p0.90` and reduces upload/unmap traffic by about 0.50 TB,
while avoiding the protected LRU demotion churn entirely. Increasing the
fixed-hot budget to 256 experts improves again: page misses fall by about 579k
versus selective `p0.90`, upload/unmap traffic falls by about 1.21 TB, and
decode improves to 167.07 ms/tok at `h32`.

The threshold sweep at the 256-expert budget shows the best single decode
latency at `h16`: 165.41 ms/tok with demand prefetch down to 99.53 ms/tok. The
`h16` repeat keeps the same residency profile and similar chain time
(`162.23 ms/tok`), but total decode regresses to 175.87 ms/tok because
non-chain overhead rises (`embed_ms_avg=11.20`). Treat `h16` as promising for
residency/chain time, but not yet a stable total-token timing. `h64` minimizes
page misses and upload traffic further, but total decode also regresses to
171.59 ms/tok because non-chain overhead rises (`embed_ms_avg=11.09`). That row
also briefly overlapped with another GPU PID during prefill, so treat it as
useful directional evidence rather than a final timing.

The attempted `fixedhot320h16` capacity row did not reach prefill. Heartbeats
showed only the bench PID, GPU use at 100%, flat memory at 59%, and no
`phase=prefill step=...` progress for more than two minutes after `phase=session
done`; it was interrupted at 153.8s and released the GPU. Do not spend more time
on `fixedhot320h16` until the session/prefill transition has finer progress
instrumentation or a smaller reproduction. The high fixed-hot skipped count
means the 256-expert budget already fills early and then rejects most later hot
admissions, which is expected for a fixed set. The next useful work is to
understand the decode-stage embed timing spikes, then rerun the best two stable
rows (`fixedhot256h16` and `fixedhot256h32`) with that timing artifact isolated.

Repo-wide `cargo fmt --check` currently reports unrelated formatting drift in
other files, so this branch intentionally used targeted formatting validation.
