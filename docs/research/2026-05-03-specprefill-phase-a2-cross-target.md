# SpecPrefill — Phase A2 cross-target measurements

**Branch:** `research/specprefill`
**Companion to:** `2026-05-03-specprefill-phase-a-results.md`
**Hardware:** AMD RX 7900 XTX (gfx1100, 24 GiB), ROCm 7.0, torch 2.10
**Models:** Qwen3.5-0.8B (draft) + Qwen3.5-9B (target), both BF16, eager attention
**Selection config:** chunk_size=32, pool_window=5, always_keep_prefix=4, always_keep_suffix=4, lookahead=4

## Summary

Bumping the target from Qwen3.5-4B (Phase A) to Qwen3.5-9B preserves the
**argmax of the next token at every keep ratio (10–90 %)** on both the
short (222 tok) and long (1354 tok) prompts. Top-5 overlap is comparable
to or *better than* 4B on the short prompt; on the long prompt it dips
slightly at 0.30 (4/5 → 3/5) and 0.70 (5/5 → 4/5). Cosine similarity
**does drop noticeably at low/mid keep ratios on long prompts**: from
0.903 to 0.684 at keep=0.30, from 0.851 to 0.809 at keep=0.10. At
keep ≥ 0.50, cossim stays ≥ 0.927 on both prompts.

**Verdict:** for a **greedy-decoding** runtime (Phase C scope), the
algorithm transfers from 4B to 9B fine — same argmax, same first
generated token. The cossim regression at low keep ratios on long
prompts is real but does not change the user-visible greedy output. We
proceed to Phase C with two adjusted defaults compared to the original
plan: `keep_ratio=0.50` (not 0.30) and an integration-test bar of
`cossim ≥ 0.90` (not 0.95) with `argmax_match` as the primary gate.

The Qwen3.6-35B-A3B BF16 cross-family probe was **not run** — the
checkpoint is 86 GiB on disk, so even with the script's "load draft,
free, load target" sequence the target alone exceeds 24 GiB by ~3×. A
quantised target (FP8 or INT4 bake) would fit but is out of scope for a
Phase A2 numbers-only probe.

## Results — long prompt (1354 tokens)

| keep | actual % | 4B cossim | 9B cossim | 4B argmax | 9B argmax | 4B top-5 | 9B top-5 |
|---|---|---|---|---|---|---|---|
| 0.10 | 12.7 % | 0.851 | 0.809 | ✓ | ✓ | 3 / 5 | 3 / 5 |
| 0.30 | 31.3 % | 0.903 | **0.684** | ✓ | ✓ | 4 / 5 | 3 / 5 |
| 0.50 | 50.1 % | 0.964 | 0.927 | ✓ | ✓ | 5 / 5 | 5 / 5 |
| 0.70 | 71.9 % | 0.967 | 0.948 | ✓ | ✓ | 5 / 5 | 4 / 5 |
| 0.90 | 90.7 % | 0.992 | 0.986 | ✓ | ✓ | 5 / 5 | 5 / 5 |

The kept-position counts at every keep ratio are byte-identical to Phase
A's long-prompt run — the draft + selection are unchanged, so the
*selection* is identical and only the target's reaction to that selection
varies. This is the cleanest possible cross-target comparison.

## Results — short prompt (222 tokens)

| keep | actual % | 4B cossim | 9B cossim | 4B argmax | 9B argmax | 4B top-5 | 9B top-5 |
|---|---|---|---|---|---|---|---|
| 0.10 | 12.6 % | 0.940 | 0.954 | ✓ | ✓ | 1 / 5 | 1 / 5 |
| 0.30 | 31.5 % | 0.931 | 0.932 | ✓ | ✓ | 2 / 5 | **3 / 5** |
| 0.50 | 50.0 % | 0.942 | 0.944 | ✓ | ✓ | 2 / 5 | **3 / 5** |
| 0.70 | 71.6 % | 0.988 | 0.984 | ✓ | ✓ | 3 / 5 | **4 / 5** |
| 0.90 | 90.5 % | 0.996 | 0.993 | ✓ | ✓ | 5 / 5 | 4 / 5 |

Short-prompt cossim is essentially unchanged between 4B and 9B (max
delta 0.014). Top-5 overlap is *better* on 9B for keep ratios 0.30 /
0.50 / 0.70.

## Why long-prompt low-keep cossim drops on 9B

Hypothesis (not verified): the 9B model has more layers (32 vs 4B's
36 — wait, 9B is layers=32, 4B is also model layers, let me not
speculate without checking the configs more carefully). The mechanism
likely involves the larger model having sharper logit distributions —
a 0.5 % shift in pre-softmax logits produces a much larger cossim drop
on a high-confidence distribution than on a flatter one. The argmax
preservation across all 10 cells supports this: the *winner* token is
chosen the same way; the runner-up tail is just less stable on bigger
models when the prefill is aggressively sparsified.

This means cossim is the wrong primary metric for a greedy runtime.
Argmax match (or top-K agreement on the actual decode beam, when the
runtime supports beams) is what matters end-to-end. We keep cossim as a
secondary regression-detection signal.

## Implications for Phase C

1. **Default `keep_ratio` = 0.50** (not 0.30). At 0.50, 9B holds
   cossim ≥ 0.927, top-5 ≥ 5/5 long-prompt and ≥ 3/5 short-prompt,
   and argmax match. At 0.30 the long-prompt cossim drops below 0.7,
   which is uncomfortable as a default even though argmax is fine.

2. **Integration-test cossim bar = 0.90, not 0.95.** The 4B numbers
   in Phase A inspired a 0.95 bar but the 9B reality is 0.927 at
   keep=0.50 on the long prompt — a 0.95 bar would flake. Argmax match
   stays the primary correctness gate; cossim ≥ 0.90 is a secondary
   "did anything obvious break" check.

3. **35B-A3B BF16 cross-family is deferred.** Won't fit BF16 in 24 GiB.
   When a quantised 35B-A3B bake exists in SuperSonic's runtime path
   (FP8 or INT4 already covered by the kernel work), we can re-run this
   probe through the SuperSonic prefill rather than HF transformers —
   that gives both a cross-family signal and a real end-to-end TTFT
   measurement at once.

4. **Sampling-based decoding remains out of scope.** The cossim drop
   at 0.10/0.30 keep ratios on long prompts would translate directly
   to perceptibly different top-p sample distributions. Greedy-only
   stays the right initial scope.

## Reproducing

```bash
~/venvs/rocm/bin/python oracle/specprefill_oracle.py \
    --draft-model /mnt/data/models/Qwen3.5-0.8B \
    --target-model /mnt/data/models/Qwen3.5-9B \
    --prompt-file /tmp/specprefill_long.txt \
    --keep-ratio 0.10 --keep-ratio 0.30 --keep-ratio 0.50 \
    --keep-ratio 0.70 --keep-ratio 0.90 \
    --out /tmp/specprefill_9b_long.json

~/venvs/rocm/bin/python oracle/specprefill_oracle.py \
    --draft-model /mnt/data/models/Qwen3.5-0.8B \
    --target-model /mnt/data/models/Qwen3.5-9B \
    --prompt-file /tmp/specprefill_prompt.txt \
    --keep-ratio 0.10 --keep-ratio 0.30 --keep-ratio 0.50 \
    --keep-ratio 0.70 --keep-ratio 0.90 \
    --out /tmp/specprefill_9b_short.json
```

Wall-clock: ~2.5 min total per run (draft load 30 s, target load 6–12 s,
quality probe ~5 s × 5 ratios).
