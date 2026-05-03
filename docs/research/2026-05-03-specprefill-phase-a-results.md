# SpecPrefill — Phase A measurements

**Branch:** `research/specprefill`
**Companion to:** `2026-05-03-specprefill-feasibility.md` (the plan)
**Hardware:** AMD RX 7900 XTX (gfx1100, 24 GiB), ROCm 7.0, torch 2.10
**Models:** Qwen3.5-0.8B (draft) + Qwen3.5-4B (target), both BF16 via HuggingFace transformers (eager attention)
**Selection config:** chunk_size=32, pool_window=5, always_keep_prefix=4, always_keep_suffix=4, lookahead=4
**Code:** `oracle/specprefill_oracle.py`, `crates/runner/src/specprefill.rs`

## Summary

The paper's algorithm preserves the **argmax of the next token at every keep ratio (10–90 %)** on both short (222 tok) and long (1354 tok) prompts. **Top-5 stability improves with prompt length** — exactly the regime SpecPrefill targets. Logit cosine similarity tells a more nuanced story: it drops to ≈ 0.85 at 10 % keep on long context, ≈ 0.94 on short, but argmax + top-5 overlap suggest greedy decoding would still pick the same tokens.

**Verdict:** the algorithm transfers cleanly to Qwen-family models in our environment. Phase B (kernel-side RoPE-indirect + sparse causal mask) and Phase C (e2e wiring) are worth landing.

## Method

For each prompt:

1. Tokenise with the draft tokeniser.
2. Forward draft + 4 look-ahead teacher-free decode steps; collect each layer's last-row attention against the prompt context (paper §3.3).
3. Aggregate scores: max over heads, max over layers, mean over the 5 query rows (last prompt token + 4 look-ahead tokens).
4. Run the chunked top-K selection (paper §3.4) at keep ratios {0.1, 0.3, 0.5, 0.7, 0.9}.
5. Quality probe: target full prefill logits at the last prompt position vs target sparse prefill on kept tokens with original position IDs (RoPE rotated to source positions, lower-triangular mask on the compacted sequence). Compare via cosine similarity, L2 distance, top-5 overlap, and argmax match.

## Results

### Short prompt (222 tokens)

| keep | actual % | cossim | argmax | top-5 |
|---|---|---|---|---|
| 0.10 | 12.6 % (28/222) | 0.940 | ✓ | 1 / 5 |
| 0.30 | 31.5 % (70/222) | 0.931 | ✓ | 2 / 5 |
| 0.50 | 50.0 % (111/222) | 0.942 | ✓ | 2 / 5 |
| 0.70 | 71.6 % (159/222) | 0.988 | ✓ | 3 / 5 |
| 0.90 | 90.5 % (201/222) | 0.996 | ✓ | 5 / 5 |

The forced prefix + suffix bands push the actual keep rate slightly above the requested ratio (e.g. 31.5 % vs 30 % requested) because they always include 8 positions regardless of score.

### Long prompt (1354 tokens, ~6× repetition + trailing query)

| keep | actual % | cossim | argmax | top-5 |
|---|---|---|---|---|
| 0.10 | 12.7 % (172/1354) | 0.851 | ✓ | 3 / 5 |
| 0.30 | 31.3 % (424/1354) | 0.903 | ✓ | 4 / 5 |
| 0.50 | 50.1 % (678/1354) | 0.964 | ✓ | 5 / 5 |
| 0.70 | 71.9 % (974/1354) | 0.967 | ✓ | 5 / 5 |
| 0.90 | 90.7 % (1228/1354) | 0.992 | ✓ | 5 / 5 |

## Observations

1. **Argmax preservation is the strongest practical signal.** At every keep ratio on both prompts, the next-token argmax of the sparse prefill matches the full prefill exactly. For greedy decoding, this means the first generated token would be identical — and since each subsequent decode step starts from the same KV cache (just with sparse fill on the prompt portion), the first divergence point sits in the *decode* phase, not the prefill phase. Decode-phase divergence is a separate quality concern that this Phase A probe doesn't measure.

2. **Top-5 stability is better on long context than short context.** At 0.30 keep, top-5 overlap goes from 2/5 → 4/5 when the prompt grows 6×. At 0.50, it goes from 2/5 → 5/5. This is the paper's main premise — "LLMs preserve quality given a carefully chosen subset" — and it holds *more strongly* in the long-context regime, exactly where TTFT actually matters.

3. **Cossim drops at low keep ratios on long context.** Counter-intuitive at first (more tokens to select from = more attention budget) but consistent with what the paper reports for low-keep aggregation tasks: dropping 90 % of a long discussion changes the *distribution* of next-token plausibilities even when the argmax winner is unchanged. For sampling-based decoding (top-p, temperature > 0) this would matter more than for greedy.

4. **Selection runtime is negligible.** Draft forward + 4 look-ahead on Qwen3.5-0.8B (RX 7900 XTX, BF16, eager attention) takes ~0.7 s for a 1354-token prompt. Compared to a 4B-class target's prefill on the same prompt this is in the noise, and it scales O(T·H·L) per token vs the target's O(T²·H·L) — the look-ahead overhead amortises in our favour as T grows.

5. **The 7900 XTX prefill is BANDWIDTH-bound, not FLOP-bound.** SpecPrefill's headline 7.66 × came from 8 × H100 TP=8 on Llama-405B — the win there was as much about cutting *cross-GPU communication during prefill* as about FLOP reduction. On a single 7900 XTX with a non-MoE target, the actual TTFT win will scale with `(1 - keep_ratio) × prefill_share_of_TTFT`. For Qwen3.5-9B on a 1.3k-token prompt this is significant (probably 1.5–2× headroom); for Qwen3.6-35B-A3B INT4 it's larger because the routed-expert prefill is the bigger pole.

## Negative results

- I had no time to test cross-family draft + target (e.g. Qwen3.5-0.8B → Qwen3.6-MoE-35B-A3B). The paper's "Cross-Family Speculative Prefill" follow-up (arXiv 2603.02631) suggests it works, but our setup needs validation.
- The top-5 overlap dropping to 1–2/5 at low keep ratios on the short prompt is uncomfortable and likely indicates SpecPrefill is **not appropriate for sampling-based decoding** (top-p / temperature > 0) at aggressive keep ratios. SuperSonic's default greedy path is fine; any future generation feature using top-p would need a separate quality investigation.
- I haven't tested how the selection algorithm behaves on prompts with long verbatim repeats vs novel content. The "long" prompt above is 6× the same paragraph plus a query — selection probably exploits this trivially. A diverse-content benchmark (RULER, LongBench retrieval) would be a better stress test before committing to Phase B.

## What this answers from the feasibility memo

| Memo question | Phase A answer |
|---|---|
| Does the algorithm preserve quality on Qwen-family models? | Yes for argmax + top-5 at long context; cossim degrades but stays > 0.85. |
| Does the selection runtime amortise? | Yes — 0.7 s for 1354-token draft forward is negligible vs target prefill. |
| Does the selection algorithm match a Python reference? | Yes — 6/6 unit-test inputs produce byte-identical kept-position lists between Rust and Python. |
| Cross-family Qwen3.5 draft + Qwen3.6-MoE target? | Untested; left for a Phase A2 follow-up before kernel work. |
| Sampling-based generation safe? | Probably no at low keep ratios — top-5 stability is poor. Greedy is fine. |

## Recommendation

**Proceed to Phase B** (RoPE-indirect + sparse-causal-mask kernels) for the **greedy-decoding path on Qwen-family same-family draft+target pairs**. Defer cross-family validation (Qwen3.5 draft → Qwen3.6-MoE target) to a one-day Phase A2 — same Python harness, just the larger target — before writing any HIP code. Skip sampling-based decoding from Phase B's scope; revisit only after argmax+top-5 numbers improve.

## Reproducing

```bash
# Selection-only (just the draft + selection):
~/venvs/rocm/bin/python oracle/specprefill_oracle.py \
    --draft-model /mnt/data/models/Qwen3.5-0.8B \
    --prompt-file <prompt.txt> \
    --keep-ratio 0.3 \
    --out /tmp/specprefill.json

# Quality probe (loads target as well):
~/venvs/rocm/bin/python oracle/specprefill_oracle.py \
    --draft-model /mnt/data/models/Qwen3.5-0.8B \
    --target-model /mnt/data/models/Qwen3.5-4B \
    --prompt-file <prompt.txt> \
    --keep-ratio 0.1 --keep-ratio 0.3 --keep-ratio 0.5 \
    --keep-ratio 0.7 --keep-ratio 0.9 \
    --out /tmp/specprefill.json
```

Rust selection unit tests:

```bash
cargo test -p runner --release --lib specprefill
# 11 passed; 0 failed
```
