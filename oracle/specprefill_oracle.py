#!/usr/bin/env python3
"""
Python reference implementation for SpecPrefill (arXiv 2502.02789).

Phase A research probe: end-to-end run of the paper's algorithm using
HuggingFace transformers, on Qwen-family draft+target pairs that fit the
gfx1100 24 GiB box. Three jobs:

  1. Reference for `crates/runner/src/specprefill.rs` — emit the same
     selection's kept positions on the same input importance scores so the
     Rust unit tests can spot any drift.
  2. End-to-end quality probe: cossim of last-step prefill logits between
     full prefill and sparse prefill (target running on kept tokens with
     original position IDs). Answers the only real "does this work for
     our models on our prompts" question before any HIP kernel work.
  3. Dump the kept-position list + scores so the Rust dryrun CLI (Phase A
     follow-up) can parse and display them without re-running torch.

What this script does NOT do:

  - Run the SuperSonic kernels. The selection runs through HF
    transformers; SuperSonic itself is unchanged on this branch.
  - Decode after the sparse prefill. We only measure last-step prefill
    logit cossim; full-generation parity is a Phase C concern after the
    sparse-attention kernels land.

Algorithm (paper §3.4):

  1. Run draft forward + N look-ahead decode steps on the prompt.
  2. Cache per-step attention scores for each look-ahead query row's
     attention against the prompt context.
  3. Aggregate: max over (layers, heads), mean over look-ahead steps →
     scalar importance per prompt token.
  4. 1-D average-pool importance with a small odd window.
  5. Walk the prompt in fixed-size chunks; in each chunk pick the top
     `ceil(keep_ratio * chunk_len)` tokens by smoothed score.
  6. Force-keep a prefix band (BOS + system) and suffix band (final query
     anchor) regardless of score.
  7. Forward target on the kept tokens only, with original position IDs
     (RoPE rotates each token by its source-prompt index, not the
     compacted slot). Compare to full-prefill logits.

Usage examples:

  # Selection-only dump (no quality measurement, smaller model needed):
  python3 oracle/specprefill_oracle.py \\
      --draft-model Qwen/Qwen3.5-0.8B \\
      --prompt-file prompts/example.txt \\
      --keep-ratio 0.3 \\
      --out /tmp/specprefill.json

  # Full quality probe (loads target as well):
  python3 oracle/specprefill_oracle.py \\
      --draft-model Qwen/Qwen3.5-0.8B \\
      --target-model Qwen/Qwen3.5-9B \\
      --prompt-file prompts/example.txt \\
      --keep-ratio 0.3 --keep-ratio 0.5 --keep-ratio 0.9 \\
      --out /tmp/specprefill.json

The script is BF16-only; INT4 / FP8 SuperSonic bakes are out of scope for
the Phase A probe (we want clean reference numbers to compare against
later, not double-quantised noise).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Selection — must stay byte-identical to crates/runner/src/specprefill.rs.
# Any divergence here breaks the Phase A parity guarantee.
# ---------------------------------------------------------------------------

def avg_pool_1d_same(scores: torch.Tensor, window: int) -> torch.Tensor:
    """1-D average pool with shrinking-boundary 'same' padding.

    Equivalent to F.avg_pool1d(..., count_include_pad=False) with the
    appropriate "same" padding length. Matches the Rust reference's
    `avg_pool_1d_same`. Window must be odd.
    """
    if window % 2 != 1:
        raise ValueError(f"window must be odd, got {window}")
    n = scores.numel()
    if window <= 1 or n == 0:
        return scores.clone()
    half = window // 2
    out = torch.empty_like(scores)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = scores[lo:hi].mean()
    return out


def select_kept_positions(
    scores: torch.Tensor,
    keep_ratio: float,
    chunk_size: int,
    pool_window: int,
    always_keep_prefix: int,
    always_keep_suffix: int,
) -> list[int]:
    """Returns sorted unique kept indices. Mirrors the Rust impl exactly,
    including the deterministic tie-break (ascending index) inside each
    chunk."""
    t = scores.numel()
    if t == 0:
        return []
    keep = 0.0 if math.isnan(keep_ratio) else max(0.0, min(1.0, keep_ratio))
    smoothed = avg_pool_1d_same(scores.float(), pool_window).cpu().tolist()
    kept = [False] * t
    prefix_len = min(always_keep_prefix, t)
    suffix_len = min(always_keep_suffix, t - prefix_len)
    for i in range(prefix_len):
        kept[i] = True
    for i in range(t - suffix_len, t):
        kept[i] = True
    chunk_size = max(1, chunk_size)
    start = 0
    while start < t:
        end = min(start + chunk_size, t)
        chunk_len = end - start
        target = min(chunk_len, math.ceil(keep * chunk_len))
        if target == 0:
            start = end
            continue
        # Descending by score, ascending by index on ties.
        order = sorted(
            range(start, end), key=lambda i: (-smoothed[i], i)
        )
        for idx in order[:target]:
            kept[idx] = True
        start = end
    return [i for i in range(t) if kept[i]]


# ---------------------------------------------------------------------------
# Importance scoring — calls the draft, aggregates attention, returns the
# per-prompt-token importance vector. The formula matches paper §3.3.
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_token_importance(
    draft_model,
    input_ids: torch.Tensor,            # [1, T]
    lookahead: int,
    device: torch.device,
) -> torch.Tensor:                       # [T] float32 (CPU)
    """Run the draft forward + N teacher-free look-ahead steps; aggregate
    last-row attention scores against the prompt context.

    Aggregation: max over (layer, head), mean over look-ahead steps.
    The first attention row we collect is the LAST prompt token's
    self-attention to the prompt — i.e. lookahead=0 already gives a
    score signal even before any decode steps."""
    T = input_ids.shape[1]
    scores_per_step: list[torch.Tensor] = []  # each [layers, heads, T]

    # Step 0: prompt forward with attentions.
    out = draft_model(
        input_ids=input_ids,
        output_attentions=True,
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values
    # Last-row attention against prompt: out.attentions is a tuple of
    # [B, H, S_q, S_k] tensors per layer. S_q = S_k = T on the prompt
    # forward; we want the LAST query row attending to all keys.
    layer_rows = []
    for attn in out.attentions:
        # attn: [1, H, T, T] — take row T-1 (last query) over all T keys.
        layer_rows.append(attn[0, :, -1, :])  # [H, T]
    scores_per_step.append(torch.stack(layer_rows, dim=0))  # [layers, H, T]

    # Pick the next-token argmax for each lookahead step (teacher-free
    # decoding — the paper does the same; the model's own predicted
    # tokens drive the look-ahead query rows).
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]

    for _ in range(lookahead):
        out = draft_model(
            input_ids=next_id,
            past_key_values=past,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        # In incremental mode each layer's attn has shape [1, H, 1, S_k]
        # where S_k now includes prompt + already-decoded look-ahead
        # tokens. We only score against the original prompt (first T
        # keys); the look-ahead-to-look-ahead attention is irrelevant
        # for importance.
        layer_rows = []
        for attn in out.attentions:
            layer_rows.append(attn[0, :, 0, :T])  # [H, T]
        scores_per_step.append(torch.stack(layer_rows, dim=0))
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Stack and aggregate. shape: [steps, layers, H, T]
    stacked = torch.stack(scores_per_step, dim=0).float()
    # max over heads → [steps, layers, T]; max over layers → [steps, T];
    # mean over look-ahead steps → [T].
    importance = stacked.amax(dim=2).amax(dim=1).mean(dim=0)
    return importance.cpu()


# ---------------------------------------------------------------------------
# Quality probe — full vs sparse prefill on the target. Only runs when a
# target model is provided.
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_sparse_quality(
    target_model,
    full_input_ids: torch.Tensor,       # [1, T]
    kept_positions: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Compare last-step prefill logits between (a) target on full prompt
    and (b) target on kept tokens with original position IDs.

    Cossim is the headline number; we also report L2 distance and the
    top-5 token agreement so a single 0.999-cossim doesn't hide a head
    that's stale on the actual argmax."""
    full_logits = target_model(
        input_ids=full_input_ids,
        return_dict=True,
    ).logits[0, -1, :].float()  # [V]

    sparse_ids = full_input_ids[:, kept_positions]  # [1, k]
    pos_ids = torch.tensor(
        [kept_positions], dtype=torch.long, device=device,
    )                                                # [1, k]
    # 4D causal mask on the COMPACTED sequence — kept tokens see all
    # earlier kept tokens. Position IDs separately drive RoPE; the mask
    # itself is just lower-triangular over the compacted length.
    k = len(kept_positions)
    mask = torch.ones(k, k, dtype=torch.bool, device=device).tril()
    # transformers expects an additive float mask of shape
    # [1, 1, k, k] with 0 for visible and -inf for masked.
    additive = torch.zeros(1, 1, k, k, dtype=dtype, device=device)
    additive.masked_fill_(~mask.view(1, 1, k, k), float("-inf"))
    sparse_logits = target_model(
        input_ids=sparse_ids,
        position_ids=pos_ids,
        attention_mask=additive,
        return_dict=True,
    ).logits[0, -1, :].float()

    cos = F.cosine_similarity(
        full_logits.unsqueeze(0),
        sparse_logits.unsqueeze(0),
        dim=-1,
    ).item()
    l2 = (full_logits - sparse_logits).norm().item()
    full_top5 = full_logits.topk(5).indices.tolist()
    sparse_top5 = sparse_logits.topk(5).indices.tolist()
    top5_overlap = len(set(full_top5) & set(sparse_top5))
    argmax_match = int(full_logits.argmax().item() == sparse_logits.argmax().item())
    return {
        "cosine_similarity": cos,
        "l2_distance": l2,
        "top5_overlap": top5_overlap,
        "argmax_match": argmax_match,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft-model", required=True,
                   help="HF id or local dir for the speculator (e.g. Qwen/Qwen3.5-0.8B).")
    p.add_argument("--target-model", default=None,
                   help="HF id or local dir for the target. If omitted, the script "
                        "runs in selection-only mode (no quality probe).")
    p.add_argument("--prompt-file", required=True, type=Path,
                   help="UTF-8 file with the prompt text.")
    p.add_argument("--keep-ratio", type=float, action="append", default=None,
                   help="One or more keep ratios to evaluate (e.g. --keep-ratio 0.3 "
                        "--keep-ratio 0.5). Default sweep: 0.1 0.3 0.5 0.7 0.9.")
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--pool-window", type=int, default=5)
    p.add_argument("--always-keep-prefix", type=int, default=4)
    p.add_argument("--always-keep-suffix", type=int, default=4)
    p.add_argument("--lookahead", type=int, default=4,
                   help="N look-ahead decode steps on the draft (paper default 4).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", required=True, type=Path,
                   help="JSON output path.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    keep_ratios = args.keep_ratio or [0.1, 0.3, 0.5, 0.7, 0.9]
    if any(not (0.0 <= r <= 1.0) for r in keep_ratios):
        log(f"[error] keep ratios must be in [0, 1]; got {keep_ratios}")
        return 2

    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        log("[error] empty prompt file")
        return 2

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    log(f"[draft] loading {args.draft_model} ({args.dtype}) on {device}…")
    t0 = time.time()
    draft_tok = AutoTokenizer.from_pretrained(args.draft_model)
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=dtype,
        attn_implementation="eager",  # eager so output_attentions works.
    ).to(device).eval()
    log(f"[draft] loaded in {time.time() - t0:.1f}s")

    input_ids = draft_tok(prompt, return_tensors="pt").input_ids.to(device)
    T = input_ids.shape[1]
    log(f"[prompt] {T} tokens after BPE")

    log(f"[score] running draft forward + {args.lookahead} look-ahead steps…")
    t0 = time.time()
    importance = score_token_importance(draft, input_ids, args.lookahead, device)
    log(f"[score] done in {time.time() - t0:.1f}s")

    # Selection at every requested keep ratio.
    selections: dict[str, list[int]] = {}
    for r in keep_ratios:
        kept = select_kept_positions(
            importance, r, args.chunk_size, args.pool_window,
            args.always_keep_prefix, args.always_keep_suffix,
        )
        selections[f"{r:.2f}"] = kept
        log(f"[select] keep_ratio={r:.2f}: kept {len(kept)}/{T} "
            f"({100.0 * len(kept) / T:.1f}%)")

    # Free draft before loading target.
    del draft
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Quality probe.
    quality: dict[str, dict[str, float]] = {}
    if args.target_model is not None:
        log(f"[target] loading {args.target_model} ({args.dtype}) on {device}…")
        t0 = time.time()
        target_tok = AutoTokenizer.from_pretrained(args.target_model)
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=dtype,
        ).to(device).eval()
        log(f"[target] loaded in {time.time() - t0:.1f}s")

        # Re-tokenise with the target's tokenizer if it differs (paper's
        # cross-family extension; for same-family Qwen3.5 the BPE is shared
        # and this is a no-op).
        target_input_ids = target_tok(prompt, return_tensors="pt").input_ids.to(device)
        if target_input_ids.shape != input_ids.shape:
            log("[quality] draft and target tokenisers disagree on length; "
                "selection mapped index-wise (cross-family runs are research-grade).")

        for r_str, kept in selections.items():
            log(f"[quality] keep_ratio={r_str}…")
            t0 = time.time()
            metrics = measure_sparse_quality(
                target, target_input_ids, kept, device, dtype,
            )
            metrics["wall_seconds"] = time.time() - t0
            quality[r_str] = metrics
            log(f"[quality] keep_ratio={r_str}: cossim={metrics['cosine_similarity']:.6f} "
                f"argmax_match={metrics['argmax_match']} "
                f"top5_overlap={metrics['top5_overlap']}/5")

    out: dict[str, Any] = {
        "schema": "specprefill-oracle-v1",
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "prompt_token_count": T,
        "lookahead": args.lookahead,
        "config": {
            "chunk_size": args.chunk_size,
            "pool_window": args.pool_window,
            "always_keep_prefix": args.always_keep_prefix,
            "always_keep_suffix": args.always_keep_suffix,
        },
        "importance_scores": importance.tolist(),
        "selections": selections,
        "quality": quality,
    }
    args.out.write_text(json.dumps(out, indent=2))
    log(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
