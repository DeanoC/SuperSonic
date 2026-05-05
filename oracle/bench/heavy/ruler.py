"""RULER long-context benchmark for two designated combos in Phase 1.

RULER is the standard long-context evaluation suite (Hsieh et al.). This adapter
wraps the upstream RULER scripts and writes per-(task, context) quality cells.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

# Conservative Phase 1 subset. Full RULER has 13 tasks; we ship the four
# most-cited ones first.
RULER_TASKS = ["niah_single_1", "niah_multikey_1", "vt", "qa_1"]


def run_ruler(binary: Path, model: str, model_dir: Path, quant: str,
              contexts: list[int], run_dir: Path) -> None:
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        for task in RULER_TASKS:
            score = _run_one(binary, model, model_dir, quant, task, ctx)
            cell = {
                "schema_version": 1,
                "model": model,
                "quant": quant,
                "eval": f"ruler_{task}_{ctx}",
                "metric": "score",
                "value": score,
                "extras": {"task": task, "context_tokens": ctx},
            }
            (out_dir / f"{model}_{quant}_ruler_{task}_{ctx}.json").write_text(
                json.dumps(cell, indent=2))


def _run_one(binary: Path, model: str, model_dir: Path, quant: str,
             task: str, ctx: int) -> float:
    """Invoke supersonic on a RULER task instance, parse `[ruler_score] <float>`.

    Like NIAH (Task 24), this expects either:
    (a) the runner to emit a `[ruler_score]` line directly (preferred), or
    (b) a Python-side scorer that consumes the runner's generated output and
        a stored RULER target.

    For Phase 1, default to path (b): generate the RULER instance via the upstream
    RULER repo (https://github.com/NVIDIA/RULER), run supersonic on the prompt,
    and score with the upstream scorer. This module's responsibility is the
    plumbing, not the scorer.
    """
    # TODO at execution time: vendor the RULER instance generator + scorer into
    # oracle/bench/heavy/ruler_data.py, OR shell out to a pinned RULER checkout
    # under tools/external/ruler/. Pick the lower-risk path during implementation.
    return 0.0  # placeholder until plumbing decision is made; cell still records the attempt
