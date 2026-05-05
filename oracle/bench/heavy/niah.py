"""NIAH (needle-in-a-haystack) for non-Llama (model, quant) combos.

Adapts oracle/arxiv_v1_smoke.py — which today only runs against the CUDA Llama
lane via `--certified-kv` — to drive any (model, quant) on HIP. Phase 1 wires
this for two combos: (qwen3.5-9b, int4) and (gemma4-e4b, int4).
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path

# Subtasks recorded by the existing harness. Fail-safe subset for Phase 1.
SUBTASKS = ["niah_single", "niah_multikey", "niah_multiquery"]


def run_niah(binary: Path, model: str, model_dir: Path, quant: str,
             contexts: list[int], run_dir: Path) -> None:
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        for subtask in SUBTASKS:
            score = _run_one(binary, model, model_dir, quant, subtask, ctx)
            cell = {
                "schema_version": 1,
                "model": model,
                "quant": quant,
                "eval": f"niah_{subtask}_{ctx}",
                "metric": "score",
                "value": score,
                "extras": {"context_tokens": ctx, "subtask": subtask},
            }
            (out_dir / f"{model}_{quant}_niah_{subtask}_{ctx}.json").write_text(
                json.dumps(cell, indent=2))


_SCORE_RE = re.compile(r"\[niah_score\]\s+([0-9.]+)", re.MULTILINE)


def _run_one(binary: Path, model: str, model_dir: Path, quant: str,
             subtask: str, ctx: int) -> float:
    """Drive supersonic with a NIAH prompt of `ctx` tokens and parse the score line.

    The runner must emit `[niah_score] <float>`; if it doesn't yet, this function
    falls back to invoking the existing oracle/arxiv_v1_smoke.py with --model
    pointing at this (model, quant) pair. The harness then computes score using
    its own scorer logic.
    """
    # TODO at execution time: confirm whether to wire in-runner scoring (preferred)
    # or shell out to arxiv_v1_smoke.py with broader --model support added (more code,
    # but reuses an existing scorer). Both are valid; pick the lower-risk path during
    # implementation based on whether the runner already exposes per-step logits.
    cmd = ["python", "oracle/arxiv_v1_smoke.py",
           "--model", model, "--model-dir", str(model_dir),
           "--quant", quant, "--subtask", subtask, "--context", str(ctx)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (res.stdout or "") + (res.stderr or "")
    m = _SCORE_RE.search(out)
    return float(m.group(1)) if m else 0.0
