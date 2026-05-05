"""Wrap lm-evaluation-harness for HellaSwag, ARC-easy, MMLU subset.

Uses the harness's HF model adapter (lm_eval --model hf), pointing at the
SAFETENSORS dir for the model. The HIP `supersonic` runtime is NOT in the
loop here — this measures the underlying-weights-as-loaded-by-HF quality,
which is the apples-to-apples reference against published scores.

For SuperSonic-quant quality (e.g. INT4 quality through OUR INT4 path), the
golden-prompt diff (Task 16) and perplexity (Task 15) are the right tools.
This task intentionally measures the HF reference for the same model, so
the gap "HF BF16 ↔ SuperSonic BF16" is bounded and visible.
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path

DEFAULT_TASKS = ["hellaswag", "arc_easy", "mmlu_high_school_european_history"]


def run_lm_eval(model: str, model_dir: Path, quant: str,
                tasks: list[str], run_dir: Path) -> None:
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["lm_eval", "--model", "hf",
           "--model_args", f"pretrained={model_dir},dtype=bfloat16",
           "--tasks", ",".join(tasks),
           "--batch_size", "1",
           "--output_path", str(out_dir / f"{model}_{quant}_lm_eval_raw.json")]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        cell = {"schema_version": 1, "model": model, "quant": quant,
                "eval": "lm_eval", "metric": "error", "value": 0.0,
                "extras": {"stderr_tail": (res.stderr or "")[-2000:]}}
        (out_dir / f"{model}_{quant}_lm_eval.json").write_text(json.dumps(cell, indent=2))
        return

    raw = json.loads((out_dir / f"{model}_{quant}_lm_eval_raw.json").read_text())
    for task in tasks:
        score = _extract_task_score(raw, task)
        cell = {"schema_version": 1, "model": model, "quant": quant,
                "eval": f"lm_eval_{task}", "metric": "acc", "value": score,
                "extras": {"task": task, "harness": "lm-evaluation-harness"}}
        (out_dir / f"{model}_{quant}_lm_eval_{task}.json").write_text(json.dumps(cell, indent=2))


def _extract_task_score(raw: dict, task: str) -> float:
    # lm-eval-harness output schema varies by version. Common location:
    # raw["results"][task]["acc"] or raw["results"][task]["acc,none"].
    results = raw.get("results", {}).get(task, {})
    for key in ("acc", "acc,none", "exact_match", "exact_match,none"):
        if key in results:
            return float(results[key])
    return 0.0
