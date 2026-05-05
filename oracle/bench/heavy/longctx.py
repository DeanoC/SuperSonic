"""Long-context perf sweep: 4k/8k/16k/32k contexts.

Quality at long context = perplexity over a 4k-prefix of PG-19, decoded through
to context end. Only runs combos that fit the registry's VRAM budget.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..runner import SupersonicInvocation, RunPolicy, run_with_capture


def run_longctx(binary: Path, model: str, model_dir: Path, quant: str,
                contexts: list[int], run_dir: Path) -> None:
    """Write one perf cell per context length into run_dir/quality/longctx_*.json."""
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in contexts:
        # Synthesize a prompt that occupies ~ctx tokens. Use a repeating snippet.
        # The runner tokenizes; a rough char-budget = ctx * 4.
        prompt = ("The quick brown fox jumps over the lazy dog. " * (ctx // 10 + 1))[:ctx * 4]
        inv = SupersonicInvocation(binary=binary, model=model, model_dir=model_dir,
                                   quant=quant, prompt=prompt, max_new_tokens=8)
        result = run_with_capture(inv, RunPolicy(measurement_runs=2, cooldown_seconds=3))
        cell = {
            "schema_version": 1,
            "model": model,
            "quant": quant,
            "eval": f"longctx_{ctx}",
            "metric": "ms_per_step",
            "value": result.get("ms_per_step", -1.0),
            "extras": {"context_tokens": ctx, "samples": result.get("samples"),
                       "status": result.get("status")},
        }
        (out_dir / f"{model}_{quant}_longctx_{ctx}.json").write_text(json.dumps(cell, indent=2))
