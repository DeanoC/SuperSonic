"""Golden-prompt diff: score (model, quant) generated text against the BF16 reference."""
from __future__ import annotations
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROMPTS_PATH = Path(__file__).parent / "golden_prompts.json"
REFERENCES_DIR = Path(__file__).parent / "golden_references"


def score_pair(reference: str, candidate: str) -> dict:
    """Compute exact_match and chrF (character-n-gram F1) between two strings."""
    em = 1.0 if reference.strip() == candidate.strip() else 0.0
    chrf = _chrf(reference, candidate, n=6)
    return {"exact_match": em, "chrf": chrf}


def aggregate_golden_results(per_prompt: list[dict], chrf_threshold: float = 0.20) -> dict:
    if not per_prompt:
        raise ValueError("no per-prompt results")
    em = sum(p["exact_match"] for p in per_prompt) / len(per_prompt)
    chrf = sum(p["chrf"] for p in per_prompt) / len(per_prompt)
    below = [p["prompt_id"] for p in per_prompt if p["chrf"] < chrf_threshold]
    return {
        "exact_match_mean": em,
        "chrf_mean": chrf,
        "below_threshold_count": len(below),
        "below_threshold_ids": below,
    }


def _chrf(ref: str, hyp: str, n: int = 6) -> float:
    """Single-order character n-gram F1. Simple baseline; no β weighting."""
    if not ref or not hyp:
        return 0.0
    ref_grams = _char_ngrams(ref, n)
    hyp_grams = _char_ngrams(hyp, n)
    if not ref_grams or not hyp_grams:
        return 0.0
    overlap = sum((ref_grams & hyp_grams).values())
    p = overlap / max(sum(hyp_grams.values()), 1)
    r = overlap / max(sum(ref_grams.values()), 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _char_ngrams(s: str, n: int) -> Counter:
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


@dataclass
class GoldenRequest:
    binary: Path
    model: str
    model_dir: Path
    quant: str


def run_golden(req: GoldenRequest) -> dict:
    """Run all golden prompts for (model, quant). Returns a quality cell.

    First BF16 run for a model populates the reference file. Non-BF16 runs
    require the reference to exist; otherwise the cell status is 'reference_missing'.
    """
    prompts = json.loads(PROMPTS_PATH.read_text())["prompts"]
    REFERENCES_DIR.mkdir(exist_ok=True)
    ref_path = REFERENCES_DIR / f"{req.model}.json"

    if req.quant == "bf16":
        outputs = {p["id"]: _generate(req, p) for p in prompts}
        ref_path.write_text(json.dumps({"model": req.model, "outputs": outputs}, indent=2))
        return {"schema_version": 1, "model": req.model, "quant": req.quant,
                "eval": "golden", "metric": "bootstrap", "value": 1.0,
                "extras": {"prompts_recorded": len(outputs)}}

    if not ref_path.exists():
        return {"schema_version": 1, "model": req.model, "quant": req.quant,
                "eval": "golden", "metric": "exact_match_mean", "value": 0.0,
                "extras": {"status": "reference_missing",
                           "hint": f"run --quants bf16 first to populate {ref_path}"}}

    ref = json.loads(ref_path.read_text())["outputs"]
    per_prompt = []
    for p in prompts:
        cand = _generate(req, p)
        scores = score_pair(ref[p["id"]], cand)
        per_prompt.append({"prompt_id": p["id"], **scores})
    agg = aggregate_golden_results(per_prompt)
    return {"schema_version": 1, "model": req.model, "quant": req.quant,
            "eval": "golden", "metric": "exact_match_mean",
            "value": agg["exact_match_mean"],
            "extras": {**agg, "per_prompt": per_prompt}}


def _generate(req: GoldenRequest, prompt: dict) -> str:
    cmd = [str(req.binary), "--model", req.model, "--model-dir", str(req.model_dir),
           "--prompt", prompt["prompt"], "--max-new-tokens", str(prompt["max_new_tokens"])]
    cmd.extend({"bf16": [], "int4": ["--int4"], "fp8r": ["--fp8-runtime"],
                "kv-fp8": ["--kv-fp8"], "int8": ["--int8"]}[req.quant])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # Supersonic prints the prompt+generation as plain stdout lines, with
    # status lines like `[gpu] ...`, `[tokens] ...`, `[result] ...` mixed in.
    # Filter to the lines that don't start with a `[bracket]` tag — what
    # remains is the actual model output.
    lines = (res.stdout or "").splitlines()
    text_lines = [l for l in lines if not l.lstrip().startswith("[")]
    return "\n".join(text_lines).strip()
